import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchreparam import ReparamModule

from torchdiffeq import odeint
from infrastructure.misc import *

import time
import os
import fire

from core.models import *

class AjointMetaRegressor:
    
    def __init__(self, 
                 reparam_model, 
                 n_inner_steps, 
                 stepsize, 
                 meta_lr, 
                 meta_reg,
                 meta_batch_size,
                 imaml,
                 heun,
                 device
                ):
        
        self.device = device
        self.model = reparam_model.to(self.device)
        
        self.meta_weights = self.model.flat_param
        
        self.alpha = meta_lr
        self.meta_reg = meta_reg
        self.meta_batch_size = meta_batch_size

        self.step_size= stepsize
        self.n_inner_steps = n_inner_steps
        self.T = stepsize*n_inner_steps
        
        self.imaml=imaml
        self.heun = heun
        
        self.t_series = torch.linspace(start=0, end=self.T, steps=n_inner_steps+1).to(self.device)
        self.t_series_inv = torch.linspace(start=self.T, end=0, steps=n_inner_steps+1).to(self.device)
        
    def get_loss_functional(self, X, y):
        
        def eval_loss(vec_param):
            pred = self.model(X, flat_param=vec_param)
            
            loss = 0.5*torch.mean(torch.square(pred-y))
            return loss
        #
        
        return eval_loss
    
    
    def get_loss_functional_imaml(self, X, y, reg):
        
        def eval_loss(vec_param):
            pred = self.model(X, flat_param=vec_param)
            
            loss = 0.5*torch.mean(torch.square(pred-y)) +\
            0.5*reg*torch.sum(torch.square(vec_param-self.meta_weights))
            return loss
        #
        
        return eval_loss

    
    def get_forward_sovle_functional(self, tr_loss_fn):
        
        def forward_solve(t, vec_param):
            d_u = torch.autograd.functional.jacobian(tr_loss_fn, vec_param)
            return -d_u
        #
        
        return forward_solve

    def ajoint_euler_solve_imaml(self, t_span, u_soln, lam0, tr_loss_fn):

        assert t_span.shape[0] == u_soln.shape[0]

        num_steps = t_span.shape[0]
        step_size = (t_span[-1] - t_span[0])/(num_steps-1)

        lam_t = lam0

        hist_lam_soln = []
        hist_lam_soln.append(lam_t)

        for t in reversed(range(num_steps)):

            u_t = u_soln[t]
            
            d_lam = torch.autograd.functional.vhp(func=tr_loss_fn, inputs=u_t, v=lam_t)[1]
            
            if self.heun:
                u_t_1 = u_soln[t-1]
                lam_t_1 = lam_t + step_size*d_lam
                d_lam_1 = torch.autograd.functional.vhp(
                    func=tr_loss_fn, inputs=u_t_1, v=lam_t_1)[1]                
                lam_t = lam_t + 0.5*step_size*(d_lam+d_lam_1)
            else:     
                lam_t = lam_t + step_size*d_lam
            #

            hist_lam_soln.insert(0, lam_t)
        #

        lam_soln = torch.vstack(hist_lam_soln)

        return lam_soln
    
    
    def int_ajoint(self, t_span, soln_span):
        int_pts = []
        
        n_steps = t_span.shape[0]
        step_size = (t_span[-1] - t_span[0])/(n_steps-1)
        
        for t in range(n_steps+1):
            int_pts.append(soln_span[t,:]*self.step_size)
        #
        
        int_pts = torch.vstack(int_pts)

        return int_pts.sum(0)
   

    def meta_loss_grad(self, Xtr, ytr, Xval, yval):
        
        meta_reg = self.meta_reg
        
        u_0 = self.meta_weights
        
        train_loss_imaml = self.get_loss_functional_imaml(Xtr, ytr, reg=meta_reg)   
        
        forward_solve = self.get_forward_sovle_functional(train_loss_imaml)        
        fwd_soln = odeint(func=forward_solve, y0=u_0, t=self.t_series, method='rk4')
                
        u_T = fwd_soln[-1,:]
        
        #print(u_T)
        
        val_loss = self.get_loss_functional(Xval, yval)
        val_loss_imaml = self.get_loss_functional_imaml(Xval, yval, reg=meta_reg)
        
        val_grad_T = torch.autograd.functional.jacobian(val_loss_imaml, u_T)
        val_grad_T_2 = torch.autograd.functional.jacobian(val_loss, u_T)+meta_reg*(u_T-u_0)
        
        lam_T = -val_grad_T
        
        ajoint_soln = self.ajoint_euler_solve_imaml(
            t_span=self.t_series_inv, 
            u_soln=fwd_soln, 
            lam0=lam_T, 
            tr_loss_fn=train_loss_imaml, 
        )
        
        int_ajoint_states = self.int_ajoint(t_span=self.t_series, soln_span=ajoint_soln)
        lam_0 = ajoint_soln[0,:]      
        val_grad_loss = lam_0 + meta_reg*int_ajoint_states - meta_reg*(u_0-u_T)  

        
        return u_T, val_grad_loss

    
    def solve_meta_grad(self, b_Xtr, b_ytr, b_Xval, b_yval):
        
        buff_u_T = []
        buff_meta_grad = []
        
        for i_task in range(self.meta_batch_size):
            
            Xtr, ytr, Xval, yval = b_Xtr[i_task], b_ytr[i_task], b_Xval[i_task], b_yval[i_task]
            
            task_u_T, task_meta_grad = self.meta_loss_grad(Xtr, ytr, Xval, yval)
            
            buff_u_T.append(task_u_T)
            buff_meta_grad.append(task_meta_grad)
        #
            
        buff_u_T = torch.vstack(buff_u_T)
        buff_meta_grad = torch.vstack(buff_meta_grad)
        
        return buff_u_T, buff_meta_grad 
    
    
    def update(self, b_Xtr, b_ytr, b_Xval, b_yval, update=True):
        
        meta_reg = self.meta_reg
        
        buff_u_T = []
        buff_meta_grad = []
        
        for i_task in range(self.meta_batch_size):

            Xtr, ytr, Xval, yval = b_Xtr[i_task], b_ytr[i_task], b_Xval[i_task], b_yval[i_task]
                  
            u_T, val_grad_loss = self.meta_loss_grad(Xtr, ytr, Xval, yval)
            
            buff_u_T.append(u_T)
            buff_meta_grad.append(val_grad_loss)
        #
        
        buff_u_T = torch.vstack(buff_u_T)
        buff_meta_grad = torch.vstack(buff_meta_grad)
        
        meta_grad = buff_meta_grad.mean(0)
        
        if update:
            updated_weights = self.meta_weights + self.alpha*meta_grad
            self.model.flat_param = torch.nn.Parameter(updated_weights)
            self.meta_weights = self.model.flat_param
        #

        return self.meta_weights
    
    
    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

        


            
    
    
    