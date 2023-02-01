import numpy as np
import torch
from sklearn import preprocessing

from infrastructure.randutils import *

class FuncBounds:
    input_bounds = {
        'Alpine2D': [[-10,10],[-10,10]],
        'CosMixture': [[-1.0,1.0]],
        'CosMixture2D': [[-1.0,1.0],[-1.0,1.0]],
    }
    param_bounds = {
        'Alpine2D': [[-5,5], [-5,5]],
        'CosMixture': [[0.1,1.0], [0.5, 2.0], [3.0,6.0]],
        'CosMixture2D': [[0.1,1.0], [0.5, 2.0], [3.0,6.0]],
    }
    
class FuncDims:
    input_dims = {
        'Alpine2D': 2,
        'CosMixture': 1,
        'CosMixture2D':2,
        'Jester': 100,
        'MovieLens1M': 100,
        'MovieLens100K': 100,
    }
    output_dims = {
        'Alpine2D': 1,
        'CosMixture': 1,
        'CosMixture2D':1,
        'Jester': 1,
        'MovieLens1M': 1,
        'MovieLens100K': 1,
    }
    param_dims = {
        'Alpine2D': 2,
        'CosMixture': 3,
        'CosMixture2D':3,
    }

class Alpine2D():
    def __init__(self, param):

        self.param = param
        self.k1 = param[0]
        self.k2 = param[0]
        
    def query(self, X):
        s1 = self.k1*np.pi/12
        s2 = self.k2*np.pi/12
        
        x1 = X[:,0].reshape([-1,1])
        x2 = X[:,1].reshape([-1,1])
        
        y1 = np.abs(x1*np.sin(x1+np.pi+s1) + x1/10.0)
        y2 = np.abs(x2*np.sin(x2+np.pi+s2) + x2/10.0)
        
        y = y1+y2
        
        return y
    
class CosMixture():
    def __init__(self, param):

        self.param = param
        self.c = param[0]
        self.phase = param[1]
        self.freq = param[2]
        
    def query(self, X):
        
        term1 = np.cos(self.freq*np.pi*X + self.phase).reshape([-1,1])
        term2 = np.square(X).reshape([-1,1])
        
        y = -self.c*term1 - term2
        
        return y
    
class CosMixture2D():
    def __init__(self, param):

        self.param = param
        self.c = param[0]
        self.phase = param[1]
        self.freq = param[2]
        
    def query(self, X):
        
        term1 = np.cos(self.freq*np.pi*X + self.phase).sum(1).reshape([-1,1])
        term2 = np.square(X).sum(1).reshape([-1,1])
        
        y = -self.c*term1 - term2
        
        return y

class MetaFuncs:
    def __init__(self, domain, Nfunc=10, sample_method='uniform', seed=1):
        
        self.domain = domain
        
        input_bounds = np.array(FuncBounds.input_bounds[domain])
        param_bounds = np.array(FuncBounds.param_bounds[domain])
        
        self.input_lb = input_bounds[:,0]
        self.input_ub = input_bounds[:,1]
        
        self.param_lb = param_bounds[:,0]
        self.param_ub = param_bounds[:,1]
                
        self.input_dim = FuncDims.input_dims[domain]
        self.output_dim = FuncDims.output_dims[domain]
        self.param_dim = FuncDims.param_dims[domain] 
        
        self.functional = {
            'Alpine2D': Alpine2D,
            'CosMixture': CosMixture,
            'CosMixture2D': CosMixture2D,
        }[self.domain]
        
        self.sampled_funcs_list = self._sample_funcs(Nfunc, sample_method, seed)
        
    def _dim_sanity_check_inputs(self, X):
        if X.ndim == 1 or X.ndim == 0:
            X = np.reshape(X, [-1, self.input_dim])
        #
        return X
    
    def _dim_sanity_check_outputs(self, y):
        if y.ndim == 1 or y.ndim == 0:
            y = np.reshape(y, [-1, self.output_dim])
        #
        return y
    
    
    def _sample_funcs(self, Nfunc, sample_method, seed):
        params_samples = generate_with_bounds(
            sample_method, 
            Nfunc, 
            self.param_lb, 
            self.param_ub, 
            seed=seed
        )
        
        funcs_list = []
        for i in range(Nfunc):
            funcs_list.append(self.functional(params_samples[i,:]))
        #
        
        return funcs_list
    
    def query(self, X, ifunc=0):   
        
        X = self._dim_sanity_check_inputs(X)
        
        y = self.sampled_funcs_list[ifunc].query(X)
        y = self._dim_sanity_check_outputs(y)
        
        return y
    
    
class MetaDatasets:
    
    def __init__(self, 
                 domain, 
                 Mtr=100, Mte=20,
                 Ntr=10, Nval=10, Nte=100,
                 batch_size=3,
                 task_seed=1, design_seed=2):
        
        self.domain = domain
        self.Ntr, self.Nval, self.Nte = Ntr, Nval, Nte
        self.Mtr, self.Mte = Mtr, Mte
        self.Nfunc = Mtr + Mte
        
        self.batch_size = batch_size
        self.batch_cnt = 0
        self.batch_perm = None
        
        self.task_seed, self.design_seed = task_seed, design_seed
        
        self.Xscaler = preprocessing.StandardScaler()
#         self.Xscaler = preprocessing.MinMaxScaler((-1.0, 1.0))
#         self.Xscaler = preprocessing.MinMaxScaler()
        self.yscaler = preprocessing.StandardScaler()
        
        self.tasks_Xtr_list = []
        self.tasks_ytr_list = []
        
        self.tasks_Xval_list = []
        self.tasks_yval_list = []
        
        self.tasks_Xte_list = []
        self.tasks_yte_list = []
        
        self.Xall = None
        self.yall = None
        
        self._generate_tasks()
        
        
    def _generate_tasks(self,):
        
        meta_funcs = MetaFuncs(domain=self.domain, Nfunc=self.Nfunc, seed=self.task_seed)
        
        Nall = (self.Ntr+self.Nval)*self.Nfunc
        
        Xmeta = generate_with_bounds(
            sample_method='uniform', 
            N=Nall, 
            lb=meta_funcs.input_lb, 
            ub=meta_funcs.input_ub, 
            seed=self.design_seed
        )
        
        Xall = []
        yall = []

        for ifunc in range(self.Nfunc):
            
            i_tr = ifunc*(self.Ntr+self.Nval)
            i_val = i_tr+self.Ntr
            i_end = (ifunc+1)*(self.Ntr+self.Nval)
            
            Xtr = Xmeta[i_tr:i_val, :].reshape([-1, meta_funcs.input_dim])
            Xval = Xmeta[i_val:i_end, :].reshape([-1, meta_funcs.input_dim])
            
            if meta_funcs.input_dim == 1:
                Xte = generate_with_bounds(
                    sample_method='linspace', 
                    N=self.Nte, 
                    lb=meta_funcs.input_lb,
                    ub=meta_funcs.input_ub,
                )
            elif meta_funcs.input_dim == 2:
                Xte = generate_with_bounds(
                    sample_method='meshgrid', 
                    N=self.Nte, 
                    lb=meta_funcs.input_lb,
                    ub=meta_funcs.input_ub,
                ) 
            elif meta_funcs.input_dim <= 40:
                Xte = generate_with_bounds(
                    sample_method='sobol', 
                    N=self.Nte, 
                    lb=meta_funcs.input_lb,
                    ub=meta_funcs.input_ub,
                ) 
            else:
                Xte = generate_with_bounds(
                    sample_method='lhs', 
                    N=self.Nte, 
                    lb=meta_funcs.input_lb,
                    ub=meta_funcs.input_ub,
                )
            
            ytr = meta_funcs.query(Xtr, ifunc)
            yval = meta_funcs.query(Xval, ifunc)
            yte = meta_funcs.query(Xte, ifunc)
            
            Xall.append(np.concatenate([Xtr, Xval, Xte],axis=0))
            yall.append(np.concatenate([ytr, yval, yte],axis=0))
            
            self.tasks_Xtr_list.append(Xtr)
            self.tasks_ytr_list.append(ytr)

            self.tasks_Xval_list.append(Xval)
            self.tasks_yval_list.append(yval)

            self.tasks_Xte_list.append(Xte)
            self.tasks_yte_list.append(yte)
        #
        
        self.Xall = np.concatenate(Xall, axis=0)
        self.yall = np.concatenate(yall, axis=0)
        
        self.Xscaler.fit(self.Xall)
        self.yscaler.fit(self.yall)
    #
    
    def get_batch_data(self, normalize=True, train=True, device=None, dtype=None, perm_seed=None):
        
        if train:
            if self.batch_cnt == 0:
                num_batches = int(self.Mtr/self.batch_size)

                perm = generate_permutation_sequence(N=self.Mtr, seed=perm_seed)[:num_batches*self.batch_size]
                perm = perm.reshape([num_batches, self.batch_size])

                self.batch_cnt = num_batches
                self.batch_perm = perm
            #

            func_indices = self.batch_perm[self.batch_cnt-1, :]
            self.batch_cnt = self.batch_cnt - 1
        else:
            func_indices = np.arange(self.Mte)
            
        #print(func_indices)
        
        batch_Xtr = []
        batch_Xval = []
        batch_Xte = []
        
        batch_ytr = []
        batch_yval = []
        batch_yte = []
        
        for ifunc in func_indices:
            Xtr, ytr, Xval, yval, Xte, yte = self.get_data(ifunc, normalize, train, device, dtype)
            
            batch_Xtr.append(Xtr)
            batch_Xval.append(Xval)
            batch_Xte.append(Xte)

            batch_ytr.append(ytr)
            batch_yval.append(yval)
            batch_yte.append(yte)
        #
        
        return batch_Xtr, batch_ytr, batch_Xval, batch_yval, batch_Xte, batch_yte
   
    
    def get_data(self, ifunc, normalize=True, train=True, device=None, dtype=None):
        
        if not train:
            ifunc += self.Mtr
        #
        
        Xtr = self.tasks_Xtr_list[ifunc]
        ytr = self.tasks_ytr_list[ifunc]
        
        Xval = self.tasks_Xval_list[ifunc]
        yval = self.tasks_yval_list[ifunc]
        
        Xte = self.tasks_Xte_list[ifunc]
        yte = self.tasks_yte_list[ifunc]
        
        if normalize:
            Xtr = self.Xscaler.transform(Xtr)
            Xval = self.Xscaler.transform(Xval)
            Xte = self.Xscaler.transform(Xte)
            
            ytr = self.yscaler.transform(ytr)
            yval = self.yscaler.transform(yval)
            yte = self.yscaler.transform(yte)
            
        if device is not None:
            Xtr = torch.from_numpy(Xtr).float().to(device)
            Xval = torch.from_numpy(Xval).float().to(device)
            Xte = torch.from_numpy(Xte).float().to(device)
            
            ytr = torch.from_numpy(ytr).float().to(device)
            yval = torch.from_numpy(yval).float().to(device)
            yte = torch.from_numpy(yte).float().to(device)
        #
        
        if dtype is not None:
            Xtr = Xtr.to(dtype)
            ytr = ytr.to(dtype)
            Xval = Xval.to(dtype)
            yval = yval.to(dtype)
            Xte = Xte.to(dtype)
            yte = yte.to(dtype)
        
        return Xtr, ytr, Xval, yval, Xte, yte
    
class MetaSynFuncsMAML:
    
    def __init__(self, 
                 domain, 
                 Mtr, 
                 Mte,
                 Ntr, 
                 Nval, 
                 Nte,
                 task_seed=100, 
                 design_seed=200):
        
        self.domain = domain
        self.Ntr, self.Nval, self.Nte = Ntr, Nval, Nte
        self.Mtr, self.Mte = Mtr, Mte
        self.Nfunc = Mtr + Mte
        
        self.task_seed, self.design_seed = task_seed, design_seed
        
        self.Xscaler = preprocessing.StandardScaler()
        self.yscaler = preprocessing.StandardScaler()
        
        self.tasks_Xtr_list = []
        self.tasks_ytr_list = []
        
        self.tasks_Xval_list = []
        self.tasks_yval_list = []
        
        self.tasks_Xte_list = []
        self.tasks_yte_list = []
        
        self.Xall = None
        self.yall = None
        
        self._generate_tasks()
        
        
    def _generate_tasks(self,):
        
        meta_funcs = MetaFuncs(domain=self.domain, Nfunc=self.Nfunc, seed=self.task_seed)
        
        Nall = (self.Ntr+self.Nval)*self.Nfunc
        
        Xmeta = generate_with_bounds(
            sample_method='uniform', 
            N=Nall, 
            lb=meta_funcs.input_lb, 
            ub=meta_funcs.input_ub, 
            seed=self.design_seed
        )
        
        Xall = []
        yall = []

        for ifunc in range(self.Nfunc):
            
            i_tr = ifunc*(self.Ntr+self.Nval)
            i_val = i_tr+self.Ntr
            i_end = (ifunc+1)*(self.Ntr+self.Nval)
            
            Xtr = Xmeta[i_tr:i_val, :].reshape([-1, meta_funcs.input_dim])
            Xval = Xmeta[i_val:i_end, :].reshape([-1, meta_funcs.input_dim])
            
            if meta_funcs.input_dim == 1:
                Xte = generate_with_bounds(
                    sample_method='linspace', 
                    N=self.Nte, 
                    lb=meta_funcs.input_lb,
                    ub=meta_funcs.input_ub,
                )
            elif meta_funcs.input_dim == 2:
                Xte = generate_with_bounds(
                    sample_method='meshgrid', 
                    N=self.Nte, 
                    lb=meta_funcs.input_lb,
                    ub=meta_funcs.input_ub,
                ) 
            elif meta_funcs.input_dim <= 40:
                Xte = generate_with_bounds(
                    sample_method='sobol', 
                    N=self.Nte, 
                    lb=meta_funcs.input_lb,
                    ub=meta_funcs.input_ub,
                ) 
            else:
                Xte = generate_with_bounds(
                    sample_method='lhs', 
                    N=self.Nte, 
                    lb=meta_funcs.input_lb,
                    ub=meta_funcs.input_ub,
                )
            
            ytr = meta_funcs.query(Xtr, ifunc)
            yval = meta_funcs.query(Xval, ifunc)
            yte = meta_funcs.query(Xte, ifunc)
            
            Xall.append(np.concatenate([Xtr, Xval, Xte],axis=0))
            yall.append(np.concatenate([ytr, yval, yte],axis=0))
            
            self.tasks_Xtr_list.append(Xtr)
            self.tasks_ytr_list.append(ytr)

            self.tasks_Xval_list.append(Xval)
            self.tasks_yval_list.append(yval)

            self.tasks_Xte_list.append(Xte)
            self.tasks_yte_list.append(yte)
        #
        
        self.Xall = np.concatenate(Xall, axis=0)
        self.yall = np.concatenate(yall, axis=0)
        
        self.Xscaler.fit(self.Xall)
        self.yscaler.fit(self.yall)
    #
   
    
    def get_data(self, ifunc, normalize=True, train=True, noise=0.00):
        
        if not train:
            ifunc += self.Mtr
        #
        
        Xtr = self.tasks_Xtr_list[ifunc]
        ytr = self.tasks_ytr_list[ifunc]
        
        Xval = self.tasks_Xval_list[ifunc]
        yval = self.tasks_yval_list[ifunc]
        
        Xte = self.tasks_Xte_list[ifunc]
        yte = self.tasks_yte_list[ifunc]
        
        if normalize:
            Xtr = self.Xscaler.transform(Xtr)
            Xval = self.Xscaler.transform(Xval)
            Xte = self.Xscaler.transform(Xte)
            
            ytr = self.yscaler.transform(ytr) + noise*np.random.normal(size=ytr.shape)
            yval = self.yscaler.transform(yval)+ noise*np.random.normal(size=yval.shape)
            yte = self.yscaler.transform(yte)
        #
        
        return Xtr, ytr, Xval, yval, Xte, yte
    
    
    