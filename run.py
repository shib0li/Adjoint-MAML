import numpy as np
import time
import os
import fire

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from torchreparam import ReparamModule
from torchdiffeq import odeint

from tqdm.auto import tqdm, trange

from data.funcs import *
from data.Synthetic import Synthetic

from core.meta_learner import AjointMetaRegressor
from core.models import *

from infrastructure.misc import *
from infrastructure.configs import *


def test_on_the_fly(domain, test_dataset, saved_dict_path, device, max_epochs):
    
    in_dim = FuncDims.input_dims[domain]
    out_dim = FuncDims.output_dims[domain]
    
    tasks_tr_rmse = []
    tasks_te_rmse = []

    model = get_fcnn_regressor(in_dim, out_dim, hidden_depth=2, hidden_width=32)

    reparam_model = ReparamModule(model).to(device)
    
    db_test = DataLoader(test_dataset, 1, shuffle=True, num_workers=1, pin_memory=False)
    
    for x_spt, y_spt, x_qry, y_qry, x_te, y_te in db_test:

        reparam_model.load_state_dict(torch.load(saved_dict_path))
        
        Xtr, ytr, Xte, yte = \
            x_spt[0].to(device), y_spt[0].to(device), x_te[0].to(device), y_te[0].to(device)
        
        hist_tr_rmse, hist_te_rmse = test_per_task_regression(
            reparam_model, Xtr, ytr, Xte, yte, max_epochs)
        #
        tasks_tr_rmse.append(hist_tr_rmse)
        tasks_te_rmse.append(hist_te_rmse)
    #
    
    tasks_tr_rmse = np.vstack(tasks_tr_rmse)
    tasks_te_rmse = np.vstack(tasks_te_rmse)
    
    return tasks_tr_rmse, tasks_te_rmse
    

def evaluation(**kwargs):
    
    config = AjointMAML_Config()
    config._parse(kwargs)

    state_dict_path = os.path.join(
        '__results__',
        '__dict__',
        'ajoint',
        config.domain,
        str(config.k_shot)+'shot-'+str(config.k_query)+'query',
    )
    
    res_pickle_path = os.path.join(
        '__results__',
        '__pkl__',
        'ajoint',
        config.domain,
        str(config.k_shot)+'shot-'+str(config.k_query)+'query',
    )
    
    log_path = os.path.join(
        '__results__',
        '__log__',
        'ajoint',
        config.domain,
        str(config.k_shot)+'shot-'+str(config.k_query)+'query',
    )
    
    create_path(log_path)
    
    exp_name = 'inner_n_steps_' + str(config.inner_n_steps) + '-' + \
               'inner_stepsize_' + str(config.inner_stepsize) + '-' + \
               'meta_batchsize_' + str(config.meta_batch) + '-' + \
               'meta_lr_' + str(config.meta_lr) + '-' + \
               'meta_reg_' + str(config.meta_reg)
    
    create_path(os.path.join(state_dict_path, exp_name))
    create_path(os.path.join(res_pickle_path, exp_name))
    
    logger = get_logger(logpath=os.path.join(log_path, exp_name+'.log'), displaying=config.verbose)
    logger.info(config)
    
    if config.dtype == 'float64':
        default_dtype = torch.float64
    else:
        default_dtype = torch.float32
    #
    
    torch.set_default_dtype(default_dtype)
    
    in_dim = FuncDims.input_dims[config.domain]
    out_dim = FuncDims.output_dims[config.domain]

    if config.domain == 'Jester' or config.domain == 'MovieLens1M' or config.domain == 'MovieLens100K':
        model = get_fcnn_regressor(in_dim, out_dim, hidden_depth=2, hidden_width=40)
    else:
        model = get_fcnn_regressor(in_dim, out_dim, hidden_depth=2, hidden_width=32)
    #
    
    logger.info(model)
    
    meta_learner = AjointMetaRegressor(
        reparam_model=ReparamModule(model), 
        n_inner_steps=config.inner_n_steps, 
        stepsize=config.inner_stepsize, 
        meta_lr=config.meta_lr, 
        meta_reg=config.meta_reg,
        meta_batch_size=config.meta_batch,
        imaml=config.imaml_reg,
        heun=config.heun,
        device=torch.device(config.device)
    )
    
    meta_dataset = Synthetic(
        mode='train', 
        batchsz=config.batchsize, 
        k_shot=config.k_shot, 
        k_query=config.k_query, 
        domain=config.domain, 
        Mtr=config.tr_tasks,
        Mte=config.te_tasks,
        Ntr=1000,
        Nval=1000,
        Nte=1000,
    )

    meta_dataset_test = Synthetic(
        mode='test', 
        batchsz=config.test_batchsize, 
        k_shot=config.k_shot, 
        k_query=100, 
        domain=config.domain, 
        Mtr=config.tr_tasks,
        Mte=config.te_tasks,
        Ntr=1000,
        Nval=1000,
        Nte=1000,
    )
    

    meta_steps = 0

    for epoch in trange(config.meta_epochs):
    
    
        db = DataLoader(meta_dataset, batch_size=config.meta_batch, shuffle=config.meta_shuffle_batch, 
                        num_workers=1, pin_memory=False)
        
        logger.info('---------------------------------')
        logger.info('         Meta Epoch '+str(epoch))
        logger.info('---------------------------------')
        

        for i_step, (x_spt, y_spt, x_qry, y_qry, x_te, y_te) in enumerate(db):


            x_spt, y_spt, x_qry, y_qry = \
                x_spt.to(default_dtype).to(torch.device(config.device)), \
                y_spt.to(default_dtype).to(torch.device(config.device)), \
                x_qry.to(default_dtype).to(torch.device(config.device)), \
                y_qry.to(default_dtype).to(torch.device(config.device))
            
            meta_weights_init = meta_learner.meta_weights
            
            t_start = time.time()            
            updated_weights = meta_learner.update(x_spt, y_spt, x_qry, y_qry)
            t_interval = time.time()-t_start
            
            meta_weights_updated = meta_learner.meta_weights
    
            if meta_steps % config.test_interval == 0:
                
                if config.heun:
                    dict_name = 'step'+str(meta_steps)+'_heun.dict'
                else:
                    dict_name = 'step'+str(meta_steps)+'.dict'
                #
                meta_learner.save_model(os.path.join(state_dict_path, exp_name, dict_name))
                
                tasks_tr_rmse, tasks_te_rmse = test_on_the_fly(
                                domain=config.domain, 
                                test_dataset=meta_dataset_test,
                                saved_dict_path=os.path.join(state_dict_path, exp_name, dict_name),
                                device=torch.device(config.device),
                                max_epochs=config.test_max_epochs,
                )
                logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                logger.info(' - avg train rmse:' + str( tasks_tr_rmse.mean(0)))
                logger.info(' - std train rmse:' + str( tasks_tr_rmse.std(0)))
                logger.info(' - avg test  rmse:' + str( tasks_te_rmse.mean(0)))
                logger.info(' - std test  rmse:' + str( tasks_te_rmse.std(0)))
                
                test_res = {}
                test_res['tasks_tr_rmse'] = tasks_tr_rmse
                test_res['tasks_te_rmse'] = tasks_te_rmse
                
                pickle_name = 'step'+str(meta_steps)+'.pkl'
                
                with open(os.path.join(res_pickle_path, exp_name, pickle_name), 'wb') as handle:
                    pickle.dump(test_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
                #
                
            #

            logger.info('\n(meta step'+str(meta_steps)+') takse ' + str(t_interval)+ ' secs')
            logger.info('  - meta_weights:'+str(meta_learner.meta_weights))
            
            if meta_steps > config.max_meta_updates:
                cprint('r', "Exceed maximum number of meta update, exit program...")
                logger.info("Exceed maximum number of meta update, exit program...")
                exit()
            #

            meta_steps += 1
            
        #
    #

    
if __name__=='__main__':
    fire.Fire(evaluation)
    
    
    