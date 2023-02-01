import os, sys
import pickle
import logging
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


def create_path(path): 
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        #
        print("Directory '%s' created successfully" % (path))
    except OSError as error:
        print("Directory '%s' can not be created" % (path))
    #
    
def get_logger(logpath, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger
    
def cprint(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()
    
    
def eval_metric_acc(model, X, y):
    N = X.shape[0]
    with torch.no_grad():
        logits = model(X)
        pred = F.softmax(logits, dim=1).argmax(1)
        correct = torch.eq(pred, y).sum().item()
        acc = correct/N
    #
    return acc


def test_per_task(model, Xtr, ytr, Xte, yte, max_epochs):
    
    hist_tr_acc = []
    hist_te_acc = []
    
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    
    for i in range(max_epochs):

        pred_logits = model(Xtr)
        loss = F.cross_entropy(pred_logits, ytr)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_tr = eval_metric_acc(model, Xtr, ytr)
        acc_te = eval_metric_acc(model, Xte, yte)

        hist_tr_acc.append(acc_tr)
        hist_te_acc.append(acc_te)
    #
    
    hist_tr_acc = np.array(hist_tr_acc)
    hist_te_acc = np.array(hist_te_acc)
    
    return hist_tr_acc, hist_te_acc


def eval_metric_rmse(model, X, y):
    with torch.no_grad():
        pred = model(X)
        assert pred.shape == y.shape
        rmse = torch.sqrt(torch.mean(torch.square(pred-y)))
        return rmse.item()

def test_per_task_regression(model, Xtr, ytr, Xte, yte, max_epochs, opt='adam'):
    
    hist_tr_rmse = []
    hist_te_rmse = []
    
    loss_fn = nn.MSELoss()
    
    if opt=='adam':
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    elif opt=='sgd':
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    #
    
    for i in range(max_epochs):

        pred = model(Xtr)
        loss = loss_fn(pred, ytr)
        
        #cprint('g', loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i%10 == 0:
            rmse_tr = eval_metric_rmse(model, Xtr, ytr)
            rmse_te = eval_metric_rmse(model, Xte, yte)

            hist_tr_rmse.append(rmse_tr)
            hist_te_rmse.append(rmse_te)
        #
    #
    
    hist_tr_rmse = np.array(hist_tr_rmse)
    hist_te_rmse = np.array(hist_te_rmse)
    
    return hist_tr_rmse, hist_te_rmse


