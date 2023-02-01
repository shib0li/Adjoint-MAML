import numpy as np
from sklearn import preprocessing
from infrastructure.randutils import *
import time

from torch.utils.data import Dataset

from data.funcs import *

class Synthetic(Dataset):
    
    def __init__(self, 
                 mode, 
                 batchsz, 
                 k_shot, 
                 k_query, 
                 domain, 
                 Mtr=100,
                 Mte=100,
                 Ntr=1000,
                 Nval=1000,
                 Nte=1000,
                ):
        
        self.mode = mode
        
        self.batchsz = batchsz
        self.k_shot = k_shot
        self.k_query = k_query
        
        self.domain = domain
        
        print('shuffle %s DB :%s, b:%d, %d-shot, %d-query' % (
        domain, mode, batchsz, k_shot, k_query))
        
        self.Mtr=Mtr
        self.Mte= Mte
        self.Ntr=Ntr
        self.Nval=Nval
        self.Nte=Nte
        
        self.tasks_dataset = MetaSynFuncsMAML(
            domain=self.domain, 
            Mtr=self.Mtr, 
            Mte=self.Mte, 
            Ntr=self.Ntr, 
            Nval=self.Nval, 
            Nte=self.Nte,
        )
        
        self.create_batch(batchsz)
        
        
    def create_batch(self, batchsz, perm_seed=1):
        
        if self.mode=='train':
            ntasks = self.Mtr
        else:
            ntasks = self.Mte
        
        perm = []
        for n in range(batchsz//ntasks):
            perm.append(generate_permutation_sequence(N=ntasks, seed=perm_seed+n))
        #

        tail_seq_len = batchsz-ntasks*(batchsz//ntasks)
        tail_seq = generate_permutation_sequence(N=ntasks, seed=perm_seed+batchsz//ntasks+1)[:tail_seq_len]
        perm.append(tail_seq)

        self.batch_ifunc_seq = np.concatenate(perm)
        
        self.batch_support_idx = []
        self.batch_query_idx = []
        
        for i in range(batchsz):
            idx_tr = generate_random_choice(a=self.Ntr, N=self.k_shot, seed=i*2)
            idx_val = generate_random_choice(a=self.Nval, N=self.k_query, seed=i*2+1)
            
            self.batch_support_idx.append(idx_tr)
            self.batch_query_idx.append(idx_val)
        #
        
        assert  self.batch_ifunc_seq.size == len(self.batch_support_idx)

        
    #
    
    def __getitem__(self, index):
        
        idx_tr = self.batch_support_idx[index]
        idx_val = self.batch_query_idx[index]

        Xtr, ytr, Xval, yval, Xte, yte = self.tasks_dataset.get_data(
            ifunc=self.batch_ifunc_seq[index], 
            train=True if self.mode=='train' else False
        )
        
        support_X = Xtr[idx_tr, :]
        support_y = ytr[idx_tr, :]
        
        query_X = Xval[idx_val, :]
        query_y = yval[idx_val, :]
        
        test_X = Xte
        test_y = yte
        
        return support_X, support_y, query_X, query_y, test_X, test_y

    def __len__(self,):
        return self.batchsz
    
    