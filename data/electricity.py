from random import triangular
import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import itertools as it
from scipy.stats import kendalltau
from data.template import *
from utils.genrandom import *
from utils.nearcorr import *

class ElectricityData():
    def __init__(self, batch_size, length, pred_len, scale=True):
        self.length = length
        self.batch_size = batch_size
        self.pred_len = pred_len
        
        self.train_data = torch.load('./data/electricity/electricity_train.pt')
        self.val_data = torch.load('./data/electricity/electricity_val.pt')
        self.test_data = torch.load('./data/electricity/electricity_test.pt')
        self.scaler = torch.load('./data/electricity/electricity_scaler.pt')

        if scale:
            self.train_data = self.scale(self.train_data)
            self.val_data = self.scale(self.val_data)
            self.test_data = self.scale(self.test_data)

        self.est_corr = self.estimate_corr()

        train_seqs, val_seqs, test_seqs = [], [], []
        train_seqs += [[k, self.train_data[k].astype(np.float32)] for k in self.train_data.keys()]
        val_seqs += [[k, self.train_data[k].astype(np.float32)] for k in self.val_data.keys()]
        test_seqs += [[k, self.train_data[k].astype(np.float32)] for k in self.test_data.keys()]
        self.train_data = DataSet(train_seqs, self.length, self.pred_len)
        self.val_data = DataSet(val_seqs, self.length, self.pred_len)
        self.test_data = DataSet(test_seqs, self.length, self.pred_len)

        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)

    def scale(self, data):
        for k in data.keys():
            data[k] = (data[k] - self.scaler[k][0]) / (self.scaler[k][1] - self.scaler[k][0])
        return data
    
    def unscale(self, k, seq):
        out = torch.zeros_like(seq)
        for i in range(len(k)):
            k_i = int(k[i])
            seq_i = seq[i]
            out[i] = (self.scaler[k_i][1] - self.scaler[k_i][0]) * seq_i + self.scaler[k_i][0]
        return out

    def estimate_corr(self, samples=10):
        length = self.length+self.pred_len

        X = []
        for j, k in enumerate(self.train_data.keys()):
            start = random_spaced(0, len(self.train_data[k])-length, 
                                    length*10, samples).astype(int)
            end = start + length
            for i in range(len(start)):
                if (np.any(self.train_data[k][start[i]:end[i]] != 0)):
                    X.append(self.train_data[k][start[i]:end[i]])
        X = np.matrix(X)
        values= []
        print('Correlation estimation progress...')
        for i, j in it.combinations(X.T, 2):
            values.append(kendalltau(i, j)[0])
        print('Correlation estimated.')
        print('Enforcing pd for correlation matrix...')
        est_corr = np.empty((length, length)).astype(np.float32)
        iu = np.triu_indices(length, 1)
        il = np.tril_indices(length, -1)
        dg = np.diag_indices(length)
        est_corr[iu] = values
        est_corr[dg] = 1
        est_corr[il] = est_corr.T[il]
        est_corr = np.sin((est_corr * np.pi / 2))
        est_corr = nearcorr(est_corr, max_iterations=1000)
        est_corr = est_corr*0.9+np.eye(length)*0.1
        print('Pd enforced.')

        return torch.tensor(est_corr.astype(np.float32))
    