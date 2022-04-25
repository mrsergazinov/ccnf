import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools as it
from random import triangular
from scipy.stats import kendalltau

from data.template import *
from utils.genrandom import *
from utils.nearcorr import *

class ElectricityData():
    def __init__(self, batch_size, length, pred_len, scale=True):
        DATA_PATH = './data/electricity/LD2011_2014.txt'
        self.length = length
        self.batch_size = batch_size
        self.pred_len = pred_len
        self.train_data, self.test_data, self.scaler = self.load_data(DATA_PATH, scale)

        # estimate correlation matrix
        self.est_corr = self.estimate_corr()

        # create datasets and dataloaders
        train_seqs, test_seqs = [], [], []
        train_seqs += [[k, self.train_data[k].astype(np.float32)] for k in self.train_data.keys()]
        test_seqs += [[k, self.train_data[k].astype(np.float32)] for k in self.test_data.keys()]
        self.train_data = DataSet(train_seqs, self.length, self.pred_len)
        self.test_data = DataSet(test_seqs, self.length, self.pred_len)

        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)

    def scale(self, k, seqs):
        out = torch.zeros_like(seqs)
        for i in range(len(k)):
            k_i = int(k[i])
            seq_i = seqs[i]
            out[i] = (seq_i - self.scaler[k_i][0]) / (self.scaler[k_i][1] - self.scaler[k_i][0])
        return out
    
    def unscale(self, k, seqs):
        out = torch.zeros_like(seqs)
        for i in range(len(k)):
            k_i = int(k[i])
            seq_i = seqs[i]
            out[i] = (self.scaler[k_i][1] - self.scaler[k_i][0]) * seq_i + self.scaler[k_i][0]
        return out
    
    def load_data(self, path, scale):
        table = pd.read_table(path, sep=';')
        table['date'] = pd.to_datetime(table.iloc[:, 0])
        table.drop(columns=['Unnamed: 0'], inplace=True)
        # filter dates between Jan 2014 and Sep 2014
        table_train = table[(table.date >= '2014-01-01') & (table.date < '2014-09-01')].copy()
        table_test = table[(table.date >= '2014-08-25') & (table.date < '2014-09-07')].copy()

        # remove outliers: households with all zero values OR values > 10000
        data_train = np.zeros((table_train.shape[0] // 4, table_train.shape[1]-1))
        for i in range(len(data_train)):
            data_train[i, :] = np.sum(table_train.iloc[(i*4):(i*4+4), :-1].values.astype(np.float32) / 4, axis=0)

        maxx = np.max(data_train, axis=0)
        remove = np.argwhere(maxx == 0).flatten().tolist()
        remove += np.argwhere(maxx > 10000).flatten().tolist()
        table_train.drop(columns=table.columns[remove], inplace=True)
        table_test.drop(columns=table.columns[remove], inplace=True)

        # prepare time series: convert to hourly readings
        data_train = np.zeros((table_train.shape[0] // 4, table_train.shape[1]-1))
        for i in range(len(data_train)):
            data_train[i, :] = np.sum(table_train.iloc[(i*4):(i*4+4), :-1].values.astype(np.float32) / 4, axis=0)
            
        data_test = np.zeros((table_test.shape[0] // 4, table_test.shape[1]-1))
        for i in range(len(data_test)):
            data_test[i, :] = np.sum(table_test.iloc[(i*4):(i*4+4), :-1].values.astype(np.float32) / 4, axis=0)

        # normalize data: min-max scaling
        minn, maxx = None, None
        if scale:
            minn = np.min(data_train, axis=0)
            maxx = np.max(data_train, axis=0)
            data_train = (data_train - minn) / (maxx - minn)
            data_test = (data_test - minn) / (maxx - minn)
        
        # create dict of data
        dict_train, dict_test, dict_scaler = {}, {}, {}
        for i in range(data_train.shape[1]):
            dict_train[i] = data_train[:, i]
            dict_test[i] = data_test[:, i]
            if scale:
                dict_scaler[i] = [minn[i], maxx[i]]
        
        return dict_train, dict_test, dict_scaler

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
    