from cgi import test
import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter

import numpy as np
import time
import os
from pprint import pprint

from model.rnvp import *
from data.electricity import *

class CCNF:
    def __init__(self, exp_id, dataset, batch_size, length, pred_len, hidden, gpu, gpu_idx=0, save_path = './checkpoints'):
        if gpu:
            self.device = torch.device('cuda', gpu_idx)
        else:
            self.device = torch.device('cpu')

        if dataset == 'electricity':
            self.data = ElectricityData(batch_size, length, pred_len)

        self.save_path = os.path.join(save_path, dataset+'.'+str(exp_id))
        self.hidden = hidden    
        self.length = length
        self.pred_len = pred_len
        self.train_loader = self.data.train_loader
        self.val_loader = self.data.val_loader
        self.test_loader = self.data.test_loader

        self.est_corr = self.data.est_corr.to(self.device)
        self.pred_mat = torch.matmul(self.est_corr[-self.pred_len:, :-self.pred_len], 
                                        self.est_corr[:-self.pred_len, :-self.pred_len].inverse())
        self.cov_cond = self.est_corr[-self.pred_len:, -self.pred_len:] - torch.matmul(self.pred_mat, self.est_corr[:-self.pred_len, -self.pred_len:])
        prior_x = distributions.MultivariateNormal(torch.zeros(self.length).to(self.device), 
                                                self.est_corr[:-self.pred_len, :-self.pred_len])
        prior_y = distributions.MultivariateNormal(torch.zeros(self.pred_len).to(self.device), self.cov_cond) 
        
        nets_x, nett_x, masks_x, nets_y, nett_y, masks_y = self.build()
        self.model = RealNVP(nets_x, nett_x, masks_x,
                                     nets_y, nett_y, masks_y, prior_x, prior_y, 
                                     self.pred_mat, self.cov_cond, self.pred_len)
        self.model = self.model.to(self.device)                 

    def build(self):
        nets_x = self.net(self.length, self.length, True)
        nett_x = self.net(self.length, self.length, False)

        nets_y = self.net(self.length+self.pred_len, self.pred_len, True)
        nett_y = self.net(self.length+self.pred_len, self.pred_len, False)

        mask1_x = [0]*(self.length // 2) + [1]*(self.length // 2)
        mask2_x = [1]*(self.length // 2) + [0]*(self.length // 2)
        mask1_y = [0]*(self.pred_len // 2) + [1]*(self.pred_len // 2)
        mask2_y = [1]*(self.pred_len // 2) + [0]*(self.pred_len // 2)
        masks_x = torch.from_numpy(np.array([mask1_x, mask2_x] * 10).astype(np.float32))
        masks_y = torch.from_numpy(np.array([mask1_y, mask2_y] * 10).astype(np.float32))

        return nets_x, nett_x, masks_x, nets_y, nett_y, masks_y
    
    def net(self, length_in, length_out, tanh):
        if tanh:
            return lambda: nn.Sequential(nn.Linear(length_in, self.hidden), 
                                        nn.LeakyReLU(), 
                                        nn.Linear(self.hidden, self.hidden), 
                                        nn.LeakyReLU(), 
                                        nn.Linear(self.hidden, length_out),
                                        nn.Tanh())
        else:
            return lambda: nn.Sequential(nn.Linear(length_in, self.hidden), 
                                        nn.LeakyReLU(), 
                                        nn.Linear(self.hidden, self.hidden), 
                                        nn.LeakyReLU(), 
                                        nn.Linear(self.hidden, length_out))
    

    def fit(self, epochs, early_stopping=None):
        if early_stopping is None:
            early_stopping = epochs
        best_loss = None
        niters = len(self.train_loader)
        optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad==True], lr=1e-4)
        count = 0
        for epoch in range(epochs):
            train_loss = []
            val_loss = []
            epoch_time = time.time()
            curr_time = time.time()
            
            for i, (_, x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                loss = -self.model.log_prob(x, y).mean()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                train_loss.append(loss.item())

                if (i+1) % 100 == 0:
                    prnt_str1 = '\titers: {0} / {1}, epoch: {2} | loss: {3:.3f}'.format((i+1), niters, epoch + 1, loss.item())
                    speed = (time.time() - curr_time) / (i+1)
                    left_time = speed * ((epochs - epoch) * niters - (i+1))
                    prnt_str2 = '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time)

                    print(prnt_str1)
                    print(prnt_str2)
                    with open(os.path.join(self.save_path, 'log_train.txt'), 'a') as f:
                        pprint(prnt_str1, f)
                        pprint(prnt_str2, f)
            
            self.model.eval()
            with torch.no_grad():
                for i, (_, x, y) in enumerate(self.val_loader):
                    x, y = x.to(self.device), y.to(self.device)
                    loss = -self.model.log_prob(x, y).mean()
                    val_loss.append(loss.item())
            self.model.train()
            
            # compute average train loss
            train_loss = np.average(train_loss)
            val_loss = np.average(val_loss)

            prnt_str = "epoch: {}, epoch time: {}, loss: {}, validation loss: {}".format(epoch+1, time.time() - curr_time, train_loss, val_loss)
            print(prnt_str)
            with open(os.path.join(self.save_path, 'log_eval.txt'), 'a') as f:
                pprint(prnt_str, f)

            # save best model
            if (best_loss is None) or (val_loss < best_loss):
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model.pt'))
                best_loss = val_loss
                count = 0
            else:
                count += 1
                if count >= early_stopping:
                    print('Early stopping...')
                    break


    def evaluate(self):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'best_model.pt')))
        self.model.eval()
        with torch.no_grad():
            rmse, mae = [], []
            for i, (k, x, y) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                _, _, _, y_pred = self.model.g(x, y)
                y, y_pred = self.data.unscale(k, y), self.data.unscale(k, y_pred)
                rmse += ((y - y_pred) ** 2).mean(dim=1).sqrt().tolist()
                mae += (y - y_pred).abs().mean(dim=1).tolist()
                if (i % 100 == 0):
                    prnt_str = "Samples processed: {0} / {1}".format(i, len(self.test_loader))
                    print(prnt_str)
                    with open(os.path.join(self.save_path, 'log_eval.txt'), 'a') as f:
                        pprint(prnt_str, f)
            prnt_str = '/n RMSE: {0:.3f}, MAE: {1:.3f}'.format(np.median(rmse), np.median(mae))
            print(prnt_str)
            with open(os.path.join(self.save_path, 'log_eval.txt'), 'a') as f:
                pprint(prnt_str, f)
    
    def getbestmodel(self):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'best_model.pt')))
        self.model.eval()
        return self.model
