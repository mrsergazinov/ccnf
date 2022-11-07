from cgi import test
import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import time
import os
from pprint import pprint


from model.rnvp import *
from dataset.electricity import *
from dataset.ts_dataset import *
from dataset.utils import QuantileLoss, smape_loss, unnormalize_tensor

class CCNF:
    def __init__(self, conf):
        # set deivce
        self.conf = conf
        self.device = conf.device

        # load data
        if conf.ds_name == 'electricity':
            self.data_formatter = ElectricityFormatter()
        loader = TSDataset
        dataset_train = loader(self.conf, self.data_formatter)
        dataset_train.train()
        dataset_val = loader(self.conf, self.data_formatter)
        dataset_val.val()
        dataset_test = loader(self.conf, self.data_formatter)
        dataset_test.test()
        self.train_loader = DataLoader(
            dataset=dataset_train, batch_size=conf.batch_size,
            num_workers=conf.n_workers, shuffle=True, pin_memory=True,
        )
        self.val_loader = DataLoader(
            dataset=dataset_val, batch_size=conf.batch_size,
            num_workers=conf.n_workers, shuffle=False, pin_memory=True,
        )
        self.test_loader = DataLoader(
            dataset=dataset_test, batch_size=conf.batch_size,
            num_workers=conf.n_workers, shuffle=False, pin_memory=True,
        )
        
        # define priors
        self.est_corr = dataset_train.estimate_corr().to(self.device)
        self.pred_mat = torch.matmul(self.est_corr[self.conf.num_encoder_steps:, :self.conf.num_encoder_steps], 
                                self.est_corr[:self.conf.num_encoder_steps, :self.conf.num_encoder_steps].inverse())
        self.cov_marg = self.est_corr[:self.conf.num_encoder_steps, :self.conf.num_encoder_steps]
        self.cov_cond = self.est_corr[self.conf.num_encoder_steps:, self.conf.num_encoder_steps:] - \
                                torch.matmul(self.pred_mat, self.est_corr[:self.conf.num_encoder_steps, 
                                    self.conf.num_encoder_steps:])
        self.cov_marg = self.cov_marg.type(torch.float32)
        self.cov_cond = self.cov_cond.type(torch.float32)
        self.pred_mat = self.pred_mat.type(torch.float32)
        prior_x = distributions.MultivariateNormal(torch.zeros(self.conf.num_encoder_steps).to(self.device), 
                                self.cov_marg)
        prior_y = distributions.MultivariateNormal(torch.zeros(self.conf.num_decoder_steps).to(self.device), 
                                self.cov_cond) 
        
        # build model
        nets_x, nett_x, masks_x, nets_y, nett_y, masks_y = self.build()
        self.model = RealNVP(nets_x, nett_x, masks_x,
                                     nets_y, nett_y, masks_y, prior_x, prior_y, 
                                     self.pred_mat, self.cov_cond)
        self.model = self.model.to(self.device)      

        # init summary writer
        self.sw = SummaryWriter(self.conf.exp_log_path)  

        # loss for inference
        # TODO: add more quantiles
        self.loss = QuantileLoss([0.5])   

    def build(self):
        nets_x = self.net(self.conf.num_encoder_steps, self.conf.num_encoder_steps, True)
        nett_x = self.net(self.conf.num_encoder_steps, self.conf.num_encoder_steps, False)

        nets_y = self.net(self.conf.total_time_steps, self.conf.num_decoder_steps, True)
        nett_y = self.net(self.conf.total_time_steps, self.conf.num_decoder_steps, False)

        mask1_x = [0]*(self.conf.num_encoder_steps // 2) + [1]*(self.conf.num_encoder_steps // 2)
        mask2_x = [1]*(self.conf.num_encoder_steps // 2) + [0]*(self.conf.num_encoder_steps // 2)
        mask1_y = [0]*(self.conf.num_decoder_steps // 2) + [1]*(self.conf.num_decoder_steps // 2)
        mask2_y = [1]*(self.conf.num_decoder_steps // 2) + [0]*(self.conf.num_decoder_steps // 2)
        masks_x = torch.from_numpy(np.array([mask1_x, mask2_x] * self.conf.num_flows).astype(np.float32))
        masks_y = torch.from_numpy(np.array([mask1_y, mask2_y] * self.conf.num_flows).astype(np.float32))

        return nets_x, nett_x, masks_x, nets_y, nett_y, masks_y
    
    def net(self, length_in, length_out, tanh):
        if tanh:
            return lambda: nn.Sequential(nn.Linear(length_in, self.conf.hidden), 
                                        nn.LeakyReLU(), 
                                        nn.Linear(self.conf.hidden, self.conf.hidden), 
                                        nn.LeakyReLU(), 
                                        nn.Linear(self.conf.hidden, length_out),
                                        nn.Tanh())
        else:
            return lambda: nn.Sequential(nn.Linear(length_in, self.conf.hidden), 
                                        nn.LeakyReLU(), 
                                        nn.Linear(self.conf.hidden, self.conf.hidden), 
                                        nn.LeakyReLU(), 
                                        nn.Linear(self.conf.hidden, length_out))
    

    def fit(self):
        best_loss = None
        optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad==True], lr=self.conf.lr)
        niters = len(self.train_loader)
        count, batches = 0, 0

        for epoch in range(self.conf.num_epochs):
            train_loss = []
            val_loss = []
            epoch_time = time.time()
            curr_time = time.time()
            
            for i, (x, y, id) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)

                loss, cache = self.model.log_prob(x, y)
                loss = -loss.mean()
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.conf.max_grad_norm)
                train_loss.append(loss.item())
                optimizer.step()

                if (i+1) % 100 == 0:
                    prnt_str1 = 'iters: {0} / {1}, epoch: {2} | loss: {3:.3f}'.format((i+1), niters, epoch + 1, loss.item())
                    speed = (time.time() - curr_time) / (i+1)
                    left_time = speed * ((self.conf.num_epochs - epoch) * niters - (i+1))
                    prnt_str2 = 'speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time)

                    self.sw.add_scalar('loss/sample_loss', loss.item(), batches)
                    self.sw.add_scalar('time/left_time', left_time, batches)
                    print(prnt_str1)
                    print(prnt_str2)
                    # plot batch normalization
                    for i in range(self.conf.num_encoder_steps):
                        self.sw.add_histogram('hist/x{0}'.format(i), cache['cache_batch']['x1'][:, i], batches)
                    for i in range(self.conf.num_decoder_steps):
                        self.sw.add_histogram('hist/y{0}'.format(i), cache['cache_batch']['x2'][:, i], batches)
                    batches += 1

            self.model.eval()
            with torch.no_grad():
                for i, (x, y, id) in enumerate(self.val_loader):
                    x, y = x.to(self.device), y.to(self.device)
                    loss, cache = self.model.log_prob(x, y)
                    loss = -loss.mean()
                    val_loss.append(loss.item())
            self.model.train()
            
            # compute average train loss
            train_loss = np.average(train_loss)
            val_loss = np.average(val_loss)
            self.sw.add_scalar('loss/train_loss', train_loss, epoch)
            self.sw.add_scalar('loss/val_loss', val_loss, epoch)
            prnt_str = "epoch: {}, epoch time: {}, loss: {}, validation loss: {}".format(epoch+1, time.time() - curr_time, train_loss, val_loss)
            print(prnt_str)
            
            # save best model
            if (best_loss is None) or (val_loss < best_loss):
                torch.save(self.model.state_dict(), os.path.join(self.conf.exp_log_path, 'best_model.pt'))
                best_loss = val_loss
                count = 0
            else:
                count += 1
                if count >= self.conf.early_stopping:
                    print('Early stopping...')
                    break
        torch.save(self.model.state_dict(), os.path.join(self.conf.exp_log_path, 'last_model.pt'))

    def evaluate(self):
        self.model.load_state_dict(torch.load(os.path.join(self.conf.exp_log_path, 'best_model.pt'), 
            map_location=self.device))
        self.model.eval()

        # return self.test_loader, self.data_formatter, self.model, self.loss

        with torch.no_grad():
            p50_risk = []
            smape = []
            for i, (x, y, id) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                dummy_y = torch.zeros_like(y)
                _, _, _, y_pred = self.model.g(x, dummy_y)

                #TODO: model should predict quantiles, quantile predictions should be attached along last axis
                # below is temp. fix
                y_pred = y_pred.unsqueeze(-1)

                # compute loss
                loss, _ = self.loss(y_pred, y)
                smape.append(smape_loss(y_pred.squeeze().detach().cpu().numpy(), y.detach().cpu().numpy()))
                
                # unnormalize tensor
                target = unnormalize_tensor(self.data_formatter, y, id[0][0])
                p50_forecast = unnormalize_tensor(self.data_formatter, y_pred.squeeze(), id[0][0])
                
                # compute risk
                p50_risk.append(self.loss.numpy_normalised_quantile_loss(p50_forecast, target, 0.5))

            prnt_str = '/n SMAPE: {0:.3f} | P50 Risk: {1:.3f}'.format(np.mean(smape), np.mean(p50_risk))
            print(prnt_str)
    
    # def getbestmodel(self):
    #     self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'best_model.pt')))
    #     self.model.eval()
    #     return self.model
