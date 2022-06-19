import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter

from utils.batchnorm import *

class RealNVP(nn.Module):
    def __init__(self, nets_x1, nett_x1, mask_x1, 
                 nets_x2, nett_x2, mask_x2, prior_x1, prior_x2, 
                 pred_mat, cov_cond):
        super(RealNVP, self).__init__()
        self.pred_mat = nn.Parameter(data=pred_mat, requires_grad=False)
        self.cov_cond = nn.Parameter(data=cov_cond, requires_grad=False)

        # self.batchnorm_x1 = nn.BatchNorm2d(self.pred_mat.shape[1])
        # self.batchnorm_x2 = nn.BatchNorm2d(self.pred_mat.shape[0])
        # self.batchnorm_x1 = nn.BatchNorm1d(self.pred_mat.shape[1])
        # self.batchnorm_x2 = nn.BatchNorm1d(self.pred_mat.shape[0])
        self.batchnorm_x1 = BatchNormInv()
        self.batchnorm_x2 = BatchNormInv()

        self.prior_x1 = prior_x1
        self.prior_x2 = prior_x2
        
        self.mask_x1 = nn.Parameter(mask_x1, requires_grad=False)
        self.t1 = torch.nn.ModuleList([nett_x1() for _ in range(len(mask_x1))])
        self.s1 = torch.nn.ModuleList([nets_x1() for _ in range(len(mask_x1))])
        
        self.mask_x2 = nn.Parameter(mask_x2, requires_grad=False)
        self.t2 = torch.nn.ModuleList([nett_x2() for _ in range(len(mask_x2))])
        self.s2 = torch.nn.ModuleList([nets_x2() for _ in range(len(mask_x2))])
        
    def g(self, x1, x2):
        # send to latent and predict (with Gaussian)
        z1, z2, _, _ = self.f(x1, x2)
        z2 = torch.matmul(self.pred_mat, z1.permute(1, 0)).permute(1, 0)
        
        # z2_upper = z2 + 1.96 * torch.diag(self.cov_cond)
        # z2_lower = z2 - 1.96 * torch.diag(self.cov_cond)
        # z2 = torch.cat([z2, z2_upper, z2_lower], 0)
        
        x1_pred, x2_pred = torch.clone(z1), torch.clone(z2)
        # x1 = torch.cat([x1]*3, 0)
        
        for i in range(len(self.t1)):
            x1_ = x1_pred * self.mask_x1[i]
            s1 = self.s1[i](x1_)
            t1 = self.t1[i](x1_)
            x1_pred = x1_ + (1 - self.mask_x1[i]) * (x1_pred * torch.exp(s1) + t1)
            
        for i in range(len(self.t1)):
            x2_ = x2_pred * self.mask_x2[i]
            x2cond_ = torch.cat([x2_, x1], 1)
            s2 = self.s2[i](x2cond_)
            t2 = self.t2[i](x2cond_)
            x2_pred = x2_ + (1 - self.mask_x2[i]) * (x2_pred * torch.exp(s2) + t2)
        
        x1_pred = self.batchnorm_x1.inverse(x1_pred)
        x2_pred = self.batchnorm_x2.inverse(x2_pred)
            
        return z1, z2, x1_pred, x2_pred

    def f(self, x1, x2):
        # batch normalization
        # x1 = x1.unsqueeze(2).unsqueeze(3)
        # x2 = x2.unsqueeze(2).unsqueeze(3)
        # x1 = x1.squeeze(3).squeeze(2)
        # x2 = x2.squeeze(3).squeeze(2)
        
        x1 = self.batchnorm_x1(x1)
        x2 = self.batchnorm_x2(x2)
        # save batch normalized x1 and x2
        cache_batch = {'x1': x1.clone().detach(), 'x2': x2.clone().detach()}

        log_det_J1, z1 = x1.new_zeros(x1.shape[0]), x1
        for i in reversed(range(len(self.t1))):
            z1_ = self.mask_x1[i] * z1
            s1 = self.s1[i](z1_)
            t1 = self.t1[i](z1_)
            z1 = (1 - self.mask_x1[i]) * ((z1 - t1) * torch.exp(-s1)) + z1_
            log_det_J1 -= s1.sum(dim=1)
        
        log_det_J2, z2 = x2.new_zeros(x2.shape[0]), x2
        for i in reversed(range(len(self.t1))):
            z2_ = self.mask_x2[i] * z2
            z2cond_ = torch.cat([z2_, x1], 1)
            s2 = self.s2[i](z2cond_)
            t2 = self.t2[i](z2cond_)
            z2 = (1 - self.mask_x2[i]) * ((z2 - t2) * torch.exp(-s2)) + z2_
            log_det_J2 -= s2.sum(dim=1)
        
        return z1, z2, log_det_J1+log_det_J2, cache_batch
    
    def log_prob(self, x1, x2):
        z1, z2, logp, cache_batch = self.f(x1, x2)
        cache = {'z1': z1, 
                    'z2': z2, 
                    'logp': logp, 
                    'prior_x1': self.prior_x1.log_prob(z1), 
                    'prior_x2': self.prior_x2.log_prob(z2),
                    'cache_batch': cache_batch}
        return self.prior_x1.log_prob(z1) + self.prior_x2.log_prob(z2) + logp, cache