import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter

class RealNVP(nn.Module):
    def __init__(self, nets_x, nett_x, mask_x, 
                 nets_y, nett_y, mask_y, prior_x, prior_y, 
                 pred_mat, cov_cond, pred_len):
        super(RealNVP, self).__init__()
        
        self.pred_len = pred_len
        self.pred_mat = nn.Parameter(data=pred_mat, requires_grad=False)
        self.cov_cond = nn.Parameter(data=cov_cond, requires_grad=False)
        
        self.prior_x = prior_x
        self.prior_y = prior_y
        
        self.mask_x = nn.Parameter(mask_x, requires_grad=False)
        self.t1 = torch.nn.ModuleList([nett_x() for _ in range(len(mask_x))])
        self.s1 = torch.nn.ModuleList([nets_x() for _ in range(len(mask_x))])
        
        self.mask_y = nn.Parameter(mask_y, requires_grad=False)
        self.t2 = torch.nn.ModuleList([nett_y() for _ in range(len(mask_y))])
        self.s2 = torch.nn.ModuleList([nets_y() for _ in range(len(mask_y))])
        
    def g(self, x1, x2):
        # send to latent and predict (with Gaussian)
        z1, z2, _ = self.f(x1, x2)
        z2 = torch.matmul(self.pred_mat, z1.permute(1, 0)).permute(1, 0)
        
        # z2_upper = z2 + 1.96 * torch.diag(self.cov_cond)
        # z2_lower = z2 - 1.96 * torch.diag(self.cov_cond)
        # z2 = torch.cat([z2, z2_upper, z2_lower], 0)
        
        x1_pred, x2_pred = torch.clone(z1), torch.clone(z2)
        # x1 = torch.cat([x1]*3, 0)
        
        for i in range(len(self.t1)):
            x1_ = x1_pred * self.mask_x[i]
            s1 = self.s1[i](x1_)
            t1 = self.t1[i](x1_)
            x1_pred = x1_ + (1 - self.mask_x[i]) * (x1_pred * torch.exp(s1) + t1)
            
        for i in range(len(self.t1)):
            x2_ = x2_pred * self.mask_y[i]
            x2cond_ = torch.cat([x2_, x1], 1)
            s2 = self.s2[i](x2cond_)
            t2 = self.t2[i](x2cond_)
            x2_pred = x2_ + (1 - self.mask_y[i]) * (x2_pred * torch.exp(s2) + t2)
            
        return z1, z2, x1_pred, x2_pred

    def f(self, x1, x2):
        log_det_J1, z1 = x1.new_zeros(x1.shape[0]), x1
        for i in reversed(range(len(self.t1))):
            z1_ = self.mask_x[i] * z1
            s1 = self.s1[i](z1_)
            t1 = self.t1[i](z1_)
            z1 = (1 - self.mask_x[i]) * ((z1 - t1) * torch.exp(-s1)) + z1_
            log_det_J1 -= s1.sum(dim=1)
        
        log_det_J2, z2 = x2.new_zeros(x2.shape[0]), x2
        for i in reversed(range(len(self.t1))):
            z2_ = self.mask_y[i] * z2
            z2cond_ = torch.cat([z2_, x1], 1)
            s2 = self.s2[i](z2cond_)
            t2 = self.t2[i](z2cond_)
            z2 = (1 - self.mask_y[i]) * ((z2 - t2) * torch.exp(-s2)) + z2_
            log_det_J2 -= s2.sum(dim=1)
        
        return z1, z2, log_det_J1+log_det_J2
    
    def log_prob(self,x1,x2):
        z1, z2, logp = self.f(x1, x2)
        return self.prior_x.log_prob(z1) + self.prior_y.log_prob(z2) + logp