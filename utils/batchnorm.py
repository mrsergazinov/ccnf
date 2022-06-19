import torch
from torch import nn
from torch.nn.parameter import Parameter

class BatchNormInv(nn.Module):
    '''
    invertible batch normalization
    '''
    def __init__(self):
        super(BatchNormInv, self).__init__()
        self.eps = 1e-5
        self.momentum = 0.1
        self.first_run = True

    def forward(self, input):
        # input: [batch_size, num_features]
        device = input.device
        if self.training:
            mean = torch.mean(input, dim=0, keepdim=True).to(device)  # [1, num_features]
            var = torch.var(input, dim=0, unbiased=False, keepdim=True).to(device)  # [1, num_features]
            if self.first_run:
                self.inp_shape = (1, input.shape[1]) # [1, num_features]
                self.weight = Parameter(torch.zeros(self.inp_shape).to(input.device), requires_grad=True)
                self.bias = Parameter(torch.zeros(self.inp_shape).to(input.device), requires_grad=True)
                self.register_buffer('running_mean', torch.zeros(self.inp_shape).to(input.device))
                self.register_buffer('running_var', torch.ones(self.inp_shape).to(input.device))
                self.first_run = False
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            bn_init = (input - mean) / torch.sqrt(var + self.eps)
        else:
            bn_init = (input - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return  torch.exp(self.weight) * bn_init + self.bias
    
    def inverse(self, input):
        bn_init = input * torch.sqrt(self.running_var + self.eps) + self.running_mean
        return (bn_init - self.bias) /  torch.exp(self.weight)