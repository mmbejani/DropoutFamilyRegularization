import torch
from torch import nn

import torch_dct as dct

import numpy as np

class SpectralDropout(nn.Module):

    def __init__(self, tau=0.1, p=0.2, device='cuda'):
        super().__init__()
        self.tau = tau
        self.p = p
        self.device = device

    def greater_mask(self, x):
        r = torch.gt(x, self.tau)
        if self.device == 'cuda':
            return r.cuda()
        return r

    def bernoulli_mask(self, shape):
        r_matrix = torch.rand(shape)
        r = torch.gt(r_matrix, self.p)
        if self.device == 'cuda':
            return r.cuda()
        return r

    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.train:
            x_flat = x.view([-1, np.prod(x.size()[1:])])
            x_dct = dct.dct(x_flat)
            r = self.greater_mask(x_dct)
            b = self.bernoulli_mask(x_flat.shape)
            y_dct = x_dct * r + x_dct * ~r * b
            y = dct.idct(y_dct)
            y = y.view(x.size())
            return y
        else:
            return x


class Bridgeout(nn.Module):

    def __init__(self,layer:nn.Module, p=0.7,q=0.7, device='cuda'):
        super().__init__()
        self.layer = layer
        self.p = p
        self.q = q
        self.device = device

    def bernoulli_mask(self, shape):
        r_matrix = torch.rand(shape)
        r = torch.gt(r_matrix, self.p)
        if self.device == 'cuda':
            return r.cuda()
        return r

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.train:
            w = self.layer.weight
            b = self.bernoulli_mask(w.size())
            noise = torch.pow(torch.abs(w), self.q / 2) * (b / self.p - 1)
            w.data = w.data + noise
        return self.layer(x)


class Shakeout(nn.Module):

    def __init__(self, layer:nn.Module, tau=0.1, c=0.1, device='cuda'):
        super().__init__()
        self.layer = layer
        self.tau = tau
        self.itau = 1 / (1 - tau)
        self.c = c
        self.softsign = nn.Softsign()
        self.device = device

    def bernoulli_mask(self, shape):
        r_matrix = torch.rand(shape)
        r = torch.gt(r_matrix, self.tau)
        if self.device == 'cuda':
            return r.cuda()
        return r

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.train:
            w = self.layer.weight
            mask = self.bernoulli_mask(w.size())
            mask_sign = self.softsign(w * mask)
            imask_sign = self.softsign(w * ~mask)
            w.data = self.c * imask_sign + \
                self.itau * (w.data + self.c * self.tau * mask_sign)
        return self.layer(x)

class DropConnect(nn.Module):

    def __init__(self, layer:nn.Module, p=0.2, device='cuda'):
        super().__init__()
        self.layer = layer
        self.p = p
        self.device = device

    def bernoulli_mask(self, shape):
        r_matrix = torch.rand(shape)
        r = torch.gt(r_matrix, self.tau)
        if self.device == 'cuda':
            return r.cuda()
        return r

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.train:
            w = self.layer.weight
            mask = self.bernoulli_mask(w.size())
            w.data = w.data * mask
        return self.layer(x)