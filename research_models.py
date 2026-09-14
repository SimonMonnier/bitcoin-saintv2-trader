"""Offline net-return predictors. Not interchangeable with live PPO policies.

Temporal patching is inspired by PatchTST (arXiv:2211.14730); this multivariate
encoder is an ablation, not a reproduction of that forecasting architecture.
"""
import torch
from torch import nn


class TemporalPatchRegressor(nn.Module):
    def __init__(self, features, lookback=96, patch=8, width=48):
        super().__init__()
        if lookback % patch:
            raise ValueError('lookback must be divisible by patch')
        self.patch=patch
        self.project=nn.Linear(features*patch,width)
        self.position=nn.Parameter(torch.zeros(1,lookback//patch,width))
        layer=nn.TransformerEncoderLayer(width,4,128,dropout=.1,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.head=nn.Sequential(nn.LayerNorm(width*2+features),nn.Linear(width*2+features,64),nn.GELU(),nn.Linear(64,2))

    def forward(self,x):
        b,t,f=x.shape
        h=self.project(x.reshape(b,t//self.patch,self.patch*f))+self.position
        h=self.encoder(h)
        return self.head(torch.cat((h.mean(1),h[:,-1],x[:,-1]),-1))


class NetReturnMLP(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(features,128),nn.GELU(),nn.LayerNorm(128),
                               nn.Dropout(.1),nn.Linear(128,64),nn.GELU(),nn.Linear(64,2))
    def forward(self,x):
        return self.net(x[:,-1])
