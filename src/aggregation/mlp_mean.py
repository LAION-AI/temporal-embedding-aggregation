"""
MLP then Mean
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

class MLPMean(nn.Module):
    def __init__(self, dim, proj_dim):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, proj_dim),
        )
    def forward(self, inp):
        x = self.mlp_head(inp)
        x = torch.mean(x, 1)

        x = (x + inp.mean(axis=-2))/2.0 # residual
        return x
