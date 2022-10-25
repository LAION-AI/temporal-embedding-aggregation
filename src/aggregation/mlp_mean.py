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
            nn.LayerNorm(dim),
            nn.Linear(dim, proj_dim)
        )
    def forward(self, x):
        x = self.mlp_head(x)
        x = torch.mean(x, 1)
        return x
