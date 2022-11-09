"""
Weighted average
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

class WAvg(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, seq_len, 1)) # 1 weight per 
        # self.w = nn.Parameter(torch.full((1, seq_len, 1), 1/200.0))
    def forward(self, x):
        x = (x * self.w) / self.w.sum()
        x = torch.sum(x, 1)
        return x
