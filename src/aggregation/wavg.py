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
    def forward(self, x):
        x = x * self.w
        x = torch.mean(x, 1)
        return x
