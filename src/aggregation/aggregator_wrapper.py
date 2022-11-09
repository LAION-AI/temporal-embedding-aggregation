import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VideoCLIP(nn.Module):
    """
    Class to wrap aggregators so we can control their normalization and logit scale
    """
    def __init__(self, aggregator):
        super().__init__()
        self.aggregator = aggregator
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, prenorm=True, postnorm=True):
        x = F.normalize(x, dim=-1) if prenorm else x
        x = self.aggregator(x)
        x = F.normalize(x, dim=-1) if postnorm else x
        return x, self.logit_scale.exp()