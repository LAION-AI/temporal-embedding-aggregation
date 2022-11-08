import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoCLIPAggregator(nn.Module):
    """
    Class to wrap aggregators so we can control their normalization and logit scale
    """
    def __init__(self, aggregator, logit_scale=100.0, normalize=True):
        super().__init__()
        self.aggregator = aggregator
        self.logit_scale = logit_scale
        self.normalize = normalize

    def forward(self, x):
        x = self.aggregator(x)
        return self.logit_scale * F.normalize(
            x, dim=-1
        ) if self.normalize else self.logit_scale * x