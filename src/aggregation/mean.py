"""
Computes the mean of all frame embeddings.

video_embedding = mean(frame_embeddings)
"""

from torch import nn
import torch.nn.functional as F
class Mean(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return F.normalize(x.mean(axis=-2), dim=-1) # assumes shape always ends with (..., n_frames, embed_dim)