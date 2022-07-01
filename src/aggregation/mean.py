"""
Computes the mean of all frame embeddings.

video_embedding = mean(frame_embeddings)
"""

from torch.nn import Module

class Mean(Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x.mean(axis=-2) # assumes shape always ends with (..., n_frames, embed_dim)
