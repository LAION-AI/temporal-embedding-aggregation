"""
Chooses one embedding as representative of whole video (usually center frame)

video_embedding = get_representative(frame_embeddings)
"""

from torch.nn import Module

class RepresentativeFrame(Module):
  def __init__(
      self,
      get_representative=lambda frames: frames[..., frames.shape[-2]//2, :], # center frame as default
  )
    super().__init__()
    self.get_representative = get_representative
  def forward(self, x):
    return self.get_representative(x)
