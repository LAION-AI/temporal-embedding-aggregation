"""combine clip with an embedding aggregator."""
# TODO temporary, shift all to open_clip
import clip
import torch
import numpy as np

from torch import nn
from torchvision.transforms import ToPILImage, Compose, ToTensor, Normalize


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class CLIPWrapper(nn.Module):
    def __init__(self, model_video, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        # TODO: overfit to OAI B/32
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.preprocess = Compose(
            [
                ToPILImage(),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        self.model_video = model_video
        self.device = device

    # NOTE: only works for single video
    def forward(self, video, text):
        # video - (n_frames, 224, 224, 3)
        # text - (1, 77)

        # TODO: hacky, make faster
        frames = []
        for fr in video:
            frames.append(self.preprocess(fr)[None, ...])
        video = torch.from_numpy(np.concatenate(frames)).to(self.device)

        frame_embeddings = self.clip_model.encode_image(video)
        text_embeddings = self.clip_model.encode_text(text)

        video_embedding = self.model_video(frame_embeddings[None, ...], None)
        return video_embedding, text_embeddings
