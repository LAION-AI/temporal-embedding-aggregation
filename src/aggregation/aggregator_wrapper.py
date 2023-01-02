import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CLIPTxt(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.token_embedding = clip.token_embedding
        self.positional_embedding = clip.positional_embedding
        self.transformer = clip.transformer
        self.ln_final = clip.ln_final
        self.text_projection = clip.text_projection
        self.attn_mask = clip.attn_mask.cuda()
        # device = self.token_embedding.device
        # self.attn_mask = self.attn_mask.cuda()

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        return x[torch.arange(x.shape[0]), torch.argmax(dim=-1)] @ self.text_projection

class VideoCLIP(nn.Module):
    """
    Class to wrap aggregators so we can control their normalization and logit scale
    """
    def __init__(self, aggregator, clip_model):
        super().__init__()
        self.aggregator = aggregator
        self.model_text = CLIPTxt(clip_model)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_text(self, x, postnorm=True):
        with torch.no_grad():
            text_embeddings = self.model_text(x).float()
        return F.normalize(text_embeddings, dim=-1) if postnorm else text_embeddings

    def encode_video(self, x, prenorm=True, postnorm=True):
        x = F.normalize(x.float(), dim=-1) if prenorm else x
        x = self.aggregator(x)
        x = F.normalize(x.float(), dim=-1) if postnorm else x
        return x

    def forward(self, video_embeddings, toks, prenorm=True, postnorm=True):
        text_embeddings = self.encode_text(toks, postnorm=postnorm)
        video_embeddings = self.encode_video(video_embeddings, prenorm=prenorm, postnorm=postnorm)
        return video_embeddings, text_embeddings, self.logit_scale.exp()
