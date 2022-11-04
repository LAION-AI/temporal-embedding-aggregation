import os
import torch
import open_clip
import numpy as np

import pandas as pd

from clip_video_encode.dataset import EmbeddingWebDatasetReader
from aggregation.representative_frame import RepresentativeFrame
from aggregation.mean import Mean
from aggregation.factory import create_model
from evaluation.retrieval import retrieval_evaluation


def center_frame(seq):
    return seq[:, seq.shape[1]//2]


if __name__ == "__main__":
    # VAL_TARS = "pipe:aws s3 cp s3://s-laion/msvd/clip_msvd/oc_h14/test/{000000000..000000007}.tar -"
    VAL_TARS = "pipe:aws s3 cp s3://s-laion/msvd/clip_msvd/oc_h14/test/{000000000..000000000}.tar -"
    val_urls = VAL_TARS 
    val_reader = EmbeddingWebDatasetReader(
        val_urls,
        standard_seq_len=200,
        batch_size=1,
        num_prepro_workers=6,
        to_tensor=False,
        enable_text=True,
        enable_meta=False
    )

    model_video, model_str = create_model("aggregation/model_configs/self_attn_default_depth20_dim1024.json", pretrained="logs/H14_depth20_8k_bs_1e-3_lr/checkpoints/epoch_3.pt")
    # model_video = Mean()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k", device=device)
    # model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
    model_text = model.encode_text

    model_video.to(device)

    ret_mets = retrieval_evaluation(model_video, model_text, val_reader, multicaption=True)
    print(ret_mets)
