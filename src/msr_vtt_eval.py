import os
import torch
import clip
import numpy as np

import pandas as pd

from clip_video_encode.dataset import EmbeddingWebDatasetReader
from aggregation.representative_frame import RepresentativeFrame
from aggregation.mean import Mean
from aggregation.factory import create_model
from evaluation.multicaption_retrieval import multicaption_retrieval_evaluation


def center_frame(seq):
    return seq[:, seq.shape[1]//2]


if __name__ == "__main__":
    VAL_TARS = "pipe:aws s3 cp s3://s-laion/msr_vtt/clip_msr_vtt/test/000000000.tar -"
    val_urls = VAL_TARS 
    val_reader = EmbeddingWebDatasetReader(
        val_urls,
        standard_seq_len=200,
        batch_size=1,
        num_prepro_workers=6,
        to_tensor=False,
        enable_text=False,
        enable_meta=True
    )

    model_video, model_str = create_model("aggregation/model_configs/self_attn_default_depth10.json", pretrained="logs/depth10_good_data/checkpoints/epoch_10.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model_text = model.encode_text

    model_video.to(device)

    ret_mets = multicaption_retrieval_evaluation(model_video, model_text, val_reader)
    print(ret_mets)
