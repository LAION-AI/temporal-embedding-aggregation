import os
import torch
import open_clip
import numpy as np

import pandas as pd

from clip_video_encode.dataset import EmbeddingWebDatasetReader
from aggregation.representative_frame import RepresentativeFrame
from aggregation.mean import Mean
from aggregation.wavg import WAvg
from aggregation.factory import create_model, load_state_dict
from aggregation.aggregator_wrapper import VideoCLIP
from evaluation.retrieval import retrieval_evaluation


def center_frame(seq):
    return seq[:, seq.shape[1]//2]


if __name__ == "__main__":
    VAL_TARS = "pipe:aws s3 cp s3://s-laion/msr_vtt/clip_msr_vtt/oc_h14/test_fix/{000000000..000000007}.tar -"
    # VAL_TARS = "/fsx/iejmac/datasets/msr-vtt/fix/dataset/{000000000..000000059}.tar"

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

    # config = "H14_depth_run_0"
    # model_config = f"experiments/model_configs/{config}.json"
    # checkpoint = f"logs/{config}/checkpoints/epoch_10.pt"
    # checkpoint = f"logs/{config}-double_normalized/checkpoints/epoch_4.pt"
    # checkpoint = "logs/H14_depth_run_0-double_normalized_remove_mean/checkpoints/epoch_3.pt"

    # model_config = "aggregation/model_configs/mlp_mean.json"
    # checkpoint = "logs/stupid-mlp_mean/checkpoints/epoch_1.pt"

    model_config = "aggregation/model_configs/wavg.json"
    checkpoint = "logs/stupid_wavg/checkpoints/epoch_1.pt"

    # model_video, model_str = create_model(model_config, pretrained=checkpoint)
    # model_video, model_str = create_model(model_config)
    # model_video.w = torch.nn.Parameter(torch.abs(model_video.w))

    # model_video = Mean()
    model_video = WAvg(200)
    sd = load_state_dict(checkpoint)
    model_video.w.data = sd["aggregator.w"]

    n_params = sum(p.numel() for p in model_video.parameters() if p.requires_grad)
    print(n_params / 1e6)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k", device=device)
    # model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    # model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
    # model_text = model.encode_text

    model_video = VideoCLIP(model_video, model)
    model_video.to(device)

    ret_mets = retrieval_evaluation(model_video, val_reader, multicaption=True)
    print(ret_mets)
