import os
import torch
import open_clip

import pandas as pd

from clip_video_encode.dataset import EmbeddingWebDatasetReader
from aggregation.representative_frame import RepresentativeFrame
from aggregation.mean import Mean
from aggregation.factory import create_model

from evaluation.retrieval import retrieval_evaluation


def center_frame(seq):
    return seq[:, seq.shape[1]//2]


if __name__ == "__main__":
    VAL_TARS = "pipe:aws s3 cp s3://s-laion/webvid/clip_webvid/oc_h14/val_fixed/{000000000..000000007}.tar -"
    # VAL_TARS = "pipe:aws s3 cp s3://s-laion/webvid/clip_webvid/oc_h14/val/{000000000..000000007}.tar -"
    # VAL_TARS = "pipe:aws s3 cp s3://s-laion/webvid/clip_webvid/oc_l14/val/{000000000..000000007}.tar -"
    # VAL_TARS = "pipe:aws s3 cp s3://s-laion/webvid/clip_webvid/oc_b32/val/{000000000..000000007}.tar -"
    # VAL_TARS = "pipe:aws s3 cp s3://s-laion/webvid/clip_webvid/oai_b32/val/{000000000..000000000}.tar -"
    # VAL_TARS = "pipe:aws s3 cp s3://s-laion/webvid/clip_webvid/val/{000000000..000000000}.tar -"
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

    # model_video = Mean()
    # model_video, model_str = create_model("aggregation/model_configs/self_attn_default_depth10_dim1024.json", pretrained="logs/H14_depth10_8k_bs_1e-3_lr/checkpoints/epoch_4.pt")

    # model_config = "aggregation/model_configs/mlp_mean.json"
    # checkpoint = "logs/stupid-mlp_mean/checkpoints/epoch_2.pt"

    model_config = "aggregation/model_configs/wavg.json"
    checkpoint = "logs/stupid-wavg/checkpoints/epoch_2.pt"
    model_video, model_str = create_model(model_config, pretrained=checkpoint)


    model_video = model_video.to("cuda")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', device=device)
    model_text = model.encode_text

    ret_mets = retrieval_evaluation(model_video, model_text, val_reader, multicaption=False)

    print(ret_mets)
