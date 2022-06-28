import os

import pandas as pd

from clip_video_encode import EmbeddingWebDatasetReader
from evaluation import ZeroShotClassification, LinearProbeClassification


def center_frame(seq):
    return seq[:, seq.shape[1]//2]


if __name__ == "__main__":
    DATA_DIR = "/mnt/data/CLIP-Kinetics700/data"
    TRAIN_TARS = "ds_{000000..000048}.tar"
    VAL_TARS = "ds_{000000..000003}.tar"

    train_urls = os.path.join(DATA_DIR, "train", TRAIN_TARS)
    val_urls = os.path.join(DATA_DIR, "val", VAL_TARS)

    train_reader = EmbeddingWebDatasetReader(
        train_urls,
        standard_seq_len=-1,
        batch_size=1,
        num_prepro_workers=16,
        to_tensor=False,
        enable_text=True,
        enable_meta=False
    )


    val_reader = EmbeddingWebDatasetReader(
        val_urls,
        standard_seq_len=-1,
        batch_size=1,
        num_prepro_workers=16,
        to_tensor=False,
        enable_text=True,
        enable_meta=False
    )

    labels = pd.read_csv(os.path.join(DATA_DIR, "annotations/train.csv"))["label"].unique().tolist()

    prompt_func = lambda text: "a photo of " + text
    # prompt_func = lambda text: text

    eval_cls = LinearProbeClassification

    ev = eval_cls(
        train_reader,
        val_reader,
        center_frame,
        labels,
    )

    res = ev.evaluate()
    print(res)
