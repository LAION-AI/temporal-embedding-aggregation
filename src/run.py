import os

import pandas as pd

from clip_video_encode.dataset import EmbeddingWebDatasetReader
from evaluation import ZeroShotClassification, LinearProbeClassification


def center_frame(seq):
    return seq[:, seq.shape[1]//2]


if __name__ == "__main__":
    DATA_DIR = "/home/iejmac/wds_kinetics"
    TRAIN_TARS = "ds_{000000..000053}.tar"
    VAL_TARS = "ds_{000054..000057}.tar"

    train_urls = os.path.join(DATA_DIR, TRAIN_TARS)
    val_urls = os.path.join(DATA_DIR, VAL_TARS)

    train_reader = EmbeddingWebDatasetReader(
        train_urls,
        standard_seq_len=-1,
        batch_size=1,
        num_prepro_workers=6,
        to_tensor=False,
        enable_text=True,
        enable_meta=False
    )


    val_reader = EmbeddingWebDatasetReader(
        val_urls,
        standard_seq_len=-1,
        batch_size=1,
        num_prepro_workers=6,
        to_tensor=False,
        enable_text=True,
        enable_meta=False
    )

    labels = pd.read_csv(os.path.join(DATA_DIR, "annotations/train.csv"))["label"].unique().tolist()

    prompt_func = lambda text: "a photo of " + text

    # eval_cls = LinearProbeClassification
    eval_cls = ZeroShotClassification

    ev = eval_cls(
        val_reader,
        labels,
        center_frame,
        prompt_func
    )

    res = ev.evaluate()
    print(res)
