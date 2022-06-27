import os

import pandas as pd

from clip_video_encode import EmbeddingWebDatasetReader
from evaluation import ZeroShotClassification


def center_frame(seq):
    return seq[:, seq.shape[1]//2]


if __name__ == "__main__":
    DATA_DIR = "/mnt/data/CLIP-Kinetics700/data"
    SPLIT = "val"
    TARS = "ds_{000000..000003}.tar"

    urls = os.path.join(DATA_DIR, SPLIT, TARS)

    reader = EmbeddingWebDatasetReader(
        urls,
        standard_seq_len=-1,
        batch_size=1,
        num_prepro_workers=16,
        to_tensor=True,
        enable_text=True,
        enable_meta=False
    )

    labels = pd.read_csv(os.path.join(DATA_DIR, "annotations/train.csv"))["label"].unique().tolist()

    prompt_func = lambda text: "A photo of " + text
    # prompt_func = lambda text: text

    zsc = ZeroShotClassification(
        reader,
        labels,
        center_frame,
        prompt_func=prompt_func,
    )

    res = zsc.evaluate()
    print(res)
