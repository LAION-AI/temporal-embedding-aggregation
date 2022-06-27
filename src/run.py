import os

import pandas as pd

from clip_video_encode import EmbeddingWebDatasetReader
from evaluation import ZeroShotClassification


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

    zsc = ZeroShotClassification(
        reader,
        labels,
        lambda x: x.mean(axis=1),
        prompt_func=prompt_func,
    )


    res = zsc.evaluate()
    print(res)
