import os
import torch
import clip
import numpy as np
import open_clip
import pandas as pd

from clip_video_encode.dataset import EmbeddingWebDatasetReader
from aggregation.mean import Mean
from aggregation.aggregator_wrapper import VideoCLIP
from evaluation.retrieval import retrieval_evaluation

class RetrievalDatasetEvaluator:
    def __init__(self, val_tars, aggregator, clip_model, multicaption=False):
        self.val_reader = EmbeddingWebDatasetReader(
            val_tars,
            standard_seq_len=200,
            batch_size=1,
            num_prepro_workers=6,
            to_tensor=False,
            enable_text=False,
            enable_meta=True
        )
        self.video_clip = VideoCLIP(aggregator, clip_model)
        self.multicaption = multicaption

    def evaluate(self):
        return retrieval_evaluation(self.video_clip, self.val_reader, self.multicaption)

oc_h14, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
oai_b32, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")

def evaluate_retrieval(aggregators, datasets, multicaption=True):
    for aggregator in aggregators:
        for tar, clip_model in datasets:
            evaluator = RetrievalDatasetEvaluator(tar, aggregator, clip_model, multicaption)
            metrics = evaluator.evaluate()
            print(metrics)

eval_data = [
    ('pipe:aws s3 cp s3://s-laion/msr_vtt/clip_msr_vtt/oc_h14/test/000000000.tar -', oc_h14),
    ('pipe:aws s3 cp s3://s-laion/msr_vtt/clip_msr_vtt/oai_b32/test/000000000.tar -', oai_b32),
    ('pipe:aws s3 cp s3://s-laion/msvd/clip_msvd/oc_h14/test/000000000.tar -', oc_h14),
    ('pipe:aws s3 cp s3://s-laion/msvd/clip_msvd/oai_b32/test/000000000.tar -', oai_b32),
]

aggregators = [
    Mean()
]

evaluate_retrieval(aggregators, eval_data, multicaption=True)