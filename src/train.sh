#!/bin/bash

python3.8 -m training.main \
    --train-data "pipe:aws s3 cp s3://s-datasets/webvid/clip_webvid/train/{000000000..000001061}.tar -" \
    --train-num-samples 10000000 \
    --val-data "pipe:aws s3 cp s3://s-datasets/webvid/clip_webvid/val/{000000000..000000000}.tar -" \
    --val-num-samples 5000 \
    --sequence-length 200 \
    --batch-size 128 \
    --workers  6 \
    --epochs 20 \
    --report-to "wandb" \
    --model "aggregation/model_configs/self_attn_default.json"
