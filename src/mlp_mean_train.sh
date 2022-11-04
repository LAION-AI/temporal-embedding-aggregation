#!/bin/bash

python3.8 -m training.main \
    --train-data "pipe:aws s3 cp s3://s-laion/webvid/clip_webvid/oc_h14/train_fixed2/{000000000..000000100}.tar -" \
    --train-num-samples 1000000 \
    --val-data "pipe:aws s3 cp s3://s-laion/webvid/clip_webvid/oc_h14/val_fixed/{000000000..000000007}.tar -" \
    --val-num-samples 2700 \
    --warmup 200 \
    --sequence-length 200 \
    --lr 1e-3 \
    --batch-size 16 \
    --workers 6 \
    --epochs 2 \
    --name "stupid-mlp_mean" \
    --report-to "" \
    --model "aggregation/model_configs/mlp_mean.json" \
