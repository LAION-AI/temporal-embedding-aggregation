#!/bin/bash

python3.8 -m training.main \
    --train-data "pipe:aws s3 cp s3://s-laion/webvid/clip_webvid/oc_h14/train_fixed2/{000000000..000000010}.tar -" \
    --train-num-samples 100000 \
    --val-data "pipe:aws s3 cp s3://s-laion/webvid/clip_webvid/oc_h14/val_fixed/{000000000..000000007}.tar -" \
    --val-num-samples 2700 \
    --warmup 200 \
    --sequence-length 200 \
    --lr 5e-4 \
    --batch-size 16 \
    --workers 6 \
    --epochs 2 \
    --name "stupid-wavg" \
    --report-to "" \
    --model "aggregation/model_configs/wavg.json" \
