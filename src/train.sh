#!/bin/bash

python3.8 -m training.main \
    --train-data "pipe:aws s3 cp s3://s-laion/webvid/clip_webvid/oc_h14/train/{000000000..000001087}.tar -" \
    --train-num-samples 10000000 \
    --val-data "pipe:aws s3 cp s3://s-laion/webvid/clip_webvid/oc_h14/val_no_overlap/{000000000..000000007}.tar -" \
    --val-num-samples 2700 \
    --warmup 200 \
    --sequence-length 200 \
    --lr 1e-3 \
    --batch-size 1024 \
    --workers 6 \
    --epochs 10 \
    --name "H14_depth20_8k_bs_1e-3_lr" \
    --report-to "wandb" \
    --model "aggregation/model_configs/self_attn_default_depth20_dim1024.json" \
    --resume "logs/H14_depth20_8k_bs_1e-3_lr/checkpoints/epoch_3.pt"
