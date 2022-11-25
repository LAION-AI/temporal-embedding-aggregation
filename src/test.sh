#!/bin/bash

rm -rf logs logs/stupid_wavg/

python3.8 -m training.main \
    --train-data "pipe:aws s3 cp s3://s-laion/acav100m/clip_acav100m/oc_h14/train_1M/{000000000..000000100}.tar -" \
    --train-num-samples 670000 \
    --dataset-resampled \
    --warmup 200 \
    --sequence-length 200 \
    --lr 5e-4 \
    --batch-size 16 \
    --workers 6 \
    --epochs 1 \
    --name "stupid_wavg" \
    --report-to "" \
    --model "aggregation/model_configs/wavg.json" \
