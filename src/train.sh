#!/bin/bash

torchrun -m training.main \
    --train-data "/home/iejmac/wds_kinetics/ds_{000000..000053}.tar" \
    --train-num-samples 536000 \
    --val-data "/home/iejmac/wds_kinetics/ds_{000054..000057}.tar" \
    --val-num-samples 39000 \
    --sequence-length 25 \
    --batch-size 128 \
    --lr 3e-4 \
    --workers  6 \
    --epochs 20 \
    --warmup 1000 \
    --name "test_dist" \
