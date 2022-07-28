#!/bin/bash

python3.8 -m training.main \
    --train-data "/fsx/iejmac/wds_kinetics/ds_{000000..000053}.tar" \
    --train-num-samples 536000 \
    --val-data "/fsx/iejmac/wds_kinetics/ds_{000054..000057}.tar" \
    --val-num-samples 39000 \
    --sequence-length 25 \
    --batch-size 128 \
    --workers  6 \
    --epochs 20 \
    --depth 1 \
