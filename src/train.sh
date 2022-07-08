#!/bin/bash

python3.8 -m training.main \
    --train-data "/home/iejmac/wds_kinetics/ds_{000000..000053}.tar" \
    --val-data "/home/iejmac/wds_kinetics/ds_{000054..000057}.tar" \
    --sequence-length 25 \
    --batch-size 128 \
    --workers  6 \
