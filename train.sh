#!/bin/sh
####
#### This file calls train.py with all hyperparameters as for the triplet metric learning experiment on In--Store Shopping Retrieval Project.

python src/train.py \
    'myexperiment_3' \
    --csv_file ../../dataset/in_shop_defense_triplet_loss_format_TRAIN.csv \
    --data_dir ../../dataset/ \
    --loss BatchHard \
    --model trinet \
    --dim 128 \
    --sampler TripletBatchSampler \
    "$@"