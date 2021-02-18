#!/bin/sh
####
#### This file calls train.py with all hyperparameters as for the triplet metric learning experiment on In--Store Shopping Retrieval Project.

#source ~/venv/tf_1.9.0_cuda9.2/bin/activate ## This is needed if you're using virtual environment in python.


IMAGE_ROOT=../datasets/
EXP_ROOT=../datasets/train/BS_fashion/ ## THIS WILL BE THE OUTPUT FOLDER

CUDA_VISIBLE_DEVICES=0 
python train.py \
    'myexperiment' \
    --csv_file ../../dataset/in_shop_defense_triplet_loss_format_TRAIN.csv \
    --data_dir ../../dataset/ \
    --loss BatchHard \
    --model trinet \
    --dim 128 \
    --sampler TripletBatchSampler \
    "$@"