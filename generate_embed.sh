#!/bin/sh
####
#### This file calls train.py with all hyperparameters as for the triplet metric learning experiment on In--Store Shopping Retrieval Project.

#source ~/venv/tf_1.9.0_cuda9.2/bin/activate ## This is needed if you're using virtual environment in python.


python embed.py --csv_file=../../dataset/in_shop_defense_triplet_loss_format_TRAIN.csv \
	--data_dir=../../dataset/ \
	--model=./experiments/myexperiment/model_3000 \
    "$@"