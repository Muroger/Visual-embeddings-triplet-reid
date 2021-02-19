#!/bin/sh
####
#### This file calls train.py with all hyperparameters as for the triplet metric learning experiment on In--Store Shopping Retrieval Project.

#source ~/venv/tf_1.9.0_cuda9.2/bin/activate ## This is needed if you're using virtual environment in python.


python src/evaluate.py --dataset=../../dataset/in_shop_defense_triplet_loss_format_TRAIN.csv \
	--data_dir=../../dataset/ \
	--model=./experiments/myexperiment_3/model_4000 \
	--gallery=../../dataset/in_shop_defense_triplet_loss_format_GALLERY.csv \
	--query=../../dataset/in_shop_defense_triplet_loss_format_QUERY.csv \
	"$@"