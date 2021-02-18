#!/bin/sh
####

python src/embed.py --csv_file=../../dataset/in_shop_defense_triplet_loss_format_TRAIN.csv \
	--data_dir=../../dataset/ \
	--model=./experiments/myexperiment/model_3000 \
    "$@"