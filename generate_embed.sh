#!/bin/sh
####

python src/embed.py --csv_file=../../dataset/in_shop_defense_triplet_loss_format_TRAIN.csv \
	--data_dir=../../dataset/ \
	--model=./experiments/myexperiment_3/model_4000 \
    "$@"

python src/embed.py --csv_file=../../dataset/in_shop_defense_triplet_loss_format_GALLERY.csv \
	--data_dir=../../dataset/ \
	--model=./experiments/myexperiment_3/model_4000 \
    "$@"


python src/embed.py --csv_file=../../dataset/in_shop_defense_triplet_loss_format_QUERY.csv \
	--data_dir=../../dataset/ \
	--model=./experiments/myexperiment_3/model_4000 \
    "$@"