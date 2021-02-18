#!/bin/sh

python vis.py --img_folder ../../dataset/ \
	--query_csv ../../dataset/in_shop_defense_triplet_loss_format_QUERY.csv \
	--query_h5 ./embed/myexperiment/in_shop_defense_triplet_loss_format_QUERY-model_3000-None.h5 \
	--gallery_csv ../../dataset/in_shop_defense_triplet_loss_format_GALLERY.csv \
	--gallery_h5 ./embed/myexperiment/in_shop_defense_triplet_loss_format_GALLERY-model_3000-None.h5 \
	--k 5 \
	--output ./top_5_viz_results \
	"$@"		