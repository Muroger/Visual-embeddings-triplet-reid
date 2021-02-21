# Description
A pytorch implementation of the ["In Defense of the Triplet Loss for Person Re-Identification"](https://arxiv.org/abs/1703.07737) paper.

For [«In-shop Clothes Retrieval Benchmark»](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) dataset.

This repository also contains also a implementation of the MGN network from the paper: 
["Learning Discriminative Features with Multiple Granularities for Person Re-Identification"](https://arxiv.org/abs/1804.01438)

["Original"](https://github.com/kilsenp/triplet-reid-pytorch)
# Requirements

- numpy
- Pillow
- h5py
- scipy
- torch
- torchvision


For evaluation and visualization:
- scikit-learn
- scipy
- h5py
- annoy

# Train
```
./train.sh
```
or

```
python3 src/train.py --csv_file path/to/ 
                 --data_dir path/to/image/base/directory 
                 --loss BatchHard
                 --model effnet
                 --dim 128
                 --sampler TripletBatchSampler
```
The script looks for the image files by joining the data_dir and image path given in the csv file.
The csv file should be a two column file with pids, fids.

To create proper .csv files.

```
python3 preprocessors/Fashion_convert2defense_triplet_format.py
```


If you would like to train using the MGN network, use the following command:

```
python3 src/train.py --csv_file path/to/ 
                 --data_dir path/to/image/base/directory 
                 --loss BatchHardSingleWithSoftmax
                 --model mgn
                 --mgn branches 1 2 3
                 --dim 256
```


# Evaluation

You can use embed.py to write out embeddings that are compatible with the 
evaluation script.


```
python3 src/embed.py --csv_file path/to/gallery/or/query/csv
                 --data_dir path/to/image/base/directory
                 --model path/to/model/file
```                 
To calculate the final scores:

```
python src/evaluate.py --dataset=file path/to/train/csv 
	--data_dir=path/to/image/base/directory
	--model=path/to/model/file
	--gallery=path/to/gallery/csv
	--query=path//or/query/csv
```
