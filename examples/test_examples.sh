#!/bin/bash
set -e
pip install fvcore iopath scikit-learn pandas matplotlib seaborn dgl dglgo -f https://data.dgl.ai/wheels/repo.html
 
python classification.py

python regression.py

python vcl.py --dataset mnist --inference ml --test # can't download and run cifar succesfully
python vcl.py --dataset mnist --inference mean-field --test

python gnn.py --num-epochs 100
python resnet.py --inference ml --num-epochs 10
python resnet.py --inference mean-field --num-epochs 10
python resnet.py --inference map --num-epochs 10

# Adding a test for one of the last-laye rones would be nice, but not sure how to handle the shipping of pre-trained weights
# python resnet.py --inference last-layer-mean-field --num-epochs 10

# pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
# python nerf.py #have not installed dependencies as they are extensive / needs a conda setup
