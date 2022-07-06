#!/bin/bash
pip install scikit-learn pandas matplotlib seaborn dgl dglgo -f https://data.dgl.ai/wheels/repo.html
 
python classification.py

python regression.py

python vcl.py --dataset mnist --inference ml --test # can't download and run cifar succesfully
python vcl.py --dataset mnist --inference mean-field --test

python gnn.py --num-epochs 100
python resnet.py --inference ml --num-epochs 10

# python nerf.py #have not installed dependencies as they are extensive
# python titanic_classification.py # not autodownloading the data yet