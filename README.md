# Property Prediction

[Open Tasks](https://www.aicures.mit.edu/tasks)

## Prerequisites

* Python 3
* PyTorch (see <https://pytorch.org/get-started/locally/>)
* PyTorch Geometric (see <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>)
* RDKit (see <http://www.rdkit.org/docs/Install.html>)
* DGL (see <https://www.dgl.ai/pages/start.html>)
* scikit-learn
* colorama
* click

```
sudo pip install scikit-learn colorama click
```

## Usage

See `python src/main.py train --help`.

e.g. train GCN (from DGL):

```
python src/main.py -v train -e 0.001 --train-with-test -m gcn --ndrop 0.90 --cuda
```

* To implement a new model, see `src/models/README.md`.
* To inspect command line processing, check out `src/main.py`.
* To inspect training phases, check out `src/train.py`.

## Results

* ToyGCN

```
python src/main.py -v train -m toy-gcn --ndrop=0.85
```

* GCN

```
python src/main.py -v train -m gcn
```

* GAT

```
python src/main.py -v train -m gat -t1 -j5 --cuda --max-iteration=3 -c .cache/gat.out -s f_score
```

* MPNN

```
python src/main.py -v train -m mpnn -c .cache/mpnn.out --cuda -t1 --max-iteration=5
```

* AdaBoost (with GAT embedding)

```
python src/main.py -v train --max-iteration=3 -c .cache/gat.out -s f_score -t1 -j5 --cuda -m adaboost --inner-model=gat
python src/main.py -v train --max-iteration=3 -c .cache/gat.out -t1 -j5 --cuda -m adaboost --inner-model=gat --ndrop 0 --train-validate -s "-loss" --swap-threshold=0.15
```

* SVM (with GAT embedding)

```
python src/main.py -v train --max-iteration=2 -c .cache/gat.out -t1 -j5 --cuda --ndrop 0 --train-validate -s "-loss" -m svm
```

* LSTM

```
python src/main.py -v train -m lstm --max-iteration=1 -t1 -j5 --cuda --ndrop=0.85 --train-validate
```

## Training & Evaluating

* GraphSAGE:

```shell
# preparation
mkdir .onefold
cd .onefold
ln -s ../data/fold_0 .
cd ..
mkdir .cache
mkdir .model

# training
python src/main.py cache -m graphsage data/train.csv -o .cache/sage.out
python src/main.py -v train -m graphsage --train-validate --train-test -c .cache/sage.out -t1 -d .onefold -s f_score -o .model/sage.out
python src/main.py -v evaluate .model/sage.out -m graphsage -d data/train.csv -c .cache/sage.out
```

Sample output:

```
(info) test: tn=2042,fp=7/tp=48,fn=0
(info) ROC-AUC: 0.9999796648771759
(info) PRC-AUC: 0.9991581632653062
```

## TODO

* Training framework
    * [x] Memory cache for molecule parsing.
    * [x] Disk cache for molecule parsing. See `cache` subcommand and `--cache-file` (`-c`) option.
    * [x] Multiprocessing. See `--num-threads` (`-t`), `--num-workers` (`-j`) and `--spawn-method` (`-sp`) options.
    * [x] Disk cache compatibility with CUDA tensors.
* Models
    * [x] GCN
    * [x] GAT
    * [x] AdaBoost
    * [x] SVM

## Notes

Since PyTorch uses OpenMP for multithreaded training, it will incur great penalty for Python's multiprocessing. It is suggested that use `-t1 -j5 -sp fork` to run cross validations (5 single-threaded worker processes). The reason for using "fork" method is that it preserves memory cache.

e.g.

```
python src/main.py cache data/train.csv -m gcn -o .cache/gcn.out
time python src/main.py -v train -m gcn --embedding-dim=64 -s f_score --max-iteration=7 -t1 -j5 --cache-file .cache/gcn.out -sp fork
```

## Contributors

* Lin Zihang ([@EZlzh](https://github.com/EZlzh))
* Xue Zhenliang ([@riteme](https://github.com/riteme))