# Property Prediction

[Open Tasks](https://www.aicures.mit.edu/tasks)

## Prerequisites

* Python 3
* PyTorch (see <https://pytorch.org/get-started/locally/>)
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
python src/main.py -v train -m toy-gcn --ndrop=0.85 > result/toy-gcn.txt
```

* GCN

```
python src/main.py -v train -m gcn > result/gcn.txt
```

* GAT

```
python src/main.py -v train -m gat -t1 -j5 --cuda --max-iteration=20 > result/gat.txt
```

## TODO

* Training framework
    * [x] Memory cache for molecule parsing.
    * [x] Disk cache for molecule parsing. See `cache` subcommand and `--cache-file` (`-c`) option.
    * [x] Multiprocessing. See `--num-threads` (`-t`), `--num-workers` (`-j`) and `--spawn-method` (`-sp`) options.
    * [ ] Disk cache compatibility with CUDA tensors.
* Models
    * [x] GCN
    * [x] GAT

## Notes

Since PyTorch uses OpenMP for multithreaded training, it will incur great penalty for Python's multiprocessing. It is suggested that use `-t1 -j5 -sp fork` to run cross validation (5 single-threaded worker processes). The reason for using "fork" method is that it preserves memory cache.

e.g.

```
python src/main.py cache data/train.csv -m gcn -o .cache/gcn.out
time python src/main.py -v train -m gcn --embedding-dim=64 -s f_score --max-iteration=7 -t1 -j5 --cache-file .cache/gcn.out -sp fork
```

## Contributors

* Lin Zihang ([@EZlzh](https://github.com/EZlzh))
* Xue Zhenliang ([@riteme](https://github.com/riteme))