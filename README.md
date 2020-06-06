# Property Prediction

[Open Tasks](https://www.aicures.mit.edu/tasks)

## Prerequisites

* Python 3
* PyTorch (see <https://pytorch.org/get-started/locally/>)
* RDKit (see <http://www.rdkit.org/docs/Install.html>)
* scikit-learn
* colorama
* click

```
sudo pip install scikit-learn colorama click
```

## Usage

See `python src/main.py --help`.

e.g.

```
python src/main.py -v train -e 0.01 -b 256
```

## TODO

* [ ] Memory cache for molecule parsing.
* [ ] Multiprocessing.

## Contributors

* Lin Zihang ([@EZlzh](https://github.com/EZlzh))
* Xue Zhenliang ([@riteme](https://github.com/riteme))