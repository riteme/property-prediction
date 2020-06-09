from typing import *
from typing import TextIO
S = TypeVar('S')
T = TypeVar('T')

from random import shuffle, random
from pathlib import Path
from hashlib import sha1
from collections import namedtuple

import torch
from sklearn import metrics
from rdkit.Chem import MolFromSmiles as parse_smiles

# Item = namedtuple('Item', ['obj', 'activity'])
class Item(NamedTuple):
    obj: Any
    activity: int

class Sampler(Generic[T]):
    def get_batch(self) -> Iterator[T]:
        ...

class RandomIterator(Sampler, Generic[T]):
    '''iterate over a list and shuffle the list at each round.

    self.data: target list.\n
    self.drop_fn: a function to determine dropout probabilities.\n
    self.pos: current position of the cursor.\n
    self.cycled: is last round finished?
    '''

    def __init__(self,
        data: List[T],
        batch_size: int,
        drop_fn: Optional[Callable[[T], float]] = None
    ):
        self.data = data
        self.batch_size = batch_size
        self.drop_fn = drop_fn
        self.pos = 0

    def get_batch(self) -> Iterator[T]:
        for i in range(self.batch_size):
            while True:
                item = self.data[self.pos]
                self.pos += 1

                if self.pos >= len(self.data):
                    self.pos = 0
                    shuffle(self.data)

                if self.drop_fn is None or self.drop_fn(item) <= random():
                    break

            yield item

class SeparateSampling(Sampler, Generic[T]):
    '''independent sampling of both types.

    self.iters: random iterators for each type.
    '''

    def __init__(self,
        data: Iterable[T],
        batch_sizes: Dict[int, int],
        type_fn: Callable[[T], int],
    ):
        '''
        data: the data sequence.\n
        batch_sizes: numbers of samples in each batch.\n
        type_fn: a function returning type id of items.
        '''
        mp: Dict[int, List[T]] = {}
        for item in data:
            idx = type_fn(item)
            if idx not in mp:
                mp[idx] = []
            mp[idx].append(item)

        self.iters: Dict[int, RandomIterator[T]] = {}
        for idx, seq in mp.items():
            self.iters[idx] = RandomIterator(seq, batch_sizes[idx])

    def get_batch(self):
        for it in self.iters.values():
            yield from it.get_batch()

class MaximalCounter:
    def __init__(self):
        self.maximal = None
        self.count = 0
        self.updated = False

    def __repr__(self):
        return f'maximal={self.maximal},count={self.count}'

    def __str__(self):
        return repr(self)

    def is_updated(self):
        result = self.updated
        self.updated = False
        return result

    def record(self, value):
        if self.maximal is None or value > self.maximal:
            self.maximal = value
            self.count += 1
            self.updated = True

def sha1hex(text: Text) -> Text:
    return sha1(text.encode('utf-8')).hexdigest()

def load_csv(src: Union[Text, Path, TextIO]) -> Dict[Text, int]:
    if isinstance(src, Text):
        src = Path(src)

    if isinstance(src, Path):
        assert src.exists(), 'Source file does not exist'
        fp = src.open('r')
    else:
        fp = src

    try:
        header = fp.readline().strip()
        assert header == 'smiles,activity', 'Format not supported'

        mp = {}
        for line in fp:
            smiles, activity = line.split(',')
            mp[smiles] = int(activity)

        return mp
    finally:
        if fp is not src:
            fp.close()

def separate_items(items: Iterable[Item]) -> Tuple[List[object], List[int]]:
    objs = []
    labels = []
    for x in items:
        objs.append(x.obj)
        labels.append(x.activity)
    return objs, labels

def stat_string(std: Sequence[int], pred_label: Sequence[int]) -> Text:
    c = metrics.confusion_matrix(std, pred_label)
    return f'tn={c[0][0]},fp={c[0][1]}/tp={c[1][1]},fn={c[1][0]}'

def evaluate_auc(std: Sequence[int], pred: torch.Tensor) -> Tuple[float, float]:
    '''Evaluate ROC-AUC and PRC-AUC, returned in a tuple.
    see <https://github.com/yangkevin2/coronavirus_data/blob/master/scripts/evaluate_auc.py>

    pred: probabilities for label 1.
    '''

    roc_auc = metrics.roc_auc_score(std, pred)
    prec, recall, _ = metrics.precision_recall_curve(std, pred)
    prc_auc = metrics.auc(recall, prec)
    return roc_auc, prc_auc

def dict_filter(d: Dict[S, T], fn: Callable[[S, T], bool]) -> Dict[S, T]:
    result = {}
    for key, value in d.items():
        if fn(key, value):
            result[key] = value
    return result