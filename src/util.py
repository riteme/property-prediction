from typing import (
    Union, Dict, TextIO,
    Text, Type, Sequence,
    Tuple, Iterable, List,
    TypeVar, Iterator, Generic,
    NamedTuple, Any
)
T = TypeVar('T')

from random import shuffle
from pathlib import Path
from hashlib import sha1
from collections import namedtuple

from sklearn import metrics

# Item = namedtuple('Item', ['obj', 'activity'])
class Item(NamedTuple):
    obj: Any
    activity: int

class RandomIterator(Generic[T]):
    '''iterate over a list and shuffle the list at each round.

    self.data: target list.\n
    self.pos: current position of the cursor.\n
    self.cycled: is last round finished?
    '''

    def __init__(self, data: List[T]):
        self.data = data
        self.pos = 0
        self.cycled = False

    def is_cycled(self) -> bool:
        result = self.cycled
        self.cycled = False
        return result

    def iterate(self, count: int) -> Iterator[T]:
        for i in range(count):
            yield self.data[self.pos]
            self.pos += 1

            if self.pos >= len(self.data):
                self.pos = 0
                self.cycled = True
                shuffle(self.data)

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

def evaluate_auc(std: Sequence[int], pred: Sequence[float]) -> Tuple[float, float]:
    '''Evaluate ROC-AUC and PRC-AUC, returned in a tuple.
    see <https://github.com/yangkevin2/coronavirus_data/blob/master/scripts/evaluate_auc.py>
    '''

    roc_auc = metrics.roc_auc_score(std, pred)
    prec, recall, _ = metrics.precision_recall_curve(std, pred)
    prc_auc = metrics.auc(recall, prec)
    return roc_auc, prc_auc
