from typing import Union, Dict, TextIO, Text, Type, Sequence, Tuple
from pathlib import Path
from hashlib import sha1

from sklearn import metrics

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

def evaluate_auc(std: Sequence[int], pred: Sequence[float]) -> Tuple[float, float]:
    '''Evaluate ROC-AUC and PRC-AUC, returned in a tuple.
    '''
    roc_auc = metrics.roc_auc_score(std, pred)
    prec, recall, _ = metrics.precision_recall_curve(std, pred)
    prc_auc = metrics.auc(recall, prec)
    return roc_auc, prc_auc
