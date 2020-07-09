from typing import Type, Text

from .base import BaseModel
from .stupid import StupidModel
from .toygcn import ToyGCN
from .gcn import GCN
from .gat import GAT
from .graphsage import GraphSAGE
from .chebnet import ChebNet
from .adaboost import AdaBoost
from .mpnn import MPNN
from .svm import SVM
from .lstm import LSTM

def select(name: Text) -> Type[BaseModel]:
    return {
        'toy-gcn': ToyGCN,
        'gcn': GCN,
        'gat': GAT,
        'graphsage': GraphSAGE,
        'chebnet': ChebNet,
        'adaboost': AdaBoost,
        'mpnn': MPNN,
        'svm': SVM,
        'lstm': LSTM
    }[name]