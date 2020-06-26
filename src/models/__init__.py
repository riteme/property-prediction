from typing import Type, Text

from .base import BaseModel
from .stupid import StupidModel
from .toygcn import ToyGCN
from .gcn import GCN
from .gat import GAT


def select(name: Text) -> Type[BaseModel]:
    '''
    name: the model name from "-m"/"--model-name" command line argument.
    '''

    return {
        'toy-gcn': ToyGCN,
        'gcn': GCN,
        'gat': GAT
    }[name]