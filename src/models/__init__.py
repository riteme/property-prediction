from typing import Type, Text

from .base import BaseModel
from .stupid import StupidModel
from .toygcn import ToyGCN
from .gcn import GCN


def select(name: Text) -> Type[BaseModel]:
    return {
        'toy-gcn': ToyGCN,
        'gcn': GCN
    }[name]