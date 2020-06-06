from typing import Any

from torch import nn
from rdkit.Chem import Mol

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def process(self, mol: Mol) -> Any:
        ...

    def forward(self, data):
        ...
