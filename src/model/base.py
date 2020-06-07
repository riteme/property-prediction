from typing import Any, Optional, Dict

import torch
from torch import nn
from rdkit.Chem import Mol

class BaseModel(nn.Module):
    def __init__(self, dev: Optional[torch.device] = None):
        super().__init__()
        self.device = dev

    def process(self, mol: Mol, atom_map: Dict[int, int]) -> Any:
        ...

    def forward(self, data):
        ...
