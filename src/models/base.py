from typing import Any, Optional

import torch
from torch import nn
from rdkit.Chem import Mol

class BaseModel(nn.Module):
    def __init__(self, dev: Optional[torch.device] = None, **kwargs):
        super().__init__()
        self.device = dev

    def process(self, mol: Mol) -> Any:
        ...