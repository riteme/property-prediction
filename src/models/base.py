from typing import Any

import torch
from torch import nn
from rdkit.Chem import Mol


class BaseModel(nn.Module):
    def __init__(self, device: torch.device, **kwargs):
        super().__init__()
        self.device = device

    @staticmethod
    def process(mol: Mol, device: torch.device):
        ...