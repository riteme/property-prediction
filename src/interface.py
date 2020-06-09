from typing import Text, Any, Type, Optional, Sequence, List, Dict

import os
from tempfile import TemporaryFile

from models import BaseModel

import torch

import util
import log

class ModelInterface:
    def __init__(self,
        model_type: Type[BaseModel],
        dev: Optional[torch.device] = None,
        **kwargs
    ):
        self.atom_map: Dict[int, int] = {}
        self.model_type = model_type
        self.device = dev
        self.kwargs = kwargs
        self.checkpoint_fp = None
        self.initialize_model()

    def initialize_model(self):
        self.inst = self.model_type(dev=self.device, **self.kwargs).to(self.device)

    def save_checkpoint(self):
        self.checkpoint_fp = TemporaryFile()
        torch.save(self.inst.state_dict(), self.checkpoint_fp)
        log.debug(f'checkpoint saved.')

    def load_checkpoint(self):
        assert self.checkpoint_fp is not None, 'No checkpoint available'
        self.checkpoint_fp.seek(0, os.SEEK_SET)
        state_dict = torch.load(self.checkpoint_fp)
        self.initialize_model()
        self.inst.load_state_dict(state_dict)
        log.debug(f'checkpoint loaded.')

    def process(self, smiles: Text) -> Any:
        '''Parse molecule
        '''

        mol = util.parse_smiles(smiles)
        assert mol is not None, 'Failed to parse SMILES string'

        for u in mol.GetAtoms():
            num = u.GetAtomicNum()
            if num not in self.atom_map:
                idx = len(self.atom_map)
                self.atom_map[num] = idx

        result = self.inst.process(mol, self.atom_map)
        return result

    def forward(self, batch: Sequence[Any]) -> torch.Tensor:
        result = torch.zeros((len(batch), 2))
        for i, data in enumerate(batch):
            result[i, :] = self.inst.forward(data)
        return result

    def predict(self, batch: Sequence[Any]) -> torch.Tensor:
        with torch.no_grad():
            raw = self.forward(batch)
            pred = raw.softmax(dim=1)
        return pred