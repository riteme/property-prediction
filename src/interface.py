from typing import Text, Any, Type, Sequence, List, Dict

import os
from tempfile import TemporaryFile

from models import BaseModel
from cache import memcached

import torch

import util
import log


class ModelInterface:
    def __init__(self,
        model_type: Type[BaseModel],
        device: torch.device,
        no_initialize: bool = False,
        **kwargs
    ):
        self.model_type = model_type
        self.device = device
        self.kwargs = kwargs
        self.checkpoint_fp = None

        if not no_initialize:
            self.initialize_model()

    def initialize_model(self):
        self.inst = self.model_type(self.device, **self.kwargs).to(self.device)

    def reset(self):
        self.clear_checkpoint()
        self.initialize_model()

    @memcached(ignore_self=True)
    def process(self, smiles: Text) -> Any:
        '''Parse molecules.
        '''
        mol = util.parse_smiles(smiles)
        assert mol is not None, 'Failed to parse SMILES string'

        result = self.model_type.process(mol, self.device)
        return result

    def save_checkpoint(self):
        self.clear_checkpoint()
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

    def clear_checkpoint(self):
        if self.checkpoint_fp is not None and not self.checkpoint_fp.closed:
            self.checkpoint_fp.close()
        self.checkpoint_fp = None

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