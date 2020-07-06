from typing import Text, Any, Type, Sequence, List, Dict

import os
from tempfile import TemporaryFile

from models import BaseModel
from cache import smiles_cache

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
        self._inst = self.model_type(self.device, **self.kwargs).to(self.device)

    def reset(self):
        self.clear_checkpoint()
        self.initialize_model()

    def encode_data(self, data: Any) -> Any:
        return self.model_type.encode_data(data, self.device)

    def decode_data(self, data: Any) -> Any:
        return self.model_type.decode_data(data, self.device)

    @smiles_cache
    def process(self, smiles: Text) -> Any:
        '''
        Parse molecules.
        '''
        mol = util.parse_smiles(smiles)
        assert mol is not None, 'Failed to parse SMILES string'

        result = self.model_type.process(mol, self.device)
        return result

    def save_checkpoint(self):
        self.clear_checkpoint()
        self.checkpoint_fp = TemporaryFile()
        torch.save(self._inst.state_dict(), self.checkpoint_fp)
        log.debug(f'checkpoint saved.')

    def load_checkpoint(self):
        assert self.checkpoint_fp is not None, 'No checkpoint available'
        self.checkpoint_fp.seek(0, os.SEEK_SET)
        state_dict = torch.load(self.checkpoint_fp)
        self.initialize_model()
        self._inst.load_state_dict(state_dict)
        log.debug(f'checkpoint loaded.')

    def clear_checkpoint(self):
        if self.checkpoint_fp is not None and not self.checkpoint_fp.closed:
            self.checkpoint_fp.close()
        self.checkpoint_fp = None

    def save_model(self, dst: util.SourceLike):
        fp = util.resolve_source(dst, mode='w')
        state_dict = self._inst.state_dict()
        torch.save(state_dict, fp)

    def load_model(self, src: util.SourceLike):
        fp = util.resolve_source(src)
        state_dict = torch.load(fp)
        self._inst.load_state_dict(state_dict)

    def forward(self, batch: Sequence[Any]) -> torch.Tensor:
        result = torch.empty((len(batch), 2))
        for i, data in enumerate(batch):
            result[i, :] = self._inst.forward(data)
        return result

    def preprocess(self, train_data: List[util.Item]):
        self._inst.preprocess(train_data)

    def postprocess(self, train_data: List[util.Item]):
        self._inst.postprocess(train_data)

    def predict(self, batch: Sequence[Any]) -> torch.Tensor:
        with torch.no_grad():
            pred = torch.empty((len(batch), 2))
            for i, data in enumerate(batch):
                pred[i, :] = self._inst.predict(data)
        return pred

    def params(self):
        return self._inst.parameters()