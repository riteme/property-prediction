from typing import Any

import torch
from torch import nn
from rdkit.Chem import Mol

# NOTE: you may use `from .base import *` to ease your life.


class BaseModel(nn.Module):
    def __init__(self, device: torch.device, **kwargs):
        super().__init__()
        self.device = device

    @staticmethod
    def process(mol: Mol, device: torch.device):
        '''
        process molecules. The processed molecules will be directly
        passed to forward phase.
        '''
        raise NotImplementedError

    @staticmethod
    def encode_data(data: Any, device: torch.device) -> Any:
        '''
        encode data for cache storing. e.g. move CUDA tensors
        back to CPU.
        '''
        return data

    @staticmethod
    def decode_data(data: Any, device: torch.device) -> Any:
        '''
        decode data for forward phase. This is the reverse procedure
        of `encode_data`. e.g. move tensors to CUDA device.
        '''
        return data

    def preprocess(self):
        '''
        this is a hook function to be called before training phase.
        '''
        pass

    def postprocess(self):
        '''
        this is a hook function to be called after training phase.
        '''
        pass

    def predict(self, data: Any) -> torch.Tensor:
        '''
        give probability prediction on `data`.
        ModelInterface.predict guarantees this function will be called
        in `torch.no_grad()` environment.

        output size: array of length 2, containing probabilies for
        label 0 & 1 respectively. You must ensure that the sum of probabilites
        summed to 1.
        '''
        return self.forward(data).softmax(dim=0)