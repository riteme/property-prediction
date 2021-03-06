from random import random

from .base import *


class StupidModel(BaseModel):
    def __init__(self, dev: torch.device, **kwargs):
        super().__init__(dev)
        self.weight = torch.nn.Parameter(
            torch.tensor([1.0, 1.0]), True
        )

    @staticmethod
    def process(mol: Mol, device: torch.device, **kwargs):
        return None

    def forward(self, data):
        val = random()
        y = (val + self.weight)**(-1) * val
        return y