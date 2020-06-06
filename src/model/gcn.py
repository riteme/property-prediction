from typing import Optional, List, NamedTuple

from .base import BaseModel

import torch
from torch import nn
from rdkit import Chem as chem

import log

class GCNGraph(NamedTuple):
    n: int
    adj: List[List[int]]
    deg: torch.Tensor
    x: torch.Tensor

class GCN(BaseModel):
    def __init__(self,
        num_iteration: int = 3,
        max_atomic_num: int = 100,
        dev: Optional[torch.device] = None
    ):
        super().__init__(dev)
        self.num_iteration = num_iteration
        self.max_atomic_num = max_atomic_num

        self.agg = nn.Linear(self.max_atomic_num, self.max_atomic_num, False)
        self.fc = nn.Linear(self.max_atomic_num, 2)
        self.relu = nn.ReLU()

    def process(self, mol: chem.Mol) -> GCNGraph:
        n = mol.GetNumAtoms()
        adj = [
            list(map(lambda x: x.GetIdx(), u.GetNeighbors())) + [u.GetIdx()]
            for u in mol.GetAtoms()
        ]
        deg = [len(u.GetNeighbors()) + 1 for u in mol.GetAtoms()]  # +1 for disconnected nodes
        idx = [u.GetAtomicNum() - 1 for u in mol.GetAtoms()]

        return GCNGraph(
            n, adj,
            torch.tensor(deg, dtype=torch.float, device=self.device),
            nn.functional.one_hot(
                torch.tensor(idx, device=self.device),
                num_classes=self.max_atomic_num
            )
        )

    def forward(self, data):
        data: GCNGraph  # cue to mypy

        h0 = data.x
        for _ in range(self.num_iteration):
            h = torch.zeros((data.n, self.max_atomic_num), device=self.device)
            for i, idx in enumerate(data.adj):
                assert data.deg[i] > 0
                y = data.deg[idx] * data.deg[i]
                z = (h0[idx] / y.sqrt().reshape(-1, 1)).sum(dim=0)
                h[i] = self.relu(self.agg(z))
            h0 = h

        pred = self.fc(h0.sum(dim=0) / data.n)
        return pred
