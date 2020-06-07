from typing import Optional, List, NamedTuple

from math import sqrt

from .base import BaseModel

import torch
from torch import nn
from rdkit import Chem as chem

import log

class GCNGraph(NamedTuple):
    n: int
    adj: torch.Tensor
    x: torch.Tensor

class GCN(BaseModel):
    def __init__(self,
        num_iteration: int = 10,
        max_atomic_num: int = 100,
        dev: Optional[torch.device] = None
    ):
        super().__init__(dev)
        self.num_iteration = num_iteration
        self.max_atomic_num = max_atomic_num

        self.agg = nn.Linear(self.max_atomic_num, self.max_atomic_num, bias=False)
        self.fc = nn.Linear(self.max_atomic_num, 2)
        self.activate = nn.Tanh()

    def process(self, mol: chem.Mol) -> GCNGraph:
        n = mol.GetNumAtoms()

        # all edges (including all self-loops) as index
        begin_idx = [u.GetBeginAtomIdx() for u in mol.GetBonds()]
        end_idx = [u.GetEndAtomIdx() for u in mol.GetBonds()]
        ran = list(range(n))
        index = [begin_idx + end_idx + ran, end_idx + begin_idx + ran]

        # construct coefficients adjacent matrix
        deg = torch.tensor([
            sqrt(1 / (len(u.GetNeighbors()) + 1))
            for u in mol.GetAtoms()
        ], device=self.device)  # +1 for disconnected nodes
        coeff = deg.reshape(-1, 1) @ deg[None, :]  # pairwise coefficients
        adj = torch.zeros((n, n), device=self.device)
        adj[index] = coeff[index]

        # node embedding
        idx = [u.GetAtomicNum() - 1 for u in mol.GetAtoms()]
        vec = nn.functional.one_hot(
            torch.tensor(idx, device=self.device),
            num_classes=self.max_atomic_num
        ).to(torch.float)

        return GCNGraph(n, adj, vec)

    def forward(self, data):
        data: GCNGraph  # cue to mypy

        h = data.x
        for _ in range(self.num_iteration):
            y = data.adj @ h
            h = self.activate(self.agg(y))

        z = h.sum(dim=0) / data.n
        pred = self.fc(z)
        return pred
