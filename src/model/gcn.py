from typing import Optional, List, NamedTuple, Dict

from math import sqrt

from .base import BaseModel

import torch
from torch import nn
from rdkit import Chem as chem

import log

class GCNGraph(NamedTuple):
    n: int
    adj: torch.Tensor
    num: torch.Tensor  # atomic numbers

class GCN(BaseModel):
    def __init__(self,
        num_iteration: int = 2,
        max_atomic_num: int = 32,
        embedding_dim: int = 64,
        dev: Optional[torch.device] = None
    ):
        super().__init__(dev)
        self.num_iteration = num_iteration
        self.max_atomic_num = max_atomic_num
        self.embedding_dim = embedding_dim

        self.embed = nn.Embedding(self.max_atomic_num, self.embedding_dim)
        self.agg = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.fc = nn.Linear(self.embedding_dim, 2)
        self.activate = nn.LeakyReLU()

    def process(self, mol: chem.Mol, atom_map: Dict[int, int]) -> GCNGraph:
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
        num = torch.tensor(
            [atom_map[u.GetAtomicNum()] for u in mol.GetAtoms()],
            device=self.device
        )

        return GCNGraph(n, adj, num)

    def forward(self, data):
        data: GCNGraph  # cue to mypy

        h = self.embed(data.num)
        for _ in range(self.num_iteration):
            y = data.adj @ h
            h = self.activate(self.agg(y))

        z = h.sum(dim=0) / data.n
        pred = self.fc(z)
        return pred
