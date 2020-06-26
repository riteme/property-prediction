from typing import List, NamedTuple, Dict

from math import sqrt

from .base import *
from . import feature
import log


class GCNGraph(NamedTuple):
    n: int
    adj: torch.Tensor
    num: torch.Tensor  # atomic numbers


class ToyGCN(BaseModel):
    def __init__(self, device: torch.device, *,
        num_iteration: int = 2,
        max_atomic_num: int = 32,
        embedding_dim: int = 64,
        **kwargs
    ):
        super().__init__(device)
        self.num_iteration = num_iteration
        self.max_atomic_num = max_atomic_num
        self.embedding_dim = embedding_dim

        self.embed = nn.Embedding(self.max_atomic_num, self.embedding_dim)
        self.agg = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.fc = nn.Linear(self.embedding_dim, 2)
        self.activate = nn.LeakyReLU()

    @staticmethod
    def process(mol: Mol, device: torch.device) -> GCNGraph:
        n = mol.GetNumAtoms() + 1  # allocate a new node for graph embedding

        # all edges (including all self-loops) as index
        begin_idx = [u.GetBeginAtomIdx() for u in mol.GetBonds()] + [n - 1] * (n - 1)
        end_idx = [u.GetEndAtomIdx() for u in mol.GetBonds()] + list(range(n - 1))
        assert len(begin_idx) == len(end_idx)
        ran = list(range(n))
        index = [begin_idx + end_idx + ran, end_idx + begin_idx + ran]

        # construct coefficients adjacent matrix
        deg = torch.tensor([
            sqrt(1 / (len(u.GetNeighbors()) + 2))
            for u in mol.GetAtoms()
        ] + [sqrt(1 / n)], device=device)
        coeff = deg.reshape(-1, 1) @ deg[None, :]  # pairwise coefficients
        adj = torch.zeros((n, n), device=device)
        adj[index] = coeff[index]

        # node embedding
        num = torch.tensor(
            [feature.ATOM_MAP[u.GetAtomicNum()] for u in mol.GetAtoms()] +
            [len(feature.ATOM_MAP)],
            device=device
        )

        return GCNGraph(n, adj, num)

    def forward(self, data):
        data: GCNGraph  # cue to mypy

        h = self.embed(data.num)
        for _ in range(self.num_iteration):
            y = data.adj @ h
            h = self.activate(self.agg(y))

        # z = h.sum(dim=0) / data.n
        z = h[data.n - 1, :]
        pred = self.fc(z)
        return pred
