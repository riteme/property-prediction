from typing import NamedTuple

from .base import BaseModel
from . import feature
import log

import torch
from torch import nn
from rdkit.Chem import Mol
from dgl import DGLGraph

from dgl.nn.pytorch import ChebConv, DenseChebConv

class ChebNetData(NamedTuple):
    n: int
    adj: torch.Tensor
    vec: torch.Tensor

class ChebNet(BaseModel):
    def __init__(self, device: torch.device, *,
        embedding_dim: int = 64,
        k: int = 4,
        no_shortcut: bool = False,
        **kwargs
    ):
        super().__init__(device)
        self.embedding_dim = embedding_dim
        self.no_shortcut = no_shortcut
        # self.aggregator_type = aggregator_type
        self.k = k
        self.embed = DenseChebConv(feature.FEATURE_DIM, embedding_dim, k)
        self.conv = DenseChebConv(embedding_dim, embedding_dim, k)
        self.fc = nn.Linear(embedding_dim, 2)
        self.activate = nn.ReLU()
    
    @staticmethod
    def process(mol: Mol, device: torch.device):
        n = mol.GetNumAtoms() + 1

        graph = DGLGraph()
        graph.add_nodes(n)
        graph.add_edges(graph.nodes(), graph.nodes())
        graph.add_edges(range(1, n), 0)
        # graph.add_edges(0, range(1, n))
        for e in mol.GetBonds():
            u, v = e.GetBeginAtomIdx(), e.GetEndAtomIdx()
            graph.add_edge(u + 1, v + 1)
            graph.add_edge(v + 1, u + 1)
        adj = graph.adjacency_matrix(transpose=False).to_dense()

        v, m = feature.mol_feature(mol)
        vec = torch.cat([
            torch.zeros((1, m)), v
        ]).to(device)

        return ChebNetData(n, adj, vec)
    
    def forward(self, data):
        data: ChebNetData

        x0 = self.embed(data.adj, data.vec)
        x = self.activate(x0)
        y0 = self.conv(data.adj, x)
        y = self.activate(y0)

        if self.no_shortcut:
            z = self.fc(y[0])
        else:
            z = self.fc(y[0] + x[0])

        return z