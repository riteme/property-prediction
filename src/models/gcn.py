from typing import NamedTuple

from .base import BaseModel
import log

import torch
from torch import nn
from rdkit.Chem import Mol
from dgl import DGLGraph

# NOTICE: GraphConv is somewhat slow. Use DenseGraphConv instead.
from dgl.nn.pytorch import GraphConv, DenseGraphConv

class GCNData(NamedTuple):
    n: int
    adj: torch.Tensor
    feature: torch.Tensor

class GCN(BaseModel):
    def __init__(self, dev,
        feature_dim: int = 32,
        embedding_dim: int = 64
    ):
        super().__init__(dev)
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim

        self.embed = DenseGraphConv(feature_dim, embedding_dim)
        self.conv = DenseGraphConv(embedding_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 2)
        self.activate = nn.ReLU()

    def process(self, mol: Mol, atom_map):
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

        feature = torch.cat([
            torch.zeros((1, self.feature_dim), device=self.device),  # node 0
            torch.nn.functional.one_hot(
                torch.tensor(
                    [atom_map[u.GetAtomicNum()] for u in mol.GetAtoms()],
                    device=self.device
                ), num_classes=self.feature_dim
            ).to(torch.float)
        ])

        return GCNData(n, adj, feature)

    def forward(self, data):
        data: GCNData

        x0 = self.embed(data.adj, data.feature)
        x = self.activate(x0)
        y0 = self.conv(data.adj, x)
        y = self.activate(y0)
        z = self.fc(y[0])
        return z
