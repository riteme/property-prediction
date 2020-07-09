from dataclasses import dataclass

from .base import BaseModel
from . import feature
import log

import torch
from torch import nn
from rdkit.Chem import Mol
from torch_geometric.data import Data

from torch_geometric.nn import GatedGraphConv


@dataclass
class GatedGraphData:
    n: int
    x: torch.Tensor
    edge_index: torch.Tensor
    m: int
    # edge_attr: torch.Tensor


class GatedGNN(BaseModel):
    def __init__(self, device: torch.device, *,
        embedding_dim: int = 64,
        no_shortcut: bool = False,
        **kwargs
    ):
        super().__init__(device)
        self.embedding_dim = embedding_dim
        self.no_shortcut = no_shortcut
        # self.aggregator_type = aggregator_type
        self.embed = GatedGraphConv(embedding_dim, 2, aggr='mean')
        self.conv2= GatedGraphConv(embedding_dim, 2, aggr='mean')
        self.fc = nn.Linear(embedding_dim, 2)
        self.activate = nn.ELU()

    @staticmethod
    def decode_data(data: GatedGraphData, device: torch.device, **kwargs) -> GatedGraphData:
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        return data

    @staticmethod
    def process(mol: Mol, device: torch.device, **kwargs):
        n = mol.GetNumAtoms() + 1

        # graph = DGLGraph()
        # graph.add_nodes(n)
        # graph.add_edges(graph.nodes(), graph.nodes())
        # graph.add_edges(range(1, n), 0)

        a1 = []
        a2 = []
        cnt = 0

        f_bonds = []

        # for i in range(0,n):
        #     a1.append(i)
        #     a2.append(i)
        #     cnt += 1

        for i in range(1, n):
            a1.append(i)
            a2.append(0)
            cnt += 1
            f_bonds.append([0] * feature.BOND_FDIM)

        # graph.add_edges(0, range(1, n))
        for e in mol.GetBonds():
            u, v = e.GetBeginAtomIdx(), e.GetEndAtomIdx()
            a1.append(u + 1)
            a2.append(v + 1)
            a1.append(v + 1)
            a2.append(u + 1)
            # bond = mol.GetBondBetweenAtoms(u, v)
            # f_bond = feature.bond_features(bond)
            # f_bonds.append(f_bond)
            # f_bonds.append(f_bond)
            # cnt += 2
            # graph.add_edge(u + 1, v + 1)
            # graph.add_edge(v + 1, u + 1)
        # adj = graph.adjacency_matrix(transpose=False).to_dense()
        edge_index = torch.tensor([a1, a2], dtype=torch.long, device=device)

        v, m = feature.mol_feature(mol)
        vec = torch.cat([
            torch.zeros((1, m)), v
        ]).to(device)

        # edge_attr = torch.rand(cnt, feature.BOND_FDIM)
        # edge_attr = torch.tensor(f_bonds, dtype=torch.float32, device=device)

        return GatedGraphData(n, vec, edge_index, cnt)

    def forward(self, data):
        data: GatedGraphData

        # edge_attr = torch.rand(data.m, 10)
        # edge_attr = data.edge_attr

        x0 = self.embed(data.x, data.edge_index)
        x = self.activate(x0)
        y0 = self.conv2(x, data.edge_index)
        y = self.activate(y0)

        if self.no_shortcut:
            z = self.fc(y[0])
        else:
            z = self.fc(y[0] + x[0])

        return z