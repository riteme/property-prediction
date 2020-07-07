from typing import NamedTuple

from .base import BaseModel
from . import feature
import log

import torch
from torch import nn
from rdkit.Chem import Mol
from torch_geometric.data import Data

from torch_geometric.nn import NNConv

class MPNNData(NamedTuple):
    n: int
    x: torch.Tensor
    edge_index: torch.Tensor
    m: int
    edge_attr: torch.Tensor

class MPNN(BaseModel):
    def __init__(self, device: torch.device, *,
        embedding_dim: int = 64,
        no_shortcut: bool = False,
        **kwargs
    ):
        super().__init__(device)
        self.embedding_dim = embedding_dim
        self.no_shortcut = no_shortcut
        # self.aggregator_type = aggregator_type
        self.embed = NNConv(feature.FEATURE_DIM, embedding_dim, nn.Sequential(nn.Linear(feature.BOND_FDIM, feature.FEATURE_DIM*embedding_dim),nn.ReLU()))
        self.conv2= NNConv(embedding_dim, embedding_dim, nn.Sequential(nn.Linear(feature.BOND_FDIM, embedding_dim*embedding_dim,nn.ReLU())))
        self.fc = nn.Linear(embedding_dim, 2)
        self.activate = nn.ReLU()
    
    @staticmethod
    def process(mol: Mol, device: torch.device):
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
        
        for i in range(1,n):
            a1.append(i)
            a2.append(0)
            cnt += 1
            f_bonds.append([0] * feature.BOND_FDIM)
    
        # graph.add_edges(0, range(1, n))
        for e in mol.GetBonds():
            u, v = e.GetBeginAtomIdx(), e.GetEndAtomIdx()
            a1.append(u+1)
            a2.append(v+1)
            a1.append(v+1)
            a2.append(u+1)
            bond = mol.GetBondBetweenAtoms(u, v)
            f_bond = feature.bond_features(bond)
            f_bonds.append(f_bond)
            f_bonds.append(f_bond)
            cnt += 2
            # graph.add_edge(u + 1, v + 1)
            # graph.add_edge(v + 1, u + 1)
        # adj = graph.adjacency_matrix(transpose=False).to_dense()
        edge_index = torch.tensor([a1,a2],dtype=torch.long)

        v, m = feature.mol_feature(mol)
        vec = torch.cat([
            torch.zeros((1, m)), v
        ]).to(device)

        import numpy as np
        # print("shape:", np.array(f_bonds).shape)
        # print("cnt:", cnt)
        # edge_attr = torch.rand(cnt, feature.BOND_FDIM)
        edge_attr = torch.tensor(f_bonds, dtype=torch.float32)
        # print("size:", edge_attr.shape)
        # edge_attr = torch.rand(cnt, feature.BOND_FDIM)
        return MPNNData(n, vec, edge_index, cnt, edge_attr)
    
    def forward(self, data):
        data: MPNNData

        # edge_addr = torch.rand(data.m, 10)
        edge_addr = data.edge_attr

        x0 = self.embed(data.x, data.edge_index, edge_addr)
        x = self.activate(x0)
        y0 = self.conv2(x, data.edge_index, edge_addr)
        y = self.activate(y0)

        if self.no_shortcut:
            z = self.fc(y[0])
        else:
            z = self.fc(y[0] + x[0])

        return z