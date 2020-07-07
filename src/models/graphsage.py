from typing import NamedTuple

from .base import *
from . import feature
import log

from dgl import DGLGraph
from dgl.nn.pytorch import SAGEConv, DenseSAGEConv

class GraphSAGEData(NamedTuple):
    n: int
    adj: torch.Tensor
    vec: torch.Tensor

class GraphSAGE(EmbeddableModel):
    def __init__(self, device: torch.device, *,
        embedding_dim: int = 64,
        aggregator_type: str = 'pool',
        no_shortcut: bool = False,
        **kwargs
    ):
        super().__init__(device)
        self.embedding_dim = embedding_dim
        self.no_shortcut = no_shortcut
        # self.aggregator_type = aggregator_type
        self.embed_layer = DenseSAGEConv(feature.ATOM_FDIM, embedding_dim)
        self.conv_layer = DenseSAGEConv(embedding_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 2)
        self.activate = nn.Tanh()

    @staticmethod
    def process(mol: Mol, device: torch.device, **kwargs):
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

        return GraphSAGEData(n, adj, vec)

    def embed(self, data: GraphSAGEData):
        x0 = self.embed_layer(data.adj, data.vec)
        x = self.activate(x0)
        y0 = self.conv_layer(data.adj, x)
        y = self.activate(y0)

        if self.no_shortcut:
            return y[0]
        else:
            return y[0] + x[0]

    def forward(self, data):
        return self.fc(self.embed(data))