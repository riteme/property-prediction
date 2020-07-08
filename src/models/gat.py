from dataclasses import dataclass

from .base import *
from . import feature
import log

from dgl import DGLGraph
from dgl.nn.pytorch import GATConv


@dataclass
class GATData:
    n: int
    graph: DGLGraph
    vec: torch.Tensor


class GAT(EmbeddableModel):
    def __init__(self, device: torch.device, *,
        embedding_dim: int = 64,
        no_shortcut: bool = False,
        **kwargs
    ):
        super().__init__(device)
        self.embedding_dim = embedding_dim
        self.no_shortcut = no_shortcut

        use_residual = not no_shortcut
        self.embed_layer = GATConv(feature.ATOM_FDIM, embedding_dim, 4, residual=use_residual)
        self.conv_layer = GATConv(embedding_dim, embedding_dim, 6, residual=use_residual)
        self.fc = nn.Linear(embedding_dim, 2)
        self.activate = nn.ELU()

    @staticmethod
    def decode_data(data: GATData, device: torch.device, **kwargs) -> GATData:
        data.vec = data.vec.to(device)
        return data

    @staticmethod
    def process(mol: Mol, device: torch.device, **kwargs) -> GATData:
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

        v, m = feature.mol_feature(mol)
        vec = torch.cat([
            torch.zeros((1, m)), v
        ]).to(device)

        return GATData(n, graph, vec)

    def embed(self, data: GATData) -> torch.Tensor:
        x0 = self.embed_layer(data.graph, data.vec)
        x1 = torch.mean(x0, dim=1)
        x = self.activate(x1)
        y0 = self.conv_layer(data.graph, x)
        y1 = torch.mean(y0, dim=1)
        y = self.activate(y1)
        return y[0]

    def forward(self, data):
        return self.fc(self.embed(data))