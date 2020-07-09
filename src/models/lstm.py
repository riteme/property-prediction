from .base import *
from rdkit.Chem import AllChem as chem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


PACKING_DIM = 64

class LSTM(EmbeddableModel):
    def __init__(self,
        device: torch.device,
        embedding_dim: int = 16,
        num_layer: int = 1,
        **kwargs
    ):
        super().__init__(device)
        self.embedding_dim = embedding_dim
        self.num_layer = num_layer

        self.seq = nn.LSTM(
            input_size=PACKING_DIM,
            hidden_size=embedding_dim,
            num_layers=num_layer,
            batch_first=True
        )
        self.fc = nn.Linear(
            in_features=embedding_dim,
            out_features=2
        )

    @staticmethod
    def decode_data(data: torch.Tensor, device: torch.device, **kwargs) -> torch.Tensor:
        return data.to(device)

    @staticmethod
    def process(mol: Mol, device: torch.device, **kwargs) -> torch.Tensor:
        bitvect = chem.RDKFingerprint(mol).ToBitString()
        vec = torch.tensor(list(
            map(float, bitvect)
        ), device=device)
        return vec.reshape(-1, PACKING_DIM)

    def embed(self, data: torch.Tensor) -> torch.Tensor:
        output, _ = self.seq(data[None, :])
        return output[0][-1]

    def forward(self, data):
        vec = self.embed(data)
        return self.fc(vec)
