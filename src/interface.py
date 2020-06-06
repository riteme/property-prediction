from typing import Text, Any, Type, Optional, Sequence, List

from model import BaseModel

import torch
from rdkit.Chem import MolFromSmiles

class ModelInterface:
    def __init__(self, model: Type[BaseModel], dev: Optional[torch.device] = None):
        self.inst = model(dev)

    def process(self, smiles: Text) -> Any:
        mol = MolFromSmiles(smiles)
        assert mol is not None, 'Failed to parse SMILES string'
        result = self.inst.process(mol)
        return result

    def forward(self, batch: Sequence[Any]) -> torch.Tensor:
        result = torch.zeros((len(batch), 2))
        for i, data in enumerate(batch):
            result[i, :] = self.inst.forward(data)
        return result

    def predict(self, batch: Sequence[Any]) -> torch.Tensor:
        with torch.no_grad():
            result = self.forward(batch)
            index = result.argmax(dim=1)
        return index