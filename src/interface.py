from typing import Text, Any, Type, Optional, Sequence, List

from model import BaseModel

import torch
from rdkit.Chem import MolFromSmiles

class ModelInterface:
    def __init__(self, model: Type[BaseModel], dev: Optional[torch.device] = None):
        self.inst = model().to(dev)

    def process(self, smiles: Text) -> Any:
        mol = MolFromSmiles(smiles)
        assert mol is not None, 'Failed to parse SMILES string'
        result = self.inst.process(mol)
        return result

    def forward(self, batch: Sequence[Any]) -> torch.Tensor:
        ...

    def predict(self, batch: Sequence[Any]) -> List[float]:
        ...
        # with torch.no_grad():
        #     ...