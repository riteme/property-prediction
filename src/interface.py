from typing import Text, Any, Type, Optional, Sequence, List, Dict

from model import BaseModel

import torch
from rdkit.Chem import MolFromSmiles

class ModelInterface:
    def __init__(self, model: Type[BaseModel], dev: Optional[torch.device] = None):
        self.atom_map: Dict[int, int] = {}
        self.inst = model(dev=dev).to(dev)

    def process(self, smiles: Text) -> Any:
        '''Parse molecule
        '''

        mol = MolFromSmiles(smiles)
        assert mol is not None, 'Failed to parse SMILES string'

        for u in mol.GetAtoms():
            num = u.GetAtomicNum()
            if num not in self.atom_map:
                idx = len(self.atom_map)
                self.atom_map[num] = idx

        result = self.inst.process(mol, self.atom_map)
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