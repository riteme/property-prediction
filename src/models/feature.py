from typing import Tuple

import torch
from rdkit.Chem import Mol

def atom_features(mol: Mol) -> Tuple[torch.Tensor, int]:
    '''
    Extract atom features.

    Returns: (feature_vec, feature_dim)
    '''
    ...