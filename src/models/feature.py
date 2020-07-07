from typing import Tuple, List, Union

import torch
from rdkit.Chem import Mol, Atom, HybridizationType
from rdkit import Chem

# see docstring of `atom_feature`
FEATURE_DIM = 63
OFFSET = torch.tensor([  # mass excluded
    0, 25, 7, 4, 7, 5, 6, 6, 1,
]).cumsum(dim=0)
ATOM_MAP = {
    6: 0, 8: 1, 7: 2, 15: 3, 16: 4,
    17: 5, 11: 6, 9: 7, 35: 8, 51: 9,
    19: 10, 20: 11, 64: 12, 53: 13, 3: 14,
    83: 15, 33: 16, 80: 17, 30: 18, 14: 19,
    82: 20, 26: 21, 78: 22, 34: 23, 27: 24
}
HYBRID_MAP = {
    HybridizationType.SP2: 0,
    HybridizationType.SP3: 1,
    HybridizationType.S: 2,
    HybridizationType.SP: 3,
    HybridizationType.UNSPECIFIED: 4,
    HybridizationType.SP3D2: 5,
    HybridizationType.SP3D: 6
}


def atom_feature(atom: Atom) -> Tuple[torch.Tensor, int]:
    '''
    Extract atom feature vector.

    Returns: (feature_vec, feature_dim)

    Features:
        # item            # dim
        atomic number     25
        exp-valence       7
        imp-valence       4
        hybridization     7
        Hs                5
        degree            6
        formal charge     6
        in ring           1
        aromatic          1
        mass / 100        1
        (sum)             63
    '''

    index = torch.tensor([
        ATOM_MAP[atom.GetAtomicNum()],
        atom.GetExplicitValence(),
        atom.GetImplicitValence(),
        HYBRID_MAP[atom.GetHybridization()],
        atom.GetTotalNumHs(),
        atom.GetDegree(),
        atom.GetFormalCharge() + 2,  # -2 ~ 3
        int(atom.IsInRing()),
        int(atom.GetIsAromatic())
    ]) + OFFSET

    vec = torch.zeros(FEATURE_DIM)
    vec[index] = 1.0
    vec[-1] = atom.GetMass() / 100

    return vec, FEATURE_DIM

def mol_feature(mol: Mol) -> Tuple[torch.Tensor, int]:
    '''
    Similar to `atom_feature`.
    '''
    return torch.cat([
        atom_feature(atom)[0][None, :]
        for atom in mol.GetAtoms()
    ]), FEATURE_DIM

def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def bond_features(bond: Chem.rdchem.Bond) -> List[ Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.
    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond

def total_bond_feature(mol: Mol) -> Tuple[torch.Tensor, int]:
    '''
    Extract bond features.

    Returns: (feature_vec, feature_dim)
    '''
    f_atoms = [atom for atom in mol.GetAtoms()]
    n_atoms = len(f_atoms)

    f_bonds = [[0] * BOND_FDIM]

    for a1 in range(n_atoms):
        for a2 in range(a1+1, n_atoms):
            bond = mol.GetBondBetweenAtoms(a1, a2)
            
            if bond is None:
                continue
            
            f_bond = bond_features(bond)
            f_bonds.append(f_bond)

    return torch.tensor(f_bonds), BOND_FDIM