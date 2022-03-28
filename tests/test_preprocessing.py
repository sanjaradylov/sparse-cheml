import pytest
from rdkit.Chem import Mol

from sparse_cheml.preprocessing import check_compounds_valid


VALID_SMILES_STRINGS = ('CCO', 'N#N', 'C#N')
INVALID_SMILES_STRINGS = ('c1ccc', 'invalid')


def test_check_compounds_valid():
    valid_molecules = check_compounds_valid(VALID_SMILES_STRINGS,
                                            invalid='nan')
    assert len(VALID_SMILES_STRINGS) == valid_molecules.shape[0]
    assert all(isinstance(m, Mol) for m in valid_molecules)

    with pytest.raises(ValueError):
        check_compounds_valid(INVALID_SMILES_STRINGS, invalid='raise')
