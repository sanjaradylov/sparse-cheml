import numpy as np
from rdkit.Chem import MolFromSmiles

from sparse_cheml.model_selection import ScaffoldKFold


SMILES_STRINGS = ('CCO', 'N#N', 'C#N', 'C', 'CCCOC', 'CC(=O)O', 'c1ccccc1',
                  'c1cnc[nH]c(=O)1')


def test_ScaffoldKFold():
    molecules = list(map(MolFromSmiles, SMILES_STRINGS))
    cv = ScaffoldKFold(n_splits=2, scaffold_func='decompose')
    n_strings = len(SMILES_STRINGS)

    for train_idx, test_idx in cv.split(molecules):
        assert (
            (np.sort(np.append(train_idx, test_idx))
             == np.arange(n_strings)).sum() == n_strings
        )
