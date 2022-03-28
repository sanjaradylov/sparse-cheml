"""Cross-validation tools."""

__all__ = ('ScaffoldKFold',)

from collections import defaultdict

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import FastFindRings
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.model_selection._split import _BaseKFold


class ScaffoldKFold(_BaseKFold):
    """Scikit-learn-compatible K-fold CV that groups molecules by scaffolds.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    scaffold_func : {'decompose', 'smiles'}, default='decompose'
        The function to use for computing scaffolds, which can be
        'decompose' for using
        ``rdkit.Chem.AllChem.MurckoDecompose`` or
        'smiles' for using
        ``rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles``.
    """

    def __init__(self, n_splits=5, scaffold_func='decompose'):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)

        valid_scaffold_funcs = {'decompose', 'smiles'}
        if scaffold_func not in valid_scaffold_funcs:
            raise ValueError(
                r'scaffold_func must be one of {valid_scaffold_funcs}, '
                r'not {scaffold_func!r}')
        self.scaffold_func = scaffold_func

    def _iter_test_indices(self, X, y=None, groups=None):
        scaffold_sets = self.get_ordered_scaffold_sets(X)

        index_buckets = [[] for _ in range(self.n_splits)]

        for group_indices in scaffold_sets:
            bucket_chosen = int(np.argmin([len(bucket)
                                           for bucket in index_buckets]))
            index_buckets[bucket_chosen].extend(group_indices)

        yield from index_buckets

    def get_ordered_scaffold_sets(self, molecules):
        # pylint: disable=line-too-long
        """Group molecules by their Bemis-Murcko scaffolds and order the groups by their sizes.

        The order is decided by comparing the size of groups, where groups with
        a larger size are placed before the ones with a smaller size.

        Parameters
        ----------
        molecules : list of rdkit.Chem.rdchem.Mol
            Pre-computed RDKit molecule instances.

        Returns
        -------
        scaffold_sets : list
            Each element of the list is a list of int,
            representing the indices of compounds with a same scaffold.

        References
        ----------
        .. [1] `DGL-lifesci ScaffoldSplitter.
           <https://lifesci.dgl.ai/_modules/dgllife/utils/splitters.html#ScaffoldSplitter.get_ordered_scaffold_sets>`_
        """
        scaffolds = defaultdict(list)
        for i, mol in enumerate(molecules):
            # For mols that have not been sanitized, we need to compute their
            # ring information
            FastFindRings(mol)
            if self.scaffold_func == 'decompose':
                mol_scaffold = Chem.MolToSmiles(AllChem.MurckoDecompose(mol
                                                                        ))
            if self.scaffold_func == 'smiles':
                mol_scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False)
            # Group molecules that have the same scaffold
            scaffolds[mol_scaffold].append(i)

        # Order groups of molecules by first comparing the size of groups
        # and then the index of the first compound in the group.
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]),
                reverse=True)
        ]

        return scaffold_sets
