"""Create descriptor data frames from SMILES sequences."""

__all__ = ('make_descriptor_df_from_mol', 'make_descriptor_df_from_smiles',
           'DescriptorTransformer')

from functools import partial

import numpy as np
from pandas import DataFrame
from rdkit.Chem import Descriptors, MolFromSmiles
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _create_def_from_lambda(new_name, lambda_name):
    exec(f'def {new_name}(mol, *a, **kw): return lambda_name(mol, *a, **kw)')
    return eval(new_name)


# pylint: disable=protected-access
_RDKIT_DESCRIPTOR_MAP = dict(Descriptors._descList)


def make_descriptor_df_from_mol(molecules, function_map=None):
    """Return dataframe of descriptors generated using `function_map`.

    Parameters
    ----------
    molecules : iterable of rdkit.Chem.Mol
        RDKit molecules
    function_map : dict, str -> callable : rdkit.Chem.Mol -> numbers.Real
        Keys represent RDKit descriptor name (see RDKit Descriptors) and
        values functions to calculate the corresponding descriptors.
        Defaults to RDKIT's available descriptors.

    Returns
    -------
    pandas.DataFrame
        Dataframe of calculated descriptors for each molecule.
    """
    function_map = function_map or _RDKIT_DESCRIPTOR_MAP

    data_frame = DataFrame()

    for name, function in function_map.items():
        descriptor = [function(molecule) for molecule in molecules]
        data_frame = data_frame.assign(**{name: descriptor})

    return data_frame


def make_descriptor_df_from_smiles(molecules, function_map=None, **mol_kw):
    """Return dataframe of descriptors generated using `function_map`.

    Parameters
    ----------
    molecules : iterable of str
        SMILES strings.
    function_map : dict, str -> callable : rdkit.Chem.Mol -> numbers.Real
        Keys represent RDKit descriptor name (see RDKit Descriptors) and
        values functions to calculate the corresponding descriptors.
        Defaults to RDKIT's available descriptors.
    mol_kw : dict, default=None
        ``rdkit.Chem.MolFromSmiles`` positional arguments.

    Returns
    -------
    pandas.DataFrame
        Dataframe of calculated descriptors for each molecule.
    """
    mol_from_smiles = partial(MolFromSmiles, **mol_kw)
    molecules = tuple(map(mol_from_smiles, molecules))

    return make_descriptor_df_from_mol(molecules, function_map)


class DescriptorTransformer(BaseEstimator, TransformerMixin):
    """Generate descriptor dataframe from molecules.

    Parameters
    ----------
    descriptor_map : dict, str -> callable, rdkit.Chem.Mol -> numbers.Real,
                     default=None
        Keys are feature names, values - callables to calculate descriptors.
        Defaults to RDKIT's available descriptors.

    Examples
    --------
    >>> from rdkit.Chem import Descriptors
    >>> desc_map = dict(Descriptors._descList)
    >>> rdkit_descriptor_tr = DescriptorTransformer(desc_map)
    """

    def __init__(self, descriptor_map=None):
        self.descriptor_map = descriptor_map or _RDKIT_DESCRIPTOR_MAP

    def fit(self, unused_molecules, unused_y=None):
        """Update internal state."""
        self.n_features_in_ = 1
        self.n_features_out_ = len(self.descriptor_map)
        return self

    def transform(self, molecules):
        """Return calculated RDKit descriptors.

        Parameters
        ----------
        molecules : iterable of rdkit.Chem.Mol
            RDKit molecules.

        Returns
        -------
        descriptors : pandas.DataFrame,
            shape = (``len(molecules)``, ``len(self.descriptor_names_)``)
        """
        check_is_fitted(self)

        if isinstance(molecules, np.ndarray) and len(molecules.shape) > 1:
            molecules = molecules[:, 0]

        return make_descriptor_df_from_mol(
            molecules=molecules, function_map=self.descriptor_map)

    def get_feature_names_out(self, unused_input_features=None):
        """Get names of fitted descriptors."""
        check_is_fitted(self)

        return np.array(list(self.descriptor_map.keys()))
