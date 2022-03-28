"""Transform SMILES into RDKit molecules."""

__all__ = ('check_compounds_valid', 'MoleculeTransformer')

import numpy as np
from rdkit.Chem import MolFromSmiles
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class InvalidSMILESError(ValueError):
    pass


def check_compounds_valid(smiles_strings, invalid='nan', **converter_kwargs):
    """Convert SMILES compounds into RDKit molecules.

    Parameters
    ----------
    smiles_strings : iterable of str
        SMILES strings of molecules.
    invalid : {'nan', 'raise'}, default='nan'
        Whether to a) replace invalid SMILES with ``numpy.NaN``, b) raise
        ``InvalidSMILESError``.

    Other Parameters
    ----------------
    converter_kwargs : dict, default={}
        Optional key-word arguments for ``rdkit.Chem.MolFromSmiles``.

    Returns
    -------
    molecules : list of rdkit.Chem.Mol

    Raises
    ------
    TypeError
        If any compound is invalid.
    """
    molecules = []

    for compound in smiles_strings:
        molecule = MolFromSmiles(compound, **converter_kwargs)
        if molecule is not None:
            molecules.append(molecule)
        elif invalid == 'nan':
            molecules.append(np.NaN)
        elif invalid == 'raise':
            raise InvalidSMILESError(
                f'cannot convert {compound!r} into molecule: invalid compound')

    return np.array(molecules, dtype=object)


class MoleculeTransformer(BaseEstimator, TransformerMixin):
    """Convert SMILES strings into RDKit molecules.

    Parameters
    ----------
    invalid : {'nan', 'raise'}, default='nan'
        Whether to a) replace invalid SMILES with ``numpy.NaN``, b) raise
        ``InvalidSMILESError``.
    converter_kwargs : dict, default={}
        Optional key-word arguments for ``rdkit.Chem.MolFromSmiles``.

    Examples
    --------
    >>> from rdkit.Chem import Mol
    >>> mt = MoleculeTransformer(invalid='nan')
    >>> smiles = ('CCO', 'C#N', 'N#N', 'invalid_smiles')
    >>> molecules = mt.fit_transform(smiles)
    """

    def __init__(self, invalid='nan', **converter_kwargs):
        self.invalid = invalid
        self.converter_kwargs = converter_kwargs

    def fit(self, unused_smiles_strings, unused_y=None):
        """Update internal state."""
        self.n_features_in_ = 1
        self.n_features_out_ = 1
        return self

    def transform(self, smiles_strings):
        """Get molecules from SMILES strings."""
        check_is_fitted(self)

        return check_compounds_valid(
            smiles_strings, invalid=self.invalid, **self.converter_kwargs)

    def get_feature_names_out(self, unused_input_features=None):
        check_is_fitted(self)

        return np.array(['Mol'])
