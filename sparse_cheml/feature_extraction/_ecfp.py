"""Create molecular fingerprints."""

__all__ = ('ECFP',)

import numpy as np
from scipy import sparse

from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_scalar
from sklearn.utils.validation import check_is_fitted


class ECFP(BaseEstimator, TransformerMixin):
    """Apply Morgan algorithm to set of compounds to get circular fingerprints.

    Parameters
    ----------
    radius : int, default 2
        The radius of fingerprint.
    n_bits : int, default 1024
        The number of bits.
    return_type : {'ndarray', 'csr_sparse', 'bitvect_list'}, default 'ndarray'
        Whether to return csr-sparse matrix, numpy array, or list of rdkit bit
        vectors.

    Examples
    --------
    >>> from sparse_cheml.preprocessing import MoleculeTransformer
    >>> molecule_tr = MoleculeTransformer()
    >>> ecfp_tr = ECFP(radius=2, n_bits=1024)
    >>> from sklearn.pipeline import make_pipeline
    >>> pipe = make_pipeline(molecule_tr, ecfp_tr)
    """

    def __init__(self, radius=2, n_bits=2048, return_type='ndarray'):
        check_scalar(radius, 'radius', int, min_val=1)
        check_scalar(n_bits, 'number of bits', int, min_val=1)
        valid_return_types = {'ndarray', 'csr_sparse', 'bitvect_list'}
        if return_type not in valid_return_types:
            raise ValueError(
                f'`return_type` must be in {valid_return_types}, '
                f'not {return_type!r}')

        self.radius = radius
        self.n_bits = n_bits
        self.return_type = return_type

    def fit(self, unused_molecules, unused_y=None):
        """Update internal state."""
        self.n_features_in_ = 1
        self.n_features_out_ = self.n_bits
        return self

    def transform(self, molecules):
        """Return circular fingerprints as bit vectors.

        Parameters
        ----------
        molecules : iterable of rdkit.Chem.Mol
            RDKit molecules.

        Returns
        -------
        np.array or scipy.sparse.csr_matrix, dtype=np.uint8
        """
        check_is_fitted(self)

        fingerprints = [
            GetMorganFingerprintAsBitVect(molecule, self.radius, self.n_bits)
            for molecule in molecules]

        if self.return_type == 'ndarray':
            return np.array(fingerprints, dtype=np.uint8)
        elif self.return_type == 'csr_sparse':
            return sparse.csr_matrix(fingerprints, dtype=np.uint8)
        elif self.return_type == 'bitvect_list':
            return fingerprints

    def get_feature_names_out(self, unused_input_features=None):
        """Get an array of ``ecfp_{i}`` strings, ``i <= self.n_bits``."""
        check_is_fitted(self)

        return np.array([
            f'ecfp{self.radius}_{str(i).zfill(len(str(self.n_bits)))}'
            for i in range(self.n_bits)])
