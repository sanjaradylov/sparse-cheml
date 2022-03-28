"""Functions to union molecular features."""

__all__ = ('create_feature_space',)

import pandas as pd
from joblib import delayed, Parallel

from ._descriptors import DescriptorTransformer
from ._ecfp import ECFP
from ..preprocessing import check_compounds_valid


def create_feature_space(smiles_strings, transformers=None, n_jobs=None,
                         **mol_kwargs):
    """Create molecular feature space using sklearn transformers.

    Parameters
    ----------
    smiles_strings : iterable of str
        SMILES strings of molecules.
    transformers : iterable of scikit-learn transformers, default=None
        Scikit-learn-compatible transformers. Defaults to
        ``(ECFP(), DescriptorTransformer())``

    Returns
    -------
    pandas.DataFrame
    """

    def transform(transformer):
        return pd.DataFrame(transformer.fit_transform(molecules),
                            columns=transformer.get_feature_names_out())

    if n_jobs is not None and n_jobs > 1:  # FIXME Should we parallelize?
        raise ValueError('Does not support n_jobs > 1')

    transformers = transformers or get_default_transformers()

    molecules = check_compounds_valid(smiles_strings, 'nan', **mol_kwargs)

    descriptor_stack = Parallel(n_jobs=n_jobs)(delayed(transform)(t)
                                               for t in transformers)

    feature_space = pd.DataFrame()
    for descriptors in descriptor_stack:
        feature_space = feature_space.join(descriptors, how='right')

    return feature_space


def get_default_transformers():
    return (ECFP(), DescriptorTransformer())
