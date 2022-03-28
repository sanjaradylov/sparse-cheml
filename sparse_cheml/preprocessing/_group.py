"""Group features for Sparse Group LASSO."""

__all__ = ('DEFAULT_GROUP_PREFIXES', 'make_groups_groupyr',
           'make_group_map_groupyr', 'make_groups_group_lasso')

import numpy as np


_REMAINING = '<REMAINING>'
DEFAULT_GROUP_PREFIXES = (
    'ecfp', 'fr_', 'FpDensityMorgan', 'PEOE', 'VSA_', 'EState_', 'SlogP_',
    'SMR_', 'Kappa', 'Chi', 'BCUT2D', 'MolWt', 'EStateIndex', 'PartialCharge',
)


def make_groups_groupyr(feature_names, group_prefixes=None):
    """Get group IDs of features for estimators from ``groupyr`` package.

    Parameters
    ----------
    feature_names : iterable of str
    group_prefixes : sequence of str, default=None
        Defaults to ``sparse_cheml.preprocessing.DEFAULT_GROUP_PREFIXES``.

    Returns
    -------
    list of ndarray, dtype=int
    """
    group_map = make_group_map_groupyr(feature_names, group_prefixes)
    return [np.array(value) for key, value in group_map.items()]


def make_groups_group_lasso(feature_names, group_prefixes=None):
    """Get group IDs of features for estimators from ``group_lasso`` package.

    Parameters
    ----------
    feature_names : iterable of str
    group_prefixes : sequence of str, default=None
        Defaults to ``sparse_cheml.preprocessing.DEFAULT_GROUP_PREFIXES``.

    Returns
    -------
    ndarray, dtype=int
    """
    group_prefixes = group_prefixes or DEFAULT_GROUP_PREFIXES
    group_map = {key: index
                 for index, key in enumerate(group_prefixes, start=1)}
    groups = [-1] * len(feature_names)

    for feature_index, feature in enumerate(feature_names):
        for group_key, group_index in group_map.items():
            if group_key in feature:
                groups[feature_index] = group_index

    return np.array(groups)


def make_group_map_groupyr(feature_names, group_prefixes=None):
    """Get group-to-count map for estimators from ``groupyr`` package.

    Parameters
    ----------
    feature_names : iterable of str
    group_prefixes : sequence of str, default=None
        Defaults to ``sparse_cheml.preprocessing.DEFAULT_GROUP_PREFIXES``.

    Returns
    -------
    dict, str -> int
    """
    group_prefixes = group_prefixes or DEFAULT_GROUP_PREFIXES
    group_map = dict.fromkeys(group_prefixes + (_REMAINING,), tuple())

    for feature_index, feature in enumerate(feature_names):
        for key in group_map:
            if key in feature:
                group_map[key] += (feature_index,)
                break
        else:
            group_map[_REMAINING] += (feature_index,)

    return group_map
