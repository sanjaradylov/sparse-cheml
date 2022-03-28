import pandas as pd

from sparse_cheml.feature_extraction import (
    DescriptorTransformer, ECFP, create_feature_space,
    make_descriptor_df_from_smiles)


SMILES_STRINGS = ('CCO', 'N#N', 'CCCOC', 'C')
ECFP_TR = ECFP()
DESCRIPTOR_TR = DescriptorTransformer()


def test_make_descriptor_df_from_smiles():
    descriptor_df = make_descriptor_df_from_smiles(SMILES_STRINGS)

    assert isinstance(descriptor_df, pd.DataFrame)
    assert len(SMILES_STRINGS) == descriptor_df.shape[0]
    assert len(DESCRIPTOR_TR.descriptor_map) == descriptor_df.shape[1]


def test_create_feature_space():
    feature_space = create_feature_space(SMILES_STRINGS)

    assert isinstance(feature_space, pd.DataFrame)
    assert len(SMILES_STRINGS) == feature_space.shape[0]

    feature_dim = len(DESCRIPTOR_TR.descriptor_map) + ECFP_TR.n_bits
    assert feature_dim == feature_space.shape[1]
