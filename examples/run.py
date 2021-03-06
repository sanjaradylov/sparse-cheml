#! /usr/bin/env python

import argparse
import pathlib
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from scipy import stats

from optuna.distributions import (
    CategoricalDistribution, IntUniformDistribution, UniformDistribution)
from optuna.integration import OptunaSearchCV

from group_lasso import LogisticGroupLasso, GroupLasso
from groupyr import LogisticSGLCV, SGLCV
from lightning.classification import CDClassifier
from lightning.regression import CDRegressor

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNetCV, HuberRegressor, LogisticRegression,
    Ridge, RidgeClassifier, SGDRegressor)
from sklearn.model_selection import cross_validate, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sparse_cheml.feature_extraction import create_feature_space
from sparse_cheml.model_selection import ScaffoldKFold
from sparse_cheml.preprocessing import (
    check_compounds_valid, make_groups_groupyr, make_groups_group_lasso)


def process_options():
    parser = argparse.ArgumentParser(
        description=(
            'Train and evaluate sparse linear models '
            '(sparse group lasso and elastic net)'
        ),
        add_help=False,
        usage='python run.py DATA SMILES_FIELD TARGET_FIELD [OPTION]...',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_options = parser.add_argument_group('data')
    data_options.add_argument(
        'csv',
        help='Path to the csv with SMILES strings and properties.',
    )
    data_options.add_argument(
        'smiles_field',
        help='Column name of SMILES strings in the csv.',
    )
    data_options.add_argument(
        'target_field',
        help='Column name of target in the csv.',
    )

    task_options = parser.add_argument_group('ml task')
    task_options.add_argument(
        '-t', '--task',
        help='Supervised learning task',
        choices=('classification', 'regression'),
        default='regression',
    )
    task_options.add_argument(
        '-m', '--selector',
        help='Sparse model for feature selection',
        choices=('enet', 'lightning', 'sgl', 'groupyr'),
        default='enet',
    )
    task_options.add_argument(
        '-p', '--predictor',
        help='Dense model for predictions',
        choices=('huber_sgd', 'huber_lbfgsb', 'ridge', 'maxent'),
        default='huber_sgd',
    )
    task_options.add_argument(
        '-a', '--calibrate',
        help='Whether to calibrate classifier',
        action='store_true',
    )

    testing_options = parser.add_argument_group('validation and testing')
    testing_options.add_argument(
        '-c', '--cv',
        help='Cross-validation scheme',
        choices=('scaffold', 'random'),
        default='random',
    )
    testing_options.add_argument(
        '-e', '--metric_val',
        help='Evaluation metric [validation]',
        choices=('mae', 'rmse', 'medae', 'roc'),
        default='rmse',
    )
    testing_options.add_argument(
        '-E', '--metric_test',
        help='Evaluation metric [testing]',
        choices=('mae', 'rmse', 'medae', 'roc'),
        default='rmse',
    )
    testing_options.add_argument(
        '-s', '--n_splits_val',
        help='Number of splits [validation]',
        type=int,
        default=5,
    )
    testing_options.add_argument(
        '-S', '--n_splits_test',
        help='Number of splits [testing]',
        type=int,
        default=10,
    )
    testing_options.add_argument(
        '-r', '--n_rounds',
        help='Number of random search rounds [validation]',
        type=int,
        default=100,
    )

    other_options = parser.add_argument_group('other')
    other_options.add_argument(
        '-v', '--verbose',
        help='Print validation and testing messages',
        action='store_false',
    )
    other_options.add_argument(
        '-j', '--n_jobs',
        help='Number of parallel jobs [validation]',
        type=int,
        default=1,
    )
    other_options.add_argument(
        '-h', '--help',
        help='Show this help message and exit',
        action='help',
    )

    return parser.parse_args()


def load_data():
    df = pd.read_csv(options.csv, usecols=[options.smiles_field,
                                           options.target_field])
    X = create_feature_space(df[options.smiles_field])
    X = X.loc[:, X.std() != 0]
    y = df[options.target_field].to_numpy()
    return X, y


def create_cvs():
    if options.cv == 'scaffold':
        outer_cv = ScaffoldKFold(n_splits=options.n_splits_test)
        inner_cv = ShuffleSplit(n_splits=options.n_splits_val,
                                test_size=0.2, random_state=random_state)
    else:
        outer_cv = ShuffleSplit(n_splits=options.n_splits_test,
                                test_size=0.1, random_state=random_state)
        inner_cv = ShuffleSplit(n_splits=options.n_splits_val,
                                test_size=0.2, random_state=random_state)

    return outer_cv, inner_cv


def get_metric(metric):
    valid_metrics = ('mae', 'rmse', 'medae', 'roc')
    sklearn_metrics = ('neg_mean_absolute_error',
                       'neg_root_mean_squared_error',
                       'neg_median_absolute_error',
                       'roc_auc')
    return dict(zip(valid_metrics, sklearn_metrics))[metric]


def create_model_pipe():
    param_grid = dict()

    features = pd.Series(X.columns)
    feature_mask = features.apply(lambda f: 'ecfp' not in f)
    numeric_features = features[feature_mask]

    def get_preprocessing():
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler(with_mean=False)
        num_tr = Pipeline([('impute', imputer), ('scale', scaler)])
        transformers = [
            ('desc', num_tr, numeric_features),
        ]
        return ColumnTransformer(
            transformers=transformers,
            remainder='passthrough',
            sparse_threshold=0.3,
            n_jobs=options.n_jobs,
            verbose_feature_names_out=False,
        )

    def get_feature_selection():
        if options.task == 'regression':

            if options.selector == 'enet':
                param_grid[
                    f'{options.selector}__threshold'
                ] = UniformDistribution(1e-4, 1e-2)
                param_grid[
                    f'{options.selector}__estimator__tol'
                ] = UniformDistribution(1e-4, 1e-1)

                return SelectFromModel(ElasticNetCV(
                    l1_ratio=stats.beta(0.6, 0.4).rvs(max(2, options.n_rounds // 3)),
                    eps=5e-3,
                    n_alphas=max(2, options.n_rounds // 3),
                    max_iter=500,
                    cv=inner_cv,
                    random_state=random_state,
                    selection='random',
                ))

            elif options.selector == 'lightning':
                param_grid[
                    f'{options.selector}__threshold'
                ] = UniformDistribution(1e-4, 1e-2)
                param_grid[
                    f'{options.selector}__estimator__tol'
                ] = UniformDistribution(1e-3, 1e-1)
                param_grid[
                    f'{options.selector}__estimator__C'
                ] = UniformDistribution(1e-4, 1e+4)
                param_grid[
                    f'{options.selector}__estimator__alpha'
                ] = UniformDistribution(1e-4, 1e+4)
                param_grid[
                    f'{options.selector}__estimator__max_steps'
                ] = IntUniformDistribution(10, 30, 5)
                param_grid[
                    f'{options.selector}__estimator__beta'
                ] = UniformDistribution(0.2, 0.9)
                param_grid[
                    f'{options.selector}__estimator__sigma'
                ] = UniformDistribution(1e-3, 1e-2)

                return SelectFromModel(CDRegressor(
                    penalty='l1/l2',
                    loss='squared',
                    max_iter=200,
                    termination='violation_sum',
                    shrinking=True,
                    debiasing=False,
                    permute=True,
                    selection='uniform',

                    verbose=False,
                    n_jobs=1,
                    random_state=0,
                ))

            elif options.selector == 'groupyr':
                param_grid[
                    f'{options.selector}__threshold'
                ] = UniformDistribution(1e-3, 1e-1)
                param_grid[
                    f'{options.selector}__estimator__tol'
                ] = UniformDistribution(1e-4, 1e-2)
                param_grid[
                    f'{options.selector}__estimator__scale_l2_by'
                ] = CategoricalDistribution(('group_length', None))

                return SelectFromModel(SGLCV(
                    groups=make_groups_groupyr(features),
                    l1_ratio=stats.uniform(0, 1).rvs(max(2, options.n_rounds // 3)),
                    eps=1e-3,
                    n_alphas=max(2, options.n_rounds // 3),
                    max_iter=200,
                    cv=inner_cv,
                    tuning_strategy='bayes',
                    scoring=get_metric(options.metric_val),

                    verbose=False,
                    n_jobs=options.n_jobs,
                    n_bayes_iter=20,
                    n_bayes_points=10,
                    random_state=random_state,
                    suppress_solver_warnings=True,
                ))

            elif options.selector == 'sgl':
                param_grid[
                    f'{options.selector}__group_reg'
                ] = UniformDistribution(5e-3, 5e-2)
                param_grid[
                    f'{options.selector}__l1_reg'
                ] = UniformDistribution(5e-3, 5e-2)
                param_grid[
                    f'{options.selector}__tol'
                ] = UniformDistribution(1e-3, 1e-1)
                param_grid[
                    f'{options.selector}__scale_reg'
                ] = CategoricalDistribution(
                    ('group_size', 'none', 'inverse_group_size'))
                param_grid[
                    f'{options.selector}__frobenius_lipschitz'
                ] = CategoricalDistribution((True, False))
                param_grid[
                    f'{options.selector}__subsampling_scheme'
                ] = UniformDistribution(0.64, 1.0)

                return GroupLasso(
                    groups=make_groups_group_lasso(features),
                    n_iter=200,

                    random_state=random_state,
                    supress_warning=True,
                )

        elif options.task == 'classification':

            if options.selector == 'enet':
                param_grid[
                    f'{options.selector}__threshold'
                ] = UniformDistribution(1e-4, 1e-2)
                param_grid[
                    f'{options.selector}__estimator__tol'
                ] = UniformDistribution(1e-4, 1e-1)
                param_grid[
                    f'{options.selector}__estimator__C'
                ] = UniformDistribution(1e-4, 1e+4)
                param_grid[
                    f'{options.selector}__estimator__l1_ratio'
                ] = UniformDistribution(0.001, 0.99)

                return SelectFromModel(LogisticRegression(
                    solver='saga',
                    penalty='elasticnet',
                    max_iter=500,

                    random_state=random_state,
                    n_jobs=1,
                ))

            elif options.selector == 'lightning':
                param_grid[
                    f'{options.selector}__threshold'
                ] = UniformDistribution(1e-4, 1e-3)
                param_grid[
                    f'{options.selector}__estimator__tol'
                ] = UniformDistribution(1e-3, 1e-1)
                param_grid[
                    f'{options.selector}__estimator__C'
                ] = UniformDistribution(1e-4, 1e+4)
                param_grid[
                    f'{options.selector}__estimator__alpha'
                ] = UniformDistribution(1e-4, 1e+4)
                param_grid[
                    f'{options.selector}__estimator__max_steps'
                ] = IntUniformDistribution(10, 30, 5)
                param_grid[
                    f'{options.selector}__estimator__beta'
                ] = UniformDistribution(0.2, 0.9)
                param_grid[
                    f'{options.selector}__estimator__sigma'
                ] = UniformDistribution(1e-3, 1e-2)

                return SelectFromModel(CDClassifier(
                    penalty='l1/l2',
                    loss='log',
                    max_iter=200,
                    termination='violation_sum',
                    shrinking=True,
                    debiasing=False,
                    permute=True,
                    selection='uniform',

                    verbose=False,
                    n_jobs=1,
                    random_state=0,
                ))

            elif options.selector == 'groupyr':
                param_grid[
                    f'{options.selector}__threshold'
                ] = UniformDistribution(1e-4, 1e-2)
                param_grid[
                    f'{options.selector}__estimator__tol'
                ] = UniformDistribution(1e-3, 1e-1)
                param_grid[
                    f'{options.selector}__estimator__scale_l2_by'
                ] = CategoricalDistribution(('group_length', None))

                return SelectFromModel(LogisticSGLCV(
                    groups=make_groups_groupyr(features),
                    l1_ratio=stats.uniform(0, 1).rvs(max(2, options.n_rounds // 3)),
                    eps=1e-3,
                    n_alphas=max(2, options.n_rounds // 3),
                    max_iter=200,
                    cv=inner_cv,
                    scoring=get_metric(options.metric_val),
                    tuning_strategy='bayes',
                    n_bayes_iter=15,

                    verbose=False,
                    n_jobs=options.n_jobs,
                    random_state=random_state,
                    suppress_solver_warnings=True,
                ))

            elif options.selector == 'sgl':
                param_grid[
                    f'{options.selector}__group_reg'
                ] = UniformDistribution(5e-5, 5e-3)
                param_grid[
                    f'{options.selector}__l1_reg'
                ] = UniformDistribution(5e-5, 5e-3)
                param_grid[
                    f'{options.selector}__tol'
                ] = UniformDistribution(1e-3, 1e-2)
                param_grid[
                    f'{options.selector}__scale_reg'
                ] = CategoricalDistribution(('group_size', 'none',
                                             'inverse_group_size'))

                return LogisticGroupLasso(
                    groups=make_groups_group_lasso(features),
                    n_iter=200,

                    random_state=random_state,
                    supress_warning=True,
                )

    def get_predictor():
        if options.task == 'regression':

            if options.predictor == 'huber_sgd':
                param_grid[
                    f'{options.predictor}__tol'
                ] = UniformDistribution(1e-4, 1e-2)
                param_grid[
                    f'{options.predictor}__alpha'
                ] = UniformDistribution(1e-7, 1e-3)
                param_grid[
                    f'{options.predictor}__epsilon'
                ] = UniformDistribution(0, y.std())

                return SGDRegressor(
                    loss='huber',
                    penalty='l2',
                    max_iter=1_000,
                    learning_rate='optimal',
                    average=True,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,

                    random_state=random_state,
                )

            elif options.predictor == 'huber_lbfgsb':
                param_grid[
                    f'{options.predictor}__epsilon'
                ] = UniformDistribution(1, 2)
                param_grid[
                    f'{options.predictor}__alpha'
                ] = UniformDistribution(1e-7, 1e-3)
                param_grid[
                    f'{options.predictor}__tol'
                ] = UniformDistribution(1e-4, 1e-1)

                return HuberRegressor(max_iter=1_000)

            elif options.predictor == 'ridge':
                param_grid[
                    f'{options.predictor}__alpha'
                ] = UniformDistribution(1e-7, 1e-3)

                return Ridge(solver='svd')

        elif options.task == 'classification':

            if options.predictor == 'maxent':
                param_grid[
                    f'{options.predictor}__tol'
                ] = UniformDistribution(1e-4, 1e-1)
                param_grid[
                    f'{options.predictor}__C'
                ] = UniformDistribution(1e-2, 1e+7)

                predictor = LogisticRegression(
                    solver='lbfgs',
                    penalty='l2',
                    max_iter=500,

                    random_state=random_state,
                    n_jobs=1,
                )

            elif options.predictor == 'ridge':
                predictor = RidgeClassifier(solver='cholesky')

                if options.calibrate:
                    predictor = CalibratedClassifierCV(
                        base_estimator=predictor,
                        method='sigmoid',
                        cv=inner_cv.n_splits,
                        ensemble=False,

                        n_jobs=options.n_jobs,
                    )

                    param_grid[
                        f'{options.predictor}__base_estimator__alpha'
                    ] = UniformDistribution(1e-7, 1e-3)
                else:
                    param_grid[
                        f'{options.predictor}__alpha'
                    ] = UniformDistribution(1e-5, 1e-1)

            return predictor

    pipe = Pipeline(
        steps=[
            ('scale', get_preprocessing()),
            (options.selector, get_feature_selection()),
            (options.predictor, get_predictor()),
        ],
        memory=None,
        verbose=False,
    )
    search = OptunaSearchCV(
        pipe,
        param_grid,
        scoring=get_metric(options.metric_val),
        refit=True,
        cv=inner_cv,
        max_iter=100,
        n_trials=options.n_rounds,
        random_state=random_state,
        error_score=np.nan,
        return_train_score=False,
        verbose=options.verbose,
    )

    return search


def evaluate():
    warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
    warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')

    if options.cv == 'scaffold':
        X_smiles = pd.read_csv(options.csv,
                               usecols=[options.smiles_field]).squeeze('columns')
        X_mol = check_compounds_valid(X_smiles)
        cv = outer_cv.split(X_mol)
    else:
        cv = outer_cv

    cv_results = cross_validate(model, X, y,
                                scoring=get_metric(options.metric_test),
                                cv=cv,
                                n_jobs=options.n_jobs,
                                return_train_score=True,
                                return_estimator=True,
                                error_score=np.nan,
                                verbose=options.verbose)

    data_name = pathlib.PurePath(options.csv).stem
    dump(cv_results, f'{data_name}-{options.selector}-cv.pkl')

    mean_train_score = cv_results['train_score'].mean()
    std_train_score = cv_results['train_score'].std()
    mean_test_score = cv_results['test_score'].mean()
    std_test_score = cv_results['test_score'].std()
    if options.task == 'regression':
        mean_train_score = -mean_train_score
        mean_test_score = -mean_test_score

    def get_sparsity(post_sparsity=True):
        n_nonzeros = []
        for pipe in cv_results['estimator']:
            best_pipe = pipe.best_estimator_
            if not post_sparsity:
                if options.selector == 'sgl':
                    best_selector = best_pipe.named_steps[options.selector]
                    n_nonzeros.append(best_selector.sparsity_mask_.sum())
                elif options.selector == 'groupyr':
                    best_meta_selector = best_pipe.named_steps[options.selector]
                    best_selector = best_meta_selector.estimator_
                    n_nonzeros.append(best_selector.sparsity_mask_.sum())
                elif options.selector == 'lightning':
                    best_meta_selector = best_pipe.named_steps[options.selector]
                    best_selector = best_meta_selector.estimator_
                    n_nonzeros.append(best_selector.n_nonzero())
            else:
                best_predictor = best_pipe.named_steps[options.predictor]
                n_nonzeros.append(best_predictor.n_features_in_)
        return (np.mean(n_nonzeros) / X.shape[1] * 100,
                np.std(n_nonzeros) / X.shape[1] * 100)

    mean_sparsity, std_sparsity = get_sparsity()

    print(f'Train score: {mean_train_score:.3f} (+/-{std_train_score:.3f}),\n'
          f'Test score: {mean_test_score:.3f} (+/-{std_test_score:.3f}),\n'
          f'Sparsity: {mean_sparsity:.1f}% (+/-{std_sparsity:.1f}%)',
          end='')

    if options.selector in {'sgl', 'groupyr', 'lightning'}:
        mean_sparsity, std_sparsity = get_sparsity(post_sparsity=False)
        print(f',\n'
              f'Pre-sparsity: {mean_sparsity:.1f}% (+/-{std_sparsity:.1f}%).')
    else:
        print('.')


if __name__ == '__main__':
    random_state = np.random.RandomState(246)
    options = process_options()
    X, y = load_data()
    outer_cv, inner_cv = create_cvs()
    model = create_model_pipe()
    evaluate()
