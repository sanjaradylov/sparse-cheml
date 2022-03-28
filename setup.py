#! /usr/bin/env python

"""Install ``sparse_cheml`` package."""

from setuptools import find_packages, setup

from sparse_cheml import __version__


setup(
    name='sparse_cheml',
    version=__version__,
    description='Sparse machine learning for molecular-property prediction',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.9',
        'Intended Audience :: Developers :: Data Scientists',
        'Topic :: Utilities :: Machine Learning :: Cheminformatics',
    ],
    author='Sanjar Ad[yi]lov',
    packages=find_packages(exclude=('tests', 'examples', 'data')),
)
