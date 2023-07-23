"""Setup for the m6anet package."""
import os
import re
from setuptools import setup,find_packages

__pkg_name__ = 'm6anet'
verstrline = open(os.path.join(__pkg_name__, '__init__.py'), 'r', encoding='utf-8').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "{}/__init__.py".'.format(__pkg_name__))

with open('README.md', encoding='utf-8') as f:
    README = f.read()

setup(
    author="Christopher Hendra",
    maintainer_email="christopher.hendra@u.nus.edu",
    name=__pkg_name__,
    license="MIT",
    description='m6anet is a python package for detection of m6a modifications from Nanopore direct RNA sequencing data.',
    version=__version__,
    long_description_content_type="text/markdown",
    long_description=README,
    url='https://github.com/GoekeLab/m6anet',
    packages=find_packages(),
    package_data={'m6anet.model': ['model_states/rna002_hct116.pt', 'model_states/rna002_arabidopsis_virc.pt', 'model_states/rna004_hek293t.pt', 'configs/model_configs/m6anet.toml',
                                   'norm_factors/rna002_hct116.joblib', 'norm_factors/rna002_arabidopsis_virc.joblib']},
    python_requires=">=3.7, <3.9",
    install_requires=[
            "numpy>=1.18.0",
            "pandas>=0.25.3",
            "scikit-learn>=0.24.0, <1.1.0; python_version=='3.7'",
            "scikit-learn>=0.24.0; python_version=='3.8'",
            "scipy>=1.4.1, <1.8.0; python_version=='3.7'",
            "scipy>=1.4.1; python_version=='3.8'",
            "ujson",
            "torch==1.6.0",
            "toml>=0.10.2",
            "tqdm",
            "typing-extensions"
            ],
    entry_points = {
        'console_scripts': [
            '{0} = {0}:main'.format(__pkg_name__),
            "m6anet-dataprep={}.deprecated.dataprep:main".format(__pkg_name__),
            "m6anet-run_inference={}.deprecated.inference:main".format(__pkg_name__),
            "m6anet-compute_norm_factors={}.deprecated.compute_norm_factors:main".format(__pkg_name__),
            "m6anet-train={}.deprecated.train:main".format(__pkg_name__)]
        },
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Science/Research',
    ],
)
