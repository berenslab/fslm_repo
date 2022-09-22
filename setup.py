from setuptools import setup, find_packages

#|-sbi_feature_importance
#	|-__init__.py
#	|-snle.py
#	|-utils.py
#	|-metrics.py
#	|-experiment_helper.py
#	|-analysis.py
#	|-toymodels.py
#	|-snpe.py
#|-ephys_helper
#	|-__init__.py
#	|-utils.py
#	|-hh_simulator.py
#	|-extractor.py
#	|-features.py
#	|-analysis.py


setup(
   name='sbi_feature_importance',
   version='0.1',
   description='Tools to evaluate feature importance in sbi, specifically for HH models.',
   author='anonymous',
   author_email='MNWNCQVOTS@gmail.com',
   packages=find_packages(),
   install_requires=['sbi', 'seaborn', 'matplotlib', 'torch', 'numpy', 'scipy', 'joblib', 'tqdm', 'brian2', 'pandas', 'six'],
)

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", 'protobuf==3.20.1'])