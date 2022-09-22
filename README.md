# Efficient identification of informative features in simulation-based inference
This repository contains the code for reproducing results of the FSML paper [openreview.net/forum?id=AYQI3rlp9tW](https://openreview.net/forum?id=AYQI3rlp9tW). An updated and refactored version of the code can be found [here](https://github.com/berenslab/fslm).

## Installation
The necessary packages can be installed via:
`pip install .`

## Usage
The repository is structured as follows:
```
|-README.md
|-setup.py
|-notebooks
|-results
|-experiments
|  |-templates
|	    |-simulate.py
|	    |-full_experiment.py
|	    |-experiment_presimulated.py
|	|-run_experiments.sh
|	|-experiment2.py
|	|-experiment3.py
|	|-experiment4.py
|	|-(...)
|-sbi_feature_importance
	|-__init__.py
	|-snle.py
	|-utils.py
	|-metrics.py
	|-experiment_helper.py
	|-analysis.py
	|-toymodels.py
	|-snpe.py
|-ephys_helper
	|-__init__.py
	|-utils.py
	|-hh_simulator.py
	|-extractor.py
	|-features.py
	|-analysis.py
```

You can find the results to our experiments [here](https://zenodo.org/record/7104245) and reproduce the figures with the provided notebooks.

The templates provide an orientation for setting up FSLME experiments with HH models.

All experiments can be run via the provided bash_script (Only feasable with larger amounts of compute)
`bash run_experiments.sh`
