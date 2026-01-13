# Pearl

This repository contains the code for Pearl.

## Installation


Create a Python 3.10 virtual environment using pyenv and install dependencies:

```bash
pyenv virtualenv 3.10.0 pearl
pyenv local pearl # cuda=11.8
```

## Data Layout

Before training or evaluation, make sure the datasets are already placed under `data/` with the following structure:

```
data/
├── images/
├── spica_test.csv
├── spica_train.csv
└── spica_val.csv
```

## Feature Extraction

Next, generate the features by following the instructions described in `save_features/README.md`. The commands there will populate the required feature files under `save_features/`.

## setup poetry
```
poetry sync
```

## Training

```bash
sh train.sh
```

## Evaluation

```bash
sh validate.sh
```
