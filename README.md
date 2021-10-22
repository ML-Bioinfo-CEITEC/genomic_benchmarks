# Genomic Benchmarks

In this repository, we collect benchmarks for classification of genomic sequences. It is shipped as a Python package, together with functions helping to download & manipulate datasets and train NN models. 
## Install

Genomic benchmarks can be installed as follows:

```
   git clone https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks.git
   cd genomic_benchmarks
   python setup.py develop
```
## Usage

The function `download_dataset` downloads the full-sequence form of the required benchmark (splitted into train and test sets, one folder for each class). If not specified otherwise, the data will be stored in `.genomic_benchmarks` subfolder of your home directory. By default, the dataset is obtained from our cloud cache (`use_cloud_cache=True`).  

```python
  from genomic_benchmarks.loc2seq import download_dataset
  
  download_dataset("human_nontata_promoters")
```

You can get basic information about the benchmark with `info` function:

```python
  from genomic_benchmarks.data_check import info
  
  info("human_nontata_promoters")
```

Getting TenforFlow Dataset for the benchmark is straightforward: 

```python
  from pathlib import Path
  import tensorflow as tf

  BATCH_SIZE = 64
  SEQ_PATH = Path.home() / '.genomic_benchmarks' / 'human_nontata_promoters' / 'train'
  CLASSES = ['negative', 'positive']

  train_dset = tf.keras.preprocessing.text_dataset_from_directory(
      directory=SEQ_TRAIN_PATH,
      batch_size=BATCH_SIZE,
      class_names=CLASSES)
```
See [How_To_Train_CNN_Classifier_With_TF.ipynb](notebooks/How_To_Train_CNN_Classifier_With_TF.ipynb) for a description how to train CNN classifier with TenforFlow.

## Introduction

[WHY ARE BENCHMARKS IMPORTANT?]

[WHAT BENCHMARKS ARE GENOMIC BENCHMARKS?]
## Structure of package

  * [datasets](datasets/): Each folder is one benchmark dataset (or a set of bechmarks in subfolders), see [README.md](datasets/README.md) for the format specification
  * [docs](docs/): Each folder contains a Python notebook that has been used for the dataset creation
  * [experiments](experiments/): Fitting a simple neural network model(s) for each benchmark, can be used as a baseline
  * [notebooks](notebooks/): Main use-cases demonstrated in a form of Jupyter notebooks 
  * [src/genomic_benchmarks](src/genomic_benchmarks/): Python module for datasets manipulation (downlading, checking, etc.) 

## Tests

TBD

## How to contribute

TBD
