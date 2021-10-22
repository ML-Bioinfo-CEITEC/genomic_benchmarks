# Genomic Benchmarks

## Introduction

[WHY ARE BENCHMARKS IMPORTANT?]

[WHAT BENCHMARKS ARE GENOMIC BENCHMARKS?]

## Install

Genomic benchmarks can be installed as a Python package as follows:

```
   git clone https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks.git
   cd genomic_benchmarks
   python setup.py develop
```

## Usage

The function `download_dataset` downloads the full-sequences form of the required format, splitted into train and test sets, one folder for each class. If not specified otherwise, the data will be stored in `.genomic_benchmarks` subfolder of your home directory. By default, the dataset is downloaded from our cloud cache but you may choose to download all the references and re-create the benchmark (`use_cloud_cache=False`).  

```python
  from genomic_benchmarks.loc2seq import download_dataset
  
  download_dataset("human_nontata_promoters")
```

You can get basic information about the benchmark with `info` function (you can get more info in [datasets](datasets/) & [docs](docs/) folders):

```python
  from genomic_benchmarks.data_check import info
  
  info("human_nontata_promoters")
```

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
