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

TBD

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
