# Genomic Benchmarks

## Introduction

[WHY ARE BENCHMARKS IMPORTANT?]

[WHAT BENCHMARKS ARE GENOMIC BENCHMARKS?]

[HOW TO USE GENOMIC BENCHMARKS?]

## Install

Genomic benchmarks can be used directly or install as a Python package as follows:

```
   git clone git@github.com:ML-Bioinfo-CEITEC/genomic_benchmarks.git
   python setup.py develop
```

or with conda

conda install [TBD]

## Structure of package

  * [datasets](datasets/): Each folder is one bechmark dataset (or a set of bechmarks in subfolders), see [README.md](datasets/README.md) for the format specification
  * [docs](docs/): Each folder contains a Python notebook that has been used for the dataset construction
  * [src/genomic_benchmarks](src/genomic_benchmarks/): Python module for datasets manipulation (downlading, checking, etc.) 


## Tests

TBD

## How to contribute

TBD
