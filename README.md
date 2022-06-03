[![PyPI version](https://badge.fury.io/py/genomic-benchmarks.svg)](https://badge.fury.io/py/genomic-benchmarks) 

# Genomic Benchmarks 🧬🏋️✔️

In this repository, we collect benchmarks for classification of genomic sequences. It is shipped as a Python package, together with functions helping to download & manipulate datasets and train NN models. 
## Install

Genomic Benchmarks can be installed as follows:

```bash
pip install genomic-benchmarks
```

To use it with papermill, TF or pytorch, install the corresponding dependencies:

```bash
# if you want to use jupyter and papermill
pip install jupyter>=1.0.0
pip install papermill>=2.3.0

# if you want to train NN with TF
pip install tensorflow>=2.6.0
pip install tensorflow-addons
pip install typing-extensions --upgrade  # fixing TF installation issue

# if you want to train NN with torch
pip install torch>=1.10.0
pip install torchtext

```

For the package development, use Python 3.8 (ideally 3.8.9) and the installation described [here](README_devel.md).

## Usage
Get the list of all datasets with the `list_datasets` function

```python
>>> from genomic_benchmarks.data_check import list_datasets
>>> 
>>> list_datasets()
['demo_coding_vs_intergenomic_seqs', 'demo_human_or_worm', 'dummy_mouse_enhancers_ensembl', 'human_enhancers_cohn', 'human_enhancers_ensembl', 'human_ensembl_regulatory',  'human_nontata_promoters', 'human_ocr_ensembl']
```

You can get basic information about the benchmark with `info` function:

```python
>>> from genomic_benchmarks.data_check import info
>>> 
>>> info("human_nontata_promoters", version=0)
Dataset `human_nontata_promoters` has 2 classes: negative, positive.

All lenghts of genomic intervals equals 251.

Totally 36131 sequences have been found, 27097 for training and 9034 for testing.
          train  test
negative  12355  4119
positive  14742  4915
```

The function `download_dataset` downloads the full-sequence form of the required benchmark (splitted into train and test sets, one folder for each class). If not specified otherwise, the data will be stored in `.genomic_benchmarks` subfolder of your home directory. By default, the dataset is obtained from our cloud cache (`use_cloud_cache=True`). 

```python
>>> from genomic_benchmarks.loc2seq import download_dataset
>>> 
>>> download_dataset("human_nontata_promoters", version=0)
Downloading 1VdUg0Zu8yfLS6QesBXwGz1PIQrTW3Ze4 into /home/petr/.genomic_benchmarks/human_nontata_promoters.zip... Done.
Unzipping...Done.
PosixPath('/home/petr/.genomic_benchmarks/human_nontata_promoters')
```

Getting TensorFlow Dataset for the benchmark and displaying samples is straightforward: 

```python
>>> from pathlib import Path
>>> import tensorflow as tf
>>> 
>>> BATCH_SIZE = 64
>>> SEQ_TRAIN_PATH = Path.home() / '.genomic_benchmarks' / 'human_nontata_promoters' / 'train'
>>> CLASSES = ['negative', 'positive']
>>> 
>>> train_dset = tf.keras.preprocessing.text_dataset_from_directory(
...     directory=SEQ_TRAIN_PATH,
...     batch_size=BATCH_SIZE,
...     class_names=CLASSES)
Found 27097 files belonging to 2 classes.
>>> 
>>> list(train_dset)[0][0][0]
<tf.Tensor: shape=(), dtype=string, numpy=b'TCCTGCCTTTCCACTTGCACCAGTTTTCCCACCCCAGCCTCAGGGCGGGGCTGCCTCGTCACTTGTCTCGGGGCAGATCTGCCCTACACACGTTAGCGCCGCGCGCAAAGCAGCCCCGCAGCACCCAGGCGCCTCCTGGCGGCGCCGCGAAGGGGCGGGGCTGTCGGCTGCGCGTTGTGCGCTGTCCCAGGTTGGAAACCAGTGCCCCAGGCGGCGAGGAGAGCGGTGCCTTGCAGGGATGCTGCGGGCGG'>
```
See [How_To_Train_CNN_Classifier_With_TF.ipynb](notebooks/How_To_Train_CNN_Classifier_With_TF.ipynb) for more detailed description how to train CNN classifier with TensorFlow.

Getting Pytorch Dataset and displaying samples is also easy:
```python
>>> from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanNontataPromoters
>>> 
>>> dset = HumanNontataPromoters(split='train', version=0)
>>> dset[0]
('CAATCTCACAGGCTCCTGGTTGTCTACCCATGGACCCAGAGGTTCTTTGACAGCTTTGGCAACCTGTCCTCTGCCTCTGCCATCATGGGCAACCCCAAAGTCAAGGCACATGGCAAGAAGGTGCTGACTTCCTTGGGAGATGCCATAAAGCACCTGGATGATCTCAAGGGCACCTTTGCCCAGCTGAGTGAACTGCACTGTGACAAGCTGCATGTGGATCCTGAGAACTTCAAGGTGAGTCCAGGAGATGT', 0)
```
See [How_To_Train_CNN_Classifier_With_Pytorch.ipynb](notebooks/How_To_Train_CNN_Classifier_With_Pytorch.ipynb) for more detailed description how to train CNN classifier with Pytorch.


## Introduction

[WHY ARE BENCHMARKS IMPORTANT?]

[WHAT BENCHMARKS ARE GENOMIC BENCHMARKS?]
## Structure of package

  * [datasets](datasets/): Each folder is one benchmark dataset (or a set of bechmarks in subfolders), see [README.md](datasets/README.md) for the format specification
  * [docs](docs/): Each folder contains a Python notebook that has been used for the dataset creation
  * [experiments](experiments/): Training a simple neural network model(s) for each benchmark dataset, can be used as a baseline
  * [notebooks](notebooks/): Main use-cases demonstrated in a form of Jupyter notebooks 
  * [src/genomic_benchmarks](src/genomic_benchmarks/): Python module for datasets manipulation (downlading, checking, etc.)
  * [tests](tests/): Unit tests for `pytest` and `pytest-cov`

## How to contribute

### How to contribute a model

If you beat our current best model on any dataset or just came with an interesting new idea, let us know about it: Make you code publicly available (GitHub repo, Colab...) and fill in the form at

https://forms.gle/pvkkrgHNCNmAAC1TA

### How to contribute a dataset

If you have an interesting genomic dataset, send us [an issue](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/issues) with the description and possibly link to the data (e.g. BED file and FASTQ reference). In the future, we will provide functions to make the import easy. 

If you are a hero, read [the specification of our dataset format](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/tree/main/datasets) and send us a pull request with new `datasets/[YOUR_DATASET_NAME]` and `docs/[YOUR_DATASET_NAME]` folders.



### How to improve code in this package

We welcome new code contributors. If you see a bug, send us [an issue](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/issues) with a [minimal reproducible example](https://stackoverflow.com/help/minimal-reproducible-example). Or even better, fix the bug and send us a pull request. 
