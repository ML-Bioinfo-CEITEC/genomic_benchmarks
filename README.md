# Genomic Benchmarks 🧬🏋️✔️

In this repository, we collect benchmarks for classification of genomic sequences. It is shipped as a Python package, together with functions helping to download & manipulate datasets and train NN models. 

## Hackathon 2021-11-19

We have collected a list of genomic datasets and are now organizing the ML hackathon to train classifiers over them. Would you join us on Friday, [November 19, 2021, 15:00 CET](https://www.timeanddate.com/worldclock/converter.html?iso=20211119T140000&p1=204&p2=136&p3=179&p4=224&p5=33&p6=176&p7=248) at [CEITEC MU](https://www.ceitec.cz/), Brno, Czechia 🇨🇿🇪🇺, or remotely? Free refreshment for all participants, swag for the winners. The event is both competitive (to prove your ML models are the best) and a learning opportunity (we will provide all the help we can). 

* Final datasets and evaluation metrics will be provided on the day of the hackathon. In principle, they will be similar to datasets currently included in this package.
* You can participate both in person at CEITEC or remotely. More information at [bit.ly/genomichackathon](https://bit.ly/genomichackathon), sign up [here](https://forms.gle/s7zoqpzXmjU6yATm6). No prior knowledge about DNA/RNA/genetics needed (you must be able to code in Python and know ML basics).
* To participate on-site, you must be vaccinated, recovered or tested (O-N-T regulations analogical to German G3 apply). Please, bring FFP2 mask.

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
['demo_coding_vs_intergenomic_seqs', 'demo_mouse_enhancers', 'human_nontata_promoters', 'human_enhancers_cohn', 'demo_human_or_worm', 'human_enhancers_ensembl']
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

TBD
