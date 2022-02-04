# How to reproduce
...
This document should guide you to install and set up the development environment on a new machine. 
The experiments were run on a Debian 10 os.
## How to set up environment
Install Python 3.8.9, create a new virtual environment and activate it. Clone this repository and do editable `pip` installation. Finally, install PyTorch and Comet.
```bash
   cd cnn_experiments
   pip install --editable .

   pip install jupyter>=1.0.0
   pip install papermill>=2.3.0

   pip install torch>=1.10.0
   pip install torchtext
   
   pip install comet-ml
```
Side note: Our package _genomic_benchmarks_ is also on pip but for this publication we use an augmented version supplied with the code. Therefore the installation through ` pip install --editable .` and **not** through downloading from the pip.

## How to run
To run all the experiments:
```
   cd cnn_experiments/experiments
   TODO RUN utils/run_all_torch_experiments.sh
   
```

To run one experiment:
```
   cd cnn_experiments/experiments
   papermill utils/torch_cnn_classifier.ipynb torch_cnn_experiments/[DATASET NAME].ipynb -p DATASET [DATASET NAME] -p EPOCHS 200 -p ITER [ITERATION NUMBER] -p PATIENCE 30
   
```
Where [DATASET NAME] should be replaced by a desired dataset name string and [ITERATION NUMBER] is an arbitrary integer identifying/labeling/naming the experiment run.

Dataset selection: "human_enhancers_cohn", "human_enhancers_ensembl", "human_nontata_promoters", "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm"

The results will be in the folder `cnn_experiments/torch_cnn_experiments/name-of-dataset`.

## Folder structure
The folder datasets contains the benchmark datasets.
The folder docs contains scripts and notebooks that have been used for creation of benchmarks.
The folder src/genomic_benchmarks contains our package handling the genomic datasets.
The folder experiments/tests contains 
The folder experiments/utils
The folder experiments/utils



