# Experiments

In this folder, we collect experimental runs of models included in `genomic_benchmarks.models` module for benchmark datasets.  

Each run is recorded as a notebook. Currently, TensorFlow experiments are run with [papermill](https://github.com/nteract/papermill), PyTorch experiments are run manually.

See the accuracy and F1 score on test sets in the table below.

## How to add an experiment

### PyTorch

To run one experiment:

```bash
papermill utils/torch_cnn_classifier.ipynb torch_cnn_experiments/[DATASET NAME].ipynb -p DATASET [DATASET NAME]
```

To run all experiments:

```bash
bash utils/run_all_torch_experiments.sh
```

### TensorFlow

To run one experiment:

```bash
papermill utils/tf_cnn_classifier.ipynb tf_cnn_experiments/[DATASET NAME].ipynb -p DATASET [DATASET NAME]
```

To run all experiments:

```bash
bash utils/run_all_tf_experiments.sh
```

## How to add another papermill script:

  * create a notebook in [utils](utils/) folder following the naming convention there
  * all parameters should be specified in one cell in the beginning of the notebook, tag it `parameters` as described [here](https://github.com/nteract/papermill#parameterizing-a-notebook)
  * run experiments using papermill for all benchmark datasets

## Results

### PyTorch CNN

| Dataset                          |   Accuracy |   F1 score |
|:---------------------------------|-----------:|-----------:|
| demo_coding_vs_intergenomic_seqs |       87.6 |       86.8 |
| demo_human_or_worm               |       93   |       92.8 |
| drosophila_enhancers_stark       |       58.6 |       44.5 |
| dummy_mouse_enhancers_ensembl    |       69   |       70.4 |
| human_enhancers_cohn             |       69.5 |       67.1 |
| human_enhancers_ensembl          |       68.9 |       56.5 |
| human_ensembl_regulatory         |       93.3 |       93.3 |
| human_nontata_promoters          |       84.6 |       83.7 |
| human_ocr_ensembl                |       68   |       66.1 |


### TensorFlow CNN

| Dataset                          |   Loss |   Accuracy |   F1 |
|:---------------------------------|-------:|-----------:|-----:|
| demo_coding_vs_intergenomic_seqs |  0.258 |       89.6 | 89.4 |
| demo_human_or_worm               |  0.148 |       94.2 | 93.2 |
| drosophila_enhancers_stark       |  0.959 |       52.4 | 69.1 |
| dummy_mouse_enhancers_ensembl    |  0.919 |       50   | 66.9 |
| human_enhancers_cohn             |  0.589 |       68.9 | 71.3 |
| human_enhancers_ensembl          |  0.421 |       81.1 | 74.6 |
| human_ensembl_regulatory         |  0.505 |       79.3 | 79.3 |
| human_nontata_promoters          |  0.319 |       86.5 | 84.4 |
| human_ocr_ensembl                |  0.585 |       68.8 | 72   |
