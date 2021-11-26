# Experiments

In this folder, we collect experimental runs of models included in `genomic_benchmarks.models` module for benchmark datasets.  

Each run is recorded as a notebook. Currently, TensorFlow experiments are run with [papermill](https://github.com/nteract/papermill), PyTorch experiments are run manually.

See the accuracy and F1 score on test sets in the table below.

## How to add an experiment

### PyTorch

See [README.md](torch_cnn_experiments/README.md).

### TensorFlow

To run one experiment:

```bash
papermill utils/tf_cnn_classifier.ipynb tf_cnn_experiments/[DATASET NAME].ipynb -p DATASET [DATASET NAME]
```

To run all experiments:

```bash
bash utils/run_all_tf_experiments.sh
```

To add another TF papermill script:

  * create a notebook in [utils](utils/) folder
  * all parameters should be specified in one cell in the beginning of the notebook, tag it `parameters` as described [here](https://github.com/nteract/papermill#parameterizing-a-notebook)
  * run experiments using papermill for all benchmark datasets

## Results

### PyTorch CNN

| Dataset                          |   Accuracy |   F1 score |
|:---------------------------------|-----------:|-----------:|
| demo_coding_vs_intergenomic_seqs |       86.6 |       85.5 |
| demo_human_or_worm               |       92.7 |       92.5 |
| demo_mouse_enhancers             |       76.4 |       71.6 |
| human_enhancers_cohn             |       69.5 |       65.4 |
| human_enhancers_ensembl          |       81.5 |       80.5 |
| human_nontata_promoters          |       83   |       81.3 |


### TensorFlow CNN

| Dataset                          |   Loss |   Accuracy |   F1 |
|:---------------------------------|-------:|-----------:|-----:|
| demo_coding_vs_intergenomic_seqs |  0.264 |       89.5 | 89.3 |
| demo_human_or_worm               |  0.168 |       93.5 | 91.7 |
| demo_mouse_enhancers             |  0.614 |       73.6 | 21.7 |
| human_enhancers_cohn             |  0.763 |       66.6 | 74.3 |
| human_enhancers_ensembl          |  0.428 |       80.9 | 80.2 |
| human_nontata_promoters          |  0.513 |       79.5 | 73.3 |