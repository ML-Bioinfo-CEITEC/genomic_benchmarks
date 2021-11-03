# Experiments

In this folder, we collect experimental runs of models included in `genomic_benchmarks.models` module for benchmark datasets.  

Each run is recorded as a notebook. Currently, TensorFlow experiments are run with [papermill](https://github.com/nteract/papermill), PyTorch experiments are run manually.

See accuracy of the prediction over test sets in the table below.

## How to add an experiment

### PyTorch

See [README.md](torch_cnn_experiments/README.md).

### TensorFlow

```bash
papermill utils/tf_cnn_classifier.ipynb tf_cnn_experiments/[DATASET NAME].ipynb -p DATASET [DATASET NAME]
```

To add another TF papermill script:

  * create a notebook in this folder
  * all parameters should be specified in one cell in the beginning of the notebook, tag it `parameters` as described [here](https://github.com/nteract/papermill#parameterizing-a-notebook)
  * run experiments using papermill for all benchmark datasets

## Results

### PyTorch CNN

| Dataset                          |   Accuracy |
|:---------------------------------|-----------:|
| demo_coding_vs_intergenomic_seqs |       87.3 |
| demo_human_or_worm               |       92.6 |
| demo_mouse_enhancers             |       77.3 |
| human_enhancers_cohn             |       67.1 |
| human_enhancers_ensembl          |       80.4 |
| human_nontata_promoters          |       81.9 |


### TensorFlow CNN

| Dataset                          |   Loss |   Accuracy |
|:---------------------------------|-------:|-----------:|
| demo_coding_vs_intergenomic_seqs |  0.310 |       87.1 |
| demo_human_or_worm               |  0.155 |       94.0 |
| demo_mouse_enhancers             |  0.655 |       57.0 |
| human_enhancers_cohn             |  0.581 |       70.8 |
| human_enhancers_ensembl          |  0.423 |       81.1 |
| human_nontata_promoters          |  0.349 |       85.2 |