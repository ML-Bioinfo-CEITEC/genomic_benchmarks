# Experiments

In this folder, we collect experimental runs of models included in `genomic_benchmarks.models` module for benchmark datasets.  

Each run is recorded as a notebook. Currently, TensorFlow experiments are run with [papermill](https://github.com/nteract/papermill), PyTorch experiments are run manually.

See accuracy of the prediction over test sets in the table below.

## How to add an experiment

### PyTorch

See [README.md](torch_cnn_experiments/README.md).

### TensorFlow

```bash
papermill tf_cnn_classifier.ipynb tf_cnn_experiments/[DATASET NAME].ipynb -p DATASET [DATASET NAME]
```

## Results

### PyTorch CNN

|Dataset name|version|Model name|Commit hash|Test acc|
|---|---|---|---|---|
|demo_coding_vs_intergenomic_seqs|0|SimpleCNN|7cf2bf2faa97e152032ddd01f11f5273b70ad69b|87.3|
|human_nontata_promoters|0|SimpleCNN|7cf2bf2faa97e152032ddd01f11f5273b70ad69b|83.7|
|demo_mouse_enhancers|0|SimpleCNN|7cf2bf2faa97e152032ddd01f11f5273b70ad69b|74.4|

### TensorFlow CNN

| Dataset                          |   Loss |   Accuracy |
|:---------------------------------|-------:|-----------:|
| demo_coding_vs_intergenomic_seqs |  0.310 |       87.1 |
| demo_human_or_worm               |  0.155 |       94.0 |
| demo_mouse_enhancers             |  0.655 |       57.0 |
| human_enhancers_cohn             |  0.581 |       70.8 |
| human_enhancers_ensembl          |  0.423 |       81.1 |
| human_nontata_promoters          |  0.349 |       85.2 |