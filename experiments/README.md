# Experiments

In this folder, each notebook is one experient = one benchmark and the appropriate CNN model fitted to it. See accuracy of the prediction over test sets in the table below.

## To add a new experiment
- Choose a dataset from `genomic_benchmarks/src/genomic_benchmarks/dataset_getters/pytorch_datasets.py`
- Copy an existing experiment notebook
- Replace the notebook's dataset with yours
- Augment config
- Run

## Results
|Dataset name|version|Model name|Commit hash|Test acc|
|---|---|---|---|---|
|demo_coding_vs_intergenomic_seqs|0|SimpleCNN|7cf2bf2faa97e152032ddd01f11f5273b70ad69b|87.3|
|human_nontata_promoters|0|SimpleCNN|7cf2bf2faa97e152032ddd01f11f5273b70ad69b|83.7|
|demo_mouse_enhancers|0|SimpleCNN|7cf2bf2faa97e152032ddd01f11f5273b70ad69b|74.4|