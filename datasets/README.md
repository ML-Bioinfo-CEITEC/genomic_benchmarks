# Datasets

Each folder contains either one benchmark or a set of benchmarks. See [docs/](../docs/) for code used to create these benchmarks.

### Naming conventions

* *dummy_...*: small datasets, used for testing purposes
* *demo_...*: middle size datasets, not necesarily biologically relevant or fully reproducible, used in demos

### Versioning

We recommend to check the version number when working with the dataset (i.e. not using default `None`). The version should be set to 0 when the dataset is proposed, after inicial curration it should be changed to 1 and then increased after every modification.

### Data format

t.b.d.

### To add a new datasets

Create a new branch. Add the new subfolders to `datasets` and `docs`. The subfolder of `docs` should contain a description of the dataset in `README.md`. If the dataset comes with the paper, link the paper. If the dataset is not taken from the paper, make sure you have described and understand the biological process behind it.

### To review a new dataset

Make sure you can run and reproduce the code. Check you can download the actual sequences and/or create a data loader. Do you understand what is behind these data? (either from the paper or the description) Aks for clarification when needed.