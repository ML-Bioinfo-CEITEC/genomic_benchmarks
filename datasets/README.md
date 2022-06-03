# Datasets

Each folder contains either one benchmark or a set of benchmarks. See [docs/](../docs/) for code used to create these benchmarks.

### Naming conventions

* *dummy_...*: small datasets, used for testing purposes
* *demo_...*: middle size datasets, not necesarily biologically relevant or fully reproducible, used in demos

### Versioning

We recommend to check the version number when working with the dataset (i.e. not using default `None`). The version should be set to 0 when the dataset is proposed, after inicial curration it should be changed to 1 and then increased after every modification.

### Data format

Each benchmark should contain `metadata.yaml` file with its main folder with the specification in YAML format, namely

   * **the version** of the benchmark (0 = in development)

   * **the classes** of genomic sequences, for each class we further need to specify

       - *url* with the reference 
       - *type* of the reference (currently, only fa.gz implemented)
       - *extra_processing*, a parameter helping to overcome some know issues with identifiers matching

The main folder should also contain two folders, `train` and `test`. Both those folders should contain gzipped CSV files, one for each class (named `class_name.csv.gz`).

The format of gzipped CSV files closely resemble BED format, the column names must be the following:

* **id**: id of a sequence
* **region**: chromosome/transcript/... to be matched with the reference
* **start**, **end**: genomic interval specification (0-based, i.e. same as in Python)
* **strand**: either '+' or '-'


### To contribute a new datasets

Create a new branch. Add the new subfolders to `datasets` and `docs`. The subfolder of `docs` should contain a description of the dataset in `README.md`. If the dataset comes with the paper, link the paper. If the dataset is not taken from the paper, make sure you have described and understand the biological process behind it.

If you have access to `cloud_cache` folder on GDrive, upload your file there and update `CLOUD_CACHE` in [cloud_caching.py](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/blob/main/src/genomic_benchmarks/loc2seq/cloud_caching.py).

### To review a new dataset

Make sure you can run and reproduce the code. Check you can download the actual sequences and/or create a data loader. Do you understand what is behind these data? (either from the paper or the description) Ask for clarification if needed.
