# Datasets documentation

This folder contains scripts and notebooks that have been used for creation of benchmarks in [dataset](../dataset/) folder. The format and the process of contribution a new benchmark is specified there.

To make the process of benchmarks creation reproducible, please, try to stick to the following principles:

* Fix the random seeds so your script produce the same dataset when calling repeatedly
* Clean up temporary files, especially if they were created inside the package structure (so they will not be accidentally pushed to GitHub) 
* It is ok to import packages that are not contained in `requirements.txt` but avoid adding unnecessary dependencies
* For Jupyter Notebook, it might be a good idea to rerun everything at the end using `Kernel -> Restart & Run All`


