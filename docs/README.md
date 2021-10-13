# Datasets documentation

This folder contains scripts and notebooks that have been used for creation of benchmarks in [dataset](../dataset/) folder. The format and the process of contribution a new benchmark is specified there.

To make the process of benchmarks creation reproducible, please, try to stick to the following principles:

* fix the random seeds so your script produce the same dataset when calling repeatedly
* clean up temporary files, especially if they were created inside the package structure (so they will not be accidentally pushed to GitHub) 
* it is ok to import packages that are not contained in `requirements.txt` but avoid adding unnecessary dependencies



