## To add a new experiment
- Run the papermill command with `genomic_benchmarks/experiments/utils/torch_cnn_classifier.ipynb` notebook and the name of the choosen dataset
`papermill utils/torch_cnn_classifier.ipynb torch_cnn_experiments/[DATASET NAME].ipynb -p DATASET [DATASET NAME]`