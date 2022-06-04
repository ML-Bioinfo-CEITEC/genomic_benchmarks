#!/bin/sh

DATASET="human_ensembl_regulatory"

dt=`basename $DATASET`
echo "Processing dataset $dt\n\n"
papermill utils/torch_cnn_classifier.ipynb torch_cnn_experiments/$dt.ipynb -p DATASET $dt -p EPOCHS 10
