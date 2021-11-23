#!/bin/bash

for f in "demo_coding_vs_intergenomic_seqs" "demo_human_or_worm" "demo_mouse_enhancers" "human_enhancers_cohn" "human_enhancers_ensembl" "human_nontata_promoters"
do
  papermill utils/torch_cnn_classifier.ipynb torch_cnn_experiments/$f.ipynb -p DATASET $f
  rc=$?
  if [[ 0 != $rc ]]; then echo Failed command: ${f}; break; fi
done