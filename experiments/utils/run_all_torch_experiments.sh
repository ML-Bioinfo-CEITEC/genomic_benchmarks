#!/bin/sh
DATASETS=`ls -1d ../datasets/*/`


for i in 1 2 3 4 5; do
  echo $i
  for d in $DATASETS 
  do
    dt=`basename $d`
    echo "Processing dataset $dt\n\n"
    papermill utils/torch_cnn_classifier.ipynb torch_cnn_experiments/$dt.ipynb -p DATASET $dt -p EPOCHS 200 -p ITER $i -p PATIENCE 30
  done
done


# END=5
# for ((i=1;i<=END;i++)); do
#   echo $i
# done

# for d in $DATASETS 
# do
#   dt=`basename $d`
#   echo "Processing dataset $dt\n\n"
#   papermill utils/torch_cnn_classifier.ipynb torch_cnn_experiments/$dt.ipynb -p DATASET $dt -p EPOCHS 10
# done


