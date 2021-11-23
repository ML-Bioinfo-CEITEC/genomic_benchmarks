DATASETS=`ls -1d ../datasets/*/`

for d in $DATASETS 
do
  dt=`basename $d`
  echo "Processing dataset papermill utils/torch_cnn_classifier.ipynb torch_cnn_experiments/[DATASET NAME].ipynb -p DATASET [DATASET NAME]\n\n"
  papermill utils/torch_cnn_classifier.ipynb torch_cnn_experiments/$dt.ipynb -p DATASET $dt
done
