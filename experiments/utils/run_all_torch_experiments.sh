DATASETS=`ls -1d ../datasets/*/`

for d in $DATASETS 
do
  dt=`basename $d`
  echo "Processing dataset $dt\n\n"
  papermill utils/torch_cnn_classifier.ipynb torch_cnn_experiments/$dt.ipynb -p DATASET $dt -p EPOCHS 10
done
