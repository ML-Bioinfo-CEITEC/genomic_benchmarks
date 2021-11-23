DATASETS=`ls -1d ../datasets/*/`

for d in $DATASETS 
do
  dt=`basename $d`
  echo "Processing dataset $dt\n\n"
  papermill utils/tf_cnn_classifier.ipynb tf_cnn_experiments/$dt.ipynb -p DATASET $dt
done
