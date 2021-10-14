# Tools

If you are adding a new tool, please, add here a short description. Do not forget to properly document your tool. Tests and demos help users to start smoothly.

## loc2seq

With `create_seq_genomic_dataset` function, you can transform a dataset given as a list of intervals (CSV files + yaml metadata file) into a dataset with the actual genomic sequences (TXT files). See the example in [demo_notebook.ipynb](loc2seq/demo/demo_notebook.ipynb).

## seq2loc

Sometimes we want to include a benchmark that was given to us as full sequences, not genomic intervals. This tool search for perfect matches in a reference and output a dataframe with genomic locations. For example, see [human non-tata promoters dataset](../../docs/human_nontata_promoters/create_datasets.ipynb).