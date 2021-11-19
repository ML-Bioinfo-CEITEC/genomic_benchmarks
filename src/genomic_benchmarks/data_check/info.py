from pathlib import Path

import numpy as np
import pandas as pd
from genomic_benchmarks.loc2seq.cloud_caching import CLOUD_CACHE
from genomic_benchmarks.utils.datasets import (
    _check_dataset_existence,
    _get_dataset_name,
    _guess_location,
)
from genomic_benchmarks.utils.paths import CACHE_PATH, DATASET_DIR_PATH

# TODO: Many of these functions are not prepared for the case when the folder in DATASET_DIR_PATH is not one benchmark but a set of benchmarks.


def info(interval_list_dataset, version=None, local_repo: bool = False):
    """
    Print info about the bechmark.

            Parameters:
                    interval_list_dataset (str or Path): Either a path or a name of dataset included in this package.

            Returns:
                    DataFrame with counts of seqeunces for each class in a training and testing sets.
    """

    interval_list_dataset = _guess_location(interval_list_dataset, local_repo)
    metadata = _check_dataset_existence(interval_list_dataset, version, local_repo)
    dataset_name = _get_dataset_name(interval_list_dataset)

    dfs = {}
    for c in metadata["classes"]:
        dfs[c] = {}
        for t in ["train", "test"]:
            dt_filename = interval_list_dataset / t / (c + ".csv.gz")
            dfs[c][t] = pd.read_csv(str(dt_filename), compression="gzip")

    classes = list(dfs.keys())
    print(f"Dataset `{dataset_name}` has {len(classes)} classes: " + ", ".join(classes) + ".\n")

    interval_lengths = np.concatenate([(dfs[c][t].end - dfs[c][t].start).to_numpy() for c in dfs for t in dfs[c]])

    if len(np.unique(interval_lengths)) == 1:
        print(f"All lenghts of genomic intervals equals {interval_lengths[0]}.\n")
    else:
        print(
            f"The lenght of genomic intervals ranges from {np.min(interval_lengths)} to {np.max(interval_lengths)}, with average {np.mean(interval_lengths)} and median {np.median(interval_lengths)}.\n"
        )

    dfs_counts = {}
    for c in dfs:
        dfs_counts[c] = {}
        for t in dfs[c]:
            dfs_counts[c][t] = dfs[c][t].shape[0]
    dfs_counts_df = pd.DataFrame(dfs_counts).T

    print(
        f"Totally {dfs_counts_df.to_numpy().sum()} sequences have been found, {dfs_counts_df.train.sum()} for training and {dfs_counts_df.test.sum()} for testing."
    )

    return dfs_counts_df


def is_downloaded(interval_list_dataset, cache_path=CACHE_PATH):
    """
    Check if the dataset is downloaded.

            Parameters:
                    interval_list_dataset (str or Path): Either a path or a name of dataset included in this package.
                    cache_path (Path): Path to the cache directory.

            Returns:
                    bool: True if the dataset is downloaded, False otherwise.
    """

    dataset_name = _get_dataset_name(interval_list_dataset)
    cache_path = Path(cache_path) / dataset_name
    return cache_path.exists()


def list_datasets(dataset_path=DATASET_DIR_PATH, local_repo: bool = False):
    """
    List all interval datasets in the package.

            Parameters:
                    dataset_path (Path): Path to the datasets folder.

            Returns:
                    list: List of dataset names.
    """

    if local_repo:
        cache_path = Path(dataset_path)
        return [x.name for x in cache_path.iterdir() if x.is_dir()]
    else:
        return list({x[0] for x in CLOUD_CACHE})
