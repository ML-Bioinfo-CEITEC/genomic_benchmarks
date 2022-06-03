import re
import shutil  # for removing and creating folders
from pathlib import Path

import pandas as pd
from genomic_benchmarks.utils.datasets import (
    _check_dataset_existence,
    _download_url,
    _fastagz2dict,
    _get_dataset_name,
    _get_reference_name,
    _guess_location,
    _remove_and_create,
    _rev,
)
from genomic_benchmarks.utils.paths import CACHE_PATH, REF_CACHE_PATH

from .cloud_caching import CLOUD_CACHE, download_from_cloud_cache


def download_dataset(
    interval_list_dataset,
    version=None,
    dest_path=CACHE_PATH,
    cache_path=REF_CACHE_PATH,
    force_download=False,
    use_cloud_cache=True,
    local_repo=False,
):
    """
    Transform an interval-list genomic dataset into a full-seq genomic dataset.

            Parameters:
                    interval_list_dataset (str or Path): Either a path or a name of dataset included in this package.
                    version (int): Version of the dataset.
                    dest_path (str or Path): Folder to store the full-seq dataset.
                    cache_path (str or Path): Folder to store the downloaded references.
                    force_download (bool): If True, force downloading of references.
                    use_cloud_cache (bool): If True, use the cloud cache for downloading a full-seq genomic datasets.
                    local_repo (bool): If True, use the local repo for getting interval-list genomic datasets.

            Returns:
                    seq_dataset_path (Path): Path to the full-seq dataset.
    """

    interval_list_dataset = _guess_location(interval_list_dataset, local_repo)
    metadata = _check_dataset_existence(interval_list_dataset, version, local_repo)
    dataset_name = _get_dataset_name(interval_list_dataset)

    if version is None:
        version = metadata["version"]
    if use_cloud_cache and ((dataset_name, version) in CLOUD_CACHE):
        Path(dest_path).mkdir(parents=True, exist_ok=True)  # to be sure "./.genomic_benchmarks" exists
        return download_from_cloud_cache((dataset_name, version), Path(dest_path) / dataset_name)

    refs = _download_references(metadata, cache_path=cache_path, force=force_download)
    fastas = _load_fastas_into_memory(refs, cache_path=cache_path)

    _remove_and_create(Path(dest_path) / dataset_name)
    _remove_and_create(Path(dest_path) / dataset_name / "train")
    _remove_and_create(Path(dest_path) / dataset_name / "test")

    for c in metadata["classes"]:
        for t in ["train", "test"]:
            dt_filename = interval_list_dataset / t / (c + ".csv.gz")
            dt = pd.read_csv(str(dt_filename), compression="gzip")

            ref_name = _get_reference_name(metadata["classes"][c]["url"])
            dt["seq"] = _fill_seq_column(fastas[ref_name], dt)

            folder_filename = Path(dest_path) / dataset_name / t / c
            _remove_and_create(folder_filename)
            for row in dt.iterrows():
                row_filename = folder_filename / (str(row[1]["id"]) + ".txt")
                row_filename.write_text(row[1]["seq"])

    return Path(dest_path) / dataset_name


EXTRA_PREPROCESSING = {
    # known extra preprocessing steps
    "default": [None, None, lambda x: x],
    "ENSEMBL_HUMAN_GENOME": [24, "MT", lambda x: "chr" + x],  # use only chromosomes, not contigs, and add chr prefix
    "ENSEMBL_MOUSE_GENOME": [21, "MT", lambda x: "chr" + x],  # use only chromosomes, not contigs, and add chr prefix
    "ENSEMBL_HUMAN_TRANSCRIPTOME": [
        190_000,
        None,
        lambda x: re.sub("ENST([0-9]*)[.][0-9]*", "ENST\\1", x),
    ],  # remove the version number from the ensembl id
}


def _load_fastas_into_memory(refs, cache_path):
    # load all references into memory
    fastas = {}
    for ref in refs:
        ref_path = Path(cache_path) / _get_reference_name(ref[0])
        ref_type = ref[1]
        ref_extra_preprocessing = ref[2] if ref[2] is not None else "default"
        if ref_extra_preprocessing not in EXTRA_PREPROCESSING:
            raise ValueError(f"Unknown extra preprocessing: {ref_extra_preprocessing}")

        if ref_type == "fa.gz":
            fasta = _fastagz2dict(
                ref_path,
                fasta_total=EXTRA_PREPROCESSING[ref_extra_preprocessing][0],
                stop_id=EXTRA_PREPROCESSING[ref_extra_preprocessing][1],
                region_name_transform=EXTRA_PREPROCESSING[ref_extra_preprocessing][2],
            )
            fastas[_get_reference_name(ref[0])] = fasta
        else:
            raise ValueError(f"Unknown reference type {ref_type}")
    return fastas


def _download_references(metadata, cache_path, force=False):
    # download all references from the metadata into cache_path folder
    cache_path = Path(cache_path)
    if not cache_path.exists():
        cache_path.mkdir(parents=True)

    refs = {(c["url"], c["type"], c.get("extra_processing")) for c in metadata["classes"].values()}

    for ref in refs:
        ref_path = cache_path / _get_reference_name(ref[0])
        if not ref_path.exists() or force:
            _download_url(ref[0], ref_path)
        else:
            print(f"Reference {ref_path} already exists. Skipping.")

    return refs


def _fill_seq_column(fasta, df):
    # fill seq column in DataFrame tab
    if not all([r in fasta for r in df["region"]]):
        missing_regions = list({r for r in df["region"] if r not in fasta})
        if len(missing_regions) > 5:
            missing_regions = missing_regions[:6]
        raise ValueError("Some regions not found in the reference, e.g. " + " ".join([str(r) for r in missing_regions]))
    output = pd.Series(
        [
            _rev(fasta[region][start:end], strand)
            for region, start, end, strand in zip(df["region"], df["start"], df["end"], df["strand"])
        ]
    )
    return output


def remove_dataset_from_disk(interval_list_dataset, version=None, dest_path=CACHE_PATH):
    """
    Remove the full-seq dataset from the disk.

            Parameters:
                    interval_list_dataset (str or Path): Either a path or a name of dataset included in this package.
                    version (int): Version of the dataset.
                    dest_path (str or Path): Folder to store the full-seq dataset.
    """
    interval_list_dataset = _guess_location(interval_list_dataset)
    metadata = _check_dataset_existence(interval_list_dataset, version)
    dataset_name = _get_dataset_name(interval_list_dataset)

    path = Path(dest_path) / dataset_name
    if path.exists():
        shutil.rmtree(path)
