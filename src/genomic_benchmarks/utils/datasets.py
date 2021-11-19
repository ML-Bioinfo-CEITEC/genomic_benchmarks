import gzip
import shutil  # for removing and creating folders
import urllib
import warnings
from pathlib import Path

import requests
import yaml
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm.autonotebook import tqdm
from yarl import URL

from .paths import DATASET_DIR_PATH, DATASET_URL_PATH


def _guess_location(dataset_path, local_repo: bool = False):
    if Path(dataset_path).exists():
        return Path(dataset_path)
    elif local_repo and (DATASET_DIR_PATH / str(dataset_path)).exists():
        return DATASET_DIR_PATH / str(dataset_path)
    elif (not local_repo) and requests.get(DATASET_URL_PATH / str(dataset_path) / "metadata.yaml").status_code == 200:
        return DATASET_URL_PATH / str(dataset_path)
    else:
        raise FileNotFoundError(f"Dataset {dataset_path} not found.")


def _check_dataset_existence(interval_list_dataset, version, local_repo: bool = False):
    # check that the dataset exists, returns its metadata

    if local_repo:
        path = Path(interval_list_dataset)
        if not path.exists():
            raise FileNotFoundError(f"Dataset {interval_list_dataset} not found.")

        metadata_path = path / "metadata.yaml"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset {interval_list_dataset} does not contain `metadata.yaml` file.")
        with open(metadata_path, "r") as fr:
            metadata = yaml.safe_load(fr)
    else:
        url = URL(interval_list_dataset)
        with urllib.request.urlopen(str(url / "metadata.yaml")) as fr:
            metadata = yaml.safe_load(fr)

    if version is not None:
        if version != metadata["version"]:
            raise ValueError(f"Dataset version {version} does not match the version in metadata {metadata['version']}.")
    else:
        warnings.warn(f"No version specified. Using version {metadata['version']}.")

    return metadata


def _get_dataset_name(path):
    # get the dataset name from the path
    return Path(str(path)).stem


def _get_reference_name(url):
    # get the reference name from the url
    # TODO: better naming scheme (e.g. taking the same file from 2 Ensembl releases)
    return url.split("/")[-1]


def _download_url(url, dest):
    # download a file from url to dest
    if Path(dest).exists():
        Path(dest).unlink()

    print(f"Downloading {url}")

    class DownloadProgressBar(tqdm):
        # for progress bar
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=str(dest)) as t:
        # TODO: adapt fastdownload code instead of urllib
        urllib.request.urlretrieve(url, filename=dest, reporthook=t.update_to)


def _fastagz2dict(fasta_path, fasta_total=None, stop_id=None, region_name_transform=lambda x: x):
    # load gzipped fasta into dictionary
    fasta = {}

    with gzip.open(fasta_path, "rt") as handle:
        for record in tqdm(SeqIO.parse(handle, "fasta"), total=fasta_total):
            fasta[region_name_transform(record.id)] = str(record.seq)
            if stop_id and (record.id == stop_id):
                # stop, do not read small contigs
                break
    return fasta


def _rev(seq, strand):
    # reverse complement
    if strand == "-":
        return str(Seq(seq).reverse_complement())
    else:
        return seq


def _remove_and_create(path):
    # cleaning step: remove the folder and then create it again
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
