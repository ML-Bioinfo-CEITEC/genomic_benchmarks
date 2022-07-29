from pathlib import Path
from unittest import mock

import genomic_benchmarks.utils.datasets
import pytest
import yaml
from genomic_benchmarks.loc2seq import download_dataset


def test_download_dataset_processes_dataset_correctly(tmp_path, monkeypatch):
    dataset_name = "human_nontata_promoters"
    version_arg = 0
    dest_path = tmp_path / 'dset_folder'
    expected_dest_path = Path(dest_path) / dataset_name
    dataset_path = Path(__file__).parent.parent / "test_data" / "datasets" / dataset_name
    with open(dataset_path / "metadata.yaml", "r") as fr:
        metadata = yaml.safe_load(fr)

    _guess_location_mock = mock.Mock(return_value=dataset_path)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, "_guess_location", _guess_location_mock
    )
    _get_dataset_name_mock = mock.Mock(return_value=dataset_name)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, "_get_dataset_name", _get_dataset_name_mock
    )
    _check_dataset_existence_mock = mock.Mock(return_value=metadata, spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, "_check_dataset_existence", _check_dataset_existence_mock
    )

    actual_dest_path = download_dataset(dataset_name, version_arg, dest_path, local_repo=False, use_cloud_cache=False)

    assert actual_dest_path == expected_dest_path
    assert (expected_dest_path).exists()
    assert any(Path(expected_dest_path / "train").iterdir())
    assert any(Path(expected_dest_path / "test").iterdir())
