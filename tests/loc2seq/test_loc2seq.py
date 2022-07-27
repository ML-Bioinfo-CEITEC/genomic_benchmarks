import pathlib
import shutil
from ensurepip import version
from pathlib import Path
from unittest import mock

import genomic_benchmarks.utils.datasets
import pandas as pd
import pytest
from genomic_benchmarks.loc2seq import remove_dataset_from_disk

# from genomic_benchmarks.utils.datasets import (
#     _check_dataset_existence,
#     _download_url,
#     _fastagz2dict,
#     _get_dataset_name,
#     _get_reference_name,
#     _guess_location,
#     _remove_and_create,
#     _rev,
# )


def test_remove_dataset_from_disk_calls_helper_methods_correctly(monkeypatch):
    interval_list_dataset_argument = "interval_list_dataset"
    version_argument = "version"
    _guess_location_mock = mock.MagicMock(return_value = interval_list_dataset_argument, spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.utils.datasets, '_guess_location', _guess_location_mock
    )
    _check_dataset_existence_mock = mock.MagicMock(spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.utils.datasets, '_check_dataset_existence', _check_dataset_existence_mock
    )
    _get_dataset_name_mock = mock.MagicMock(spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.utils.datasets, '_get_dataset_name', _get_dataset_name_mock
    )

    remove_dataset_from_disk(interval_list_dataset_argument, dest_path='', version=version_argument)

    _guess_location_mock.assert_called_with(interval_list_dataset_argument)
    _check_dataset_existence_mock.assert_called_with(interval_list_dataset_argument, version_argument)
    _get_dataset_name_mock.assert_called_with(interval_list_dataset_argument)


@mock.patch("genomic_benchmarks.utils.datasets._guess_location", mock.Mock())
@mock.patch("genomic_benchmarks.utils.datasets._check_dataset_existence", mock.Mock())
def test_remove_dataset_from_disk_removes_dataset(tmp_path, monkeypatch):
    dataset_name = 'dataset_name'
    dest_path = tmp_path / 'folder'
    path_should_not_exist = dest_path / dataset_name
    path_should_not_exist.mkdir(parents=True, exist_ok=True)
    _get_dataset_name_mock = mock.MagicMock(return_value = dataset_name,spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.utils.datasets, '_get_dataset_name', _get_dataset_name_mock
    )

    remove_dataset_from_disk("interval_list_dataset", dest_path=dest_path)

    assert not Path(path_should_not_exist).exists()
