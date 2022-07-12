import pathlib
import shutil
from pathlib import Path
from unittest import mock

import genomic_benchmarks.utils.datasets
import pandas as pd
import pytest
from genomic_benchmarks.loc2seq import loc2seq, remove_dataset_from_disk

# @mock.patch("genomic_benchmarks.utils.datasets._guess_location", mock.MagicMock(return_value='dummy_interval_list_dataset'))
# def test_remove_dataset_from_disk(monkeypatch):
#     print('00')

#     interval_list_dataset = ''

#     # def _guess_location_mock():
#     #     print('_guess_location_mock')
#     #     return interval_list_dataset
#     # monkeypatch.setattr(genomic_benchmarks.utils.datasets, '_guess_location', _guess_location_mock)

#     def _check_dataset_existence_mock():
#         return 'https://dummy.restapiexample.com/api/v1/employee/32'
#     monkeypatch.setattr(genomic_benchmarks.utils.datasets, '_check_dataset_existence', _check_dataset_existence_mock)

#     def _get_dataset_name_mock():
#         return ''
#     monkeypatch.setattr(genomic_benchmarks.utils.datasets, '_get_dataset_name', _get_dataset_name_mock)

#     def exists_mock(any):
#         return True
#     monkeypatch.setattr(Path, 'exists', exists_mock)

#     rmtree_mock = mock.MagicMock()
#     monkeypatch.setattr(shutil, 'rmtree', rmtree_mock)


#     remove_dataset_from_disk(interval_list_dataset, dest_path='')


#     rmtree_mock.assert_called_with(interval_list_dataset)
