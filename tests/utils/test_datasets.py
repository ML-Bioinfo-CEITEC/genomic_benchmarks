from pathlib import Path
from threading import activeCount

import pytest
from genomic_benchmarks.utils.datasets import _get_dataset_name


def test__get_dataset_name_returns_correct_string_from_path():
    expected = 'name'

    actual = _get_dataset_name('C:/an_/exaple-of__]a][[--path/  /' + expected)

    assert expected == actual


def test__get_dataset_name_returns_correct_string():
    expected = 'name'

    actual = _get_dataset_name(expected)

    assert expected == actual


# ToDo _get_dataset_name(None) should not return the string 'None'
def test__get_dataset_name_fails_for_None():
    expected = None

    actual = _get_dataset_name(expected)

    assert not expected == actual


def test__get_dataset_name_returns_empty_for_empty_string():
    expected = ''

    actual = _get_dataset_name(expected)

    assert expected == actual


# ToDo is correct behaviour? _get_dataset_name('/a/directory/path/') returns "path"
def test__get_dataset_name_returns_empty_for_directory_path():
    expected = ''

    actual = _get_dataset_name('/a/directory/path/' + expected)

    # assert expected == actual

