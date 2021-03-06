import urllib.request
from pathlib import Path
from unittest import mock

import pytest
import requests
from genomic_benchmarks.loc2seq.loc2seq import EXTRA_PREPROCESSING
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
from genomic_benchmarks.utils.paths import DATASET_DIR_PATH, DATASET_URL_PATH


def test__guess_location_returns_path_for_existing_path():
    expected = Path.home()

    actual = _guess_location(dataset_path = expected)

    assert expected == actual


def test__guess_location_returns_path_for_local_repo_and_dataset_in_datasets_folder(monkeypatch):
    dataset_path = 'dummy'
    expected =  DATASET_DIR_PATH / dataset_path
    def path_exists_mock(path):
        if str(path) == str(expected):
            return True
        else:
            return False
    monkeypatch.setattr(Path, 'exists', path_exists_mock)

    actual = _guess_location(dataset_path = dataset_path, local_repo=True)

    assert expected == actual


def test__guess_location_returns_path_for_not_local_repo_and_existing_url(monkeypatch):
    dataset_path = 'dummy'
    expected = DATASET_URL_PATH / str(dataset_path)

    def path_exists_mock(path):
        return False
    monkeypatch.setattr(Path, 'exists', path_exists_mock)

    def requests_get_mock(path):
        status_code_mock = mock.Mock()
        if (expected / "metadata.yaml" == path):
            status_code_mock.status_code = 200
            return status_code_mock
        else:
            status_code_mock.status_code = 'wrong url'
            return status_code_mock
    monkeypatch.setattr(requests, 'get', requests_get_mock)

    actual = _guess_location(dataset_path = dataset_path, local_repo=False)

    assert expected == actual


def test__guess_location_fails_for_None_path():
    with pytest.raises(TypeError):
        _guess_location(dataset_path = None)


def test__guess_location_fails_for_not_existing_dataset():
    with pytest.raises(FileNotFoundError):
        _guess_location(dataset_path = 'not_existing')


def test__check_dataset_existence_fails_for_not_existing_path_and_local_repo():
    with pytest.raises(FileNotFoundError):
        _check_dataset_existence('not_existing', version=0, local_repo=True)


def test__check_dataset_existence_fails_for_not_existing_metadata_and_local_repo():
    with pytest.raises(FileNotFoundError):
        _check_dataset_existence('', version=0, local_repo=True)


# ToDo Add tests for _check_dataset_existence
# def test__check_dataset_existence_calls_urlopen_correctly(monkeypatch):
#     url = 'http://www.example.com/'
#     expected =  url + "metadata.yaml"

# # https://stackoverflow.com/questions/1289894/how-do-i-mock-an-open-used-in-a-with-statement-using-the-mock-framework-in-pyth
#     # urlopen_mock = mock.MagicMock()
#     monkeypatch.setattr(urllib.request, 'urlopen', urlopen_mock)

#     _check_dataset_existence(url, version=0, local_repo=False)

#     urlopen_mock.assert_called_with(expected)


def test__get_dataset_name_returns_correct_string_from_path():
    expected = 'name'

    actual = _get_dataset_name('C:/an_/exaple-of__]a][[--path/  /' + expected)

    assert expected == actual


def test__get_dataset_name_returns_empty_for_directory_path():
    expected = 'name'

    actual = _get_dataset_name('/a/directory/path/' + expected + '/')

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


def test__get_reference_name_correctly_returns_name():
    expected = '50016862'

    actual = _get_reference_name('https://stackoverflow.com/questions/' + expected)

    assert expected == actual


def test__get_reference_name_returns_empty_string_for_empty_reference_name():
    expected = ''

    actual = _get_reference_name('https://stackoverflow.com/questions/50016862/')

    assert expected == actual


def test__get_reference_name_returns_empty_string_for_empty_input():
    expected = ''

    actual = _get_reference_name('')

    assert expected == actual


def test__get_reference_name_fails_for_None_input():
    with pytest.raises(AttributeError):
        _get_reference_name(None)


def test__download_url_calls_urlretrieve_correctly(monkeypatch):
    url = 'http://www.example.com/'
    dest = 'dummy_dest'

    path_exists_mock = mock.Mock(return_value=False)
    monkeypatch.setattr(Path, 'exists', path_exists_mock)

    urlretrieve_mock = mock.MagicMock()
    monkeypatch.setattr(urllib.request, 'urlretrieve', urlretrieve_mock)

    _download_url(url, dest)

    urlretrieve_mock.assert_called_with(url, filename = dest, reporthook = mock.ANY)


def test__download_url_calls_unlink_for_existing_path(monkeypatch):
    url = 'http://www.example.com/'
    dest = 'dummy_dest'

    path_exists_mock = mock.Mock(return_value=True)
    monkeypatch.setattr(Path, 'exists', path_exists_mock)

    unlink_mock = mock.MagicMock()
    monkeypatch.setattr(Path, 'unlink', unlink_mock)

    _download_url(url, dest)

    unlink_mock.assert_called()


def test__fastagz2dict():
    expected = {'test': 'ACCCCAGGGGAACTTCCAAGCACCTGAGCTTTTGTCCTTTCTAGTTCTGTGCATAGGTGCTGCCTTCCTGGGGGTAGGCACTTCTGTGAGTCCCTGCTGCAGGGGCAGAAGTGCATGTCTGCTGCACCAGAAGCCCCCGCTATTGATGCTCTCGGCTCCATTGGGAGAGCAGCTGCCAACTCAGCTTCTTCTACCTCCTACTTCATGGCAGACCAGGCCTTCACAGCTAATGGGGAGAGGAACTCATGTTACCTCTGCAGGCCTGGGGTCCTGAGGGGGTCTTTTGGCTTCAGCCTGTTCCCCCAGAGGCTTGATCATCCCACATTGTCCCTTCAGCTCAGCTGCTCTTCTCCCCCACCCACCCTGGGATGTGGGTGCTCTGGGCTGAACCAAGGCTATGGAGCCATCCTAGGGCAGGTAGCACTGAGGCTCCTGTGGAAACAGGAGCCACCTGCTCAGGAGACCCCTTTCCTGAGGAAGTCCTTACCTCTCCCCTTGAGATGTAAAAATGGTCCAGCAGAGACAAGCTCCCGTGGAAAACAGACAGGAGCATGGGGGCAGCTGTCATGGCTGTGGCGGGCACTTTTCCTCAGAGTTTCT'}
    fasta_path = Path(__file__).parents[0] / '../test_data/test_reference.fa.gz'

    actual = _fastagz2dict(fasta_path)

    assert expected == actual


def test__fastagz2dict_fails_for_not_existing_path():
    fasta_path = Path(__file__).parents[0] / '../test_data/not_existing.fa.gz'

    with pytest.raises(FileNotFoundError):
        _fastagz2dict(fasta_path)


def test__rev_returns_reverse_complement():
    expected = 'YNCTATCGGGGG'

    actual = _rev(seq = 'CCCCCGATAGNR', strand = '-')

    assert expected == actual


def test__rev_returns_identity_for_empty_strand():
    expected = 'CCCCCGATAGNR'

    actual = _rev(seq = expected, strand = None)

    assert expected == actual


def test__rev_fails_for_None_seq_with_strand():
    with pytest.raises(TypeError):
        _rev(seq = None, strand = '-')


def test__rev_returns_identity_for_empty_string_with_strand():
    expected = ''

    actual = _rev(seq = expected, strand = '-')

    assert expected == actual


def test__rev_returns_identity_for_empty_string_without_strand():
    expected = ''

    actual = _rev(seq = expected, strand = None)

    assert expected == actual


def test__remove_and_create_for_existing_directory(tmp_path):
    dir_path = tmp_path / "sub" / "subsub"
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / "hello.txt"
    file_path.write_text('dummy')

    _remove_and_create(dir_path)

    assert Path(dir_path).exists()
    assert not Path(file_path).exists()


def test__remove_and_create_creates_directory_for_not_existing_directory(tmp_path):
    dir_path = tmp_path / "sub" / "subsub"

    _remove_and_create(dir_path)

    assert Path(dir_path).exists()

