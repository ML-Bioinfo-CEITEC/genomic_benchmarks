from pathlib import Path
from unittest import mock

import genomic_benchmarks.loc2seq
import genomic_benchmarks.utils.datasets
import pytest
import yaml
from genomic_benchmarks.loc2seq import download_dataset, remove_dataset_from_disk
from genomic_benchmarks.loc2seq.cloud_caching import CLOUD_CACHE
from genomic_benchmarks.loc2seq.loc2seq import EXTRA_PREPROCESSING
from genomic_benchmarks.utils.datasets import _get_reference_name
from genomic_benchmarks.utils.paths import REF_CACHE_PATH


def test_load_fastas_into_memory_calls__fastagz2dict_correctly(monkeypatch):
    cache_path = REF_CACHE_PATH
    refs = {(
        'http://ftp.ensembl.org/pub/release-97/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.toplevel.fa.gz',
        'fa.gz',
        'ENSEMBL_HUMAN_GENOME'
    )}
    ref_extra_preprocessing = next(iter(refs))[2]
    print(ref_extra_preprocessing)
    _fastagz2dict_mock = mock.MagicMock(return_value=None, spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, "_fastagz2dict", _fastagz2dict_mock
    )

    genomic_benchmarks.loc2seq.loc2seq._load_fastas_into_memory(refs, cache_path)

    _fastagz2dict_mock.assert_called_with(
        cache_path / _get_reference_name(next(iter(refs))[0]),
        fasta_total=EXTRA_PREPROCESSING[ref_extra_preprocessing][0],
        stop_id=EXTRA_PREPROCESSING[ref_extra_preprocessing][1],
        region_name_transform=EXTRA_PREPROCESSING[ref_extra_preprocessing][2],
    )


@pytest.mark.slow
def test_load_fastas_into_memory_returns_correct_fastas(tmp_path, monkeypatch):
    refs = {(
        'http://ftp.ensembl.org/pub/release-97/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.toplevel.fa.gz',
        'fa.gz',
        'ENSEMBL_HUMAN_GENOME'
    )}

    actual_fastas = genomic_benchmarks.loc2seq.loc2seq._load_fastas_into_memory(refs, REF_CACHE_PATH)

    assert actual_fastas['Homo_sapiens.GRCh38.dna.toplevel.fa.gz']['chr1']
    assert len(actual_fastas['Homo_sapiens.GRCh38.dna.toplevel.fa.gz']['chr1']) == 248956422


def test_load_fastas_into_memory_raises_error_on_wrong_ref_type():
    refs = {(
        'http://ftp.ensembl.org/pub/release-97/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.toplevel.fa.gz',
        'WRONG_REF_TYPE',
        'ENSEMBL_HUMAN_GENOME'
    )}

    with pytest.raises(ValueError):
        genomic_benchmarks.loc2seq.loc2seq._load_fastas_into_memory(refs, REF_CACHE_PATH)


def test_load_fastas_into_memory_raises_error_on_wrong_ref_extra_preprocessing():
    refs = {(
        'http://ftp.ensembl.org/pub/release-97/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.toplevel.fa.gz',
        'fa.gz',
        'WRONG_EXTRA_PREPROCESSING'
    )}

    with pytest.raises(ValueError):
        genomic_benchmarks.loc2seq.loc2seq._load_fastas_into_memory(refs, REF_CACHE_PATH)


# downloads real references
@pytest.mark.slow
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


def test_download_dataset_calls_helper_functions_correctly(monkeypatch):
    interval_list_dataset_arg = "interval_list_dataset"
    version_arg = "version"
    local_repo_arg = False
    expected_metadata = {'classes':[]}
    expected_cache_path = 'expected_cache_path'
    expected_force_download = 'expected_force_download'
    expected_refs = 'expected_refs'
    _guess_location_mock = mock.MagicMock(return_value = interval_list_dataset_arg, spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, '_guess_location', _guess_location_mock
    )
    _check_dataset_existence_mock = mock.MagicMock(return_value=expected_metadata, spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, '_check_dataset_existence', _check_dataset_existence_mock
    )
    _get_dataset_name_mock = mock.MagicMock(spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, '_get_dataset_name', _get_dataset_name_mock
    )
    _download_references_mock = mock.MagicMock(return_value = expected_refs, spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, '_download_references', _download_references_mock
    )
    _load_fastas_into_memory_mock = mock.MagicMock(spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, '_load_fastas_into_memory', _load_fastas_into_memory_mock
    )

    download_dataset(
        interval_list_dataset_arg,
        version_arg,
        '',
        expected_cache_path,
        expected_force_download,
        local_repo=local_repo_arg
    )

    _guess_location_mock.assert_called_with(interval_list_dataset_arg, local_repo_arg)
    _check_dataset_existence_mock.assert_called_with(interval_list_dataset_arg, version_arg, local_repo_arg)
    _get_dataset_name_mock.assert_called_with(interval_list_dataset_arg)
    _download_references_mock.assert_called_with(expected_metadata, cache_path=expected_cache_path, force=expected_force_download)
    _load_fastas_into_memory_mock.assert_called_with(expected_refs, cache_path=expected_cache_path)



@mock.patch("genomic_benchmarks.loc2seq.loc2seq._guess_location", mock.Mock())
@mock.patch("genomic_benchmarks.loc2seq.loc2seq._download_references", mock.Mock())
@mock.patch("genomic_benchmarks.loc2seq.loc2seq._load_fastas_into_memory", mock.Mock())
def test_download_dataset_downloads_from_cloud_cache_correctly(tmp_path, monkeypatch):
    cloud_cache_item = list(CLOUD_CACHE.items())
    cloud_cache_item.sort()
    dataset_name = cloud_cache_item[0][0][0]
    version_arg = cloud_cache_item[0][0][1]
    dest_path = tmp_path / 'folder'
    _get_dataset_name_mock = mock.Mock(return_value=dataset_name)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, "_get_dataset_name", _get_dataset_name_mock
    )
    _check_dataset_existence_mock = mock.Mock(return_value={"version":version_arg, "classes":{}}, spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, "_check_dataset_existence", _check_dataset_existence_mock
    )
    download_from_cloud_cache_mock = mock.MagicMock(spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, 'download_from_cloud_cache', download_from_cloud_cache_mock
    )

    download_dataset(dataset_name, version_arg, dest_path, local_repo=False, use_cloud_cache=True)

    download_from_cloud_cache_mock.assert_called_with((dataset_name, version_arg), Path(dest_path) / dataset_name)


@mock.patch("genomic_benchmarks.loc2seq.loc2seq._guess_location", mock.Mock())
@mock.patch("genomic_benchmarks.loc2seq.loc2seq._download_references", mock.Mock())
@mock.patch("genomic_benchmarks.loc2seq.loc2seq._load_fastas_into_memory", mock.Mock())
def test_download_dataset_creates_folder_structure(tmp_path, monkeypatch):
    cloud_cache_item = list(CLOUD_CACHE.items())
    cloud_cache_item.sort()
    dataset_name = cloud_cache_item[0][0][0]
    version_arg = cloud_cache_item[0][0][1]
    dest_path = tmp_path / 'folder'
    path_should_exist = Path(dest_path) / dataset_name
    _get_dataset_name_mock = mock.Mock(return_value=dataset_name)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, "_get_dataset_name", _get_dataset_name_mock
    )
    _check_dataset_existence_mock = mock.Mock(return_value={"classes":{}}, spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, "_check_dataset_existence", _check_dataset_existence_mock
    )

    assert not Path(path_should_exist).exists()
    download_dataset(dataset_name, version_arg, dest_path, local_repo=False, use_cloud_cache=False)

    assert Path(path_should_exist).exists()


def test_remove_dataset_from_disk_calls_helper_methods_correctly(monkeypatch):
    interval_list_dataset_argument = "interval_list_dataset"
    version_argument = "version"
    _guess_location_mock = mock.MagicMock(return_value = interval_list_dataset_argument, spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, '_guess_location', _guess_location_mock
    )
    _check_dataset_existence_mock = mock.MagicMock(spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, '_check_dataset_existence', _check_dataset_existence_mock
    )
    _get_dataset_name_mock = mock.MagicMock(spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, '_get_dataset_name', _get_dataset_name_mock
    )

    remove_dataset_from_disk(interval_list_dataset_argument, dest_path='', version=version_argument)

    _guess_location_mock.assert_called_with(interval_list_dataset_argument)
    _check_dataset_existence_mock.assert_called_with(interval_list_dataset_argument, version_argument)
    _get_dataset_name_mock.assert_called_with(interval_list_dataset_argument)


@mock.patch("genomic_benchmarks.loc2seq.loc2seq._guess_location", mock.Mock())
@mock.patch("genomic_benchmarks.loc2seq.loc2seq._check_dataset_existence", mock.Mock())
def test_remove_dataset_from_disk_removes_dataset(tmp_path, monkeypatch):
    dataset_name = 'dataset_name'
    dest_path = tmp_path / 'folder'
    path_should_not_exist = dest_path / dataset_name
    path_should_not_exist.mkdir(parents=True, exist_ok=True)
    _get_dataset_name_mock = mock.MagicMock(return_value = dataset_name,spec=True)
    monkeypatch.setattr(
        genomic_benchmarks.loc2seq.loc2seq, '_get_dataset_name', _get_dataset_name_mock
    )

    remove_dataset_from_disk("interval_list_dataset", dest_path=dest_path)

    assert not Path(path_should_not_exist).exists()
