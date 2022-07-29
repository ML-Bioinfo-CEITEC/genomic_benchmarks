from pathlib import Path
from unittest import mock

import gdown
import pytest
from genomic_benchmarks.loc2seq.loc2seq import CLOUD_CACHE, download_from_cloud_cache


def test_download_from_cloud_cache_fails_for_unknown_key():
    with pytest.raises(ValueError):
        download_from_cloud_cache(file_key='unknown', dest_path=None)


@mock.patch("gdown.download", mock.Mock())
@mock.patch("shutil.unpack_archive", mock.Mock())
@mock.patch("pathlib.Path.unlink", mock.Mock())
def test_download_from_cloud_cache_removes_old_directory_for_force_download(tmp_path):
    dest_path = tmp_path / "sub"
    dir_path = dest_path / "subsub"
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / "hello.txt"
    file_path.write_text('dummy')

    download_from_cloud_cache(file_key=list(CLOUD_CACHE)[0], dest_path=dest_path, force_download=True)

    assert not Path(dest_path).exists()


@mock.patch("shutil.unpack_archive", mock.Mock())
@mock.patch("pathlib.Path.unlink", mock.Mock())
def test_download_from_cloud_cache_calls_download_correctly(tmp_path, monkeypatch):
    file_key = list(CLOUD_CACHE)[0]
    dest_path = tmp_path / "sub"
    download_mock = mock.MagicMock()
    monkeypatch.setattr(gdown, 'download', download_mock)

    download_from_cloud_cache(file_key=file_key, dest_path=dest_path, force_download=False)

    download_mock.assert_called_with(id=CLOUD_CACHE[file_key], output=str(dest_path) + ".zip")


@mock.patch("shutil.unpack_archive", mock.Mock())
@mock.patch("pathlib.Path.unlink", mock.Mock())
@mock.patch("gdown.download", mock.Mock())
def test_download_from_cloud_cache_returns_correct_path(tmp_path):
    file_key = list(CLOUD_CACHE)[0]
    expected_dest_path = tmp_path / "sub"

    actual_dest_path = download_from_cloud_cache(file_key=file_key, dest_path=expected_dest_path, force_download=False)

    assert actual_dest_path == expected_dest_path
