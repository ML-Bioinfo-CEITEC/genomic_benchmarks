from pathlib import Path
from unittest import mock

import pytest
from genomic_benchmarks.loc2seq.loc2seq import CLOUD_CACHE, download_from_cloud_cache


def test_download_from_cloud_cache_fails_for_unknown_ket():
    with pytest.raises(ValueError):
        download_from_cloud_cache(file_key='unknown', dest_path=None)

# @mock.patch("gdown.download", mock.Mock())
# @mock.patch("shutil.unpack_archive", mock.Mock())
# def test_download_from_cloud_cache_removes_old_directory_for_force_download(tmp_path):
#     dest_path = tmp_path / "sub"
#     dir_path = dest_path / "subsub"
#     dir_path.mkdir(parents=True, exist_ok=True)
#     file_path = dir_path / "hello.txt"
#     file_path.write_text('dummy')
#     # mock
#     # gdown.download
#     # shutil.unpack_archive
#     # Path(str(dest_path) + ".zip").unlink()

#     download_from_cloud_cache(file_key=list(CLOUD_CACHE)[0], dest_path=dest_path, force_download=True)

#     assert not Path(dest_path).exists()




