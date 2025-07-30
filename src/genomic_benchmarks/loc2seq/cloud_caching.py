import shutil
from pathlib import Path

import urllib.request
import shutil

# files are cached on Zenodo: https://zenodo.org/records/16605299
CLOUD_CACHE = {
    ("human_nontata_promoters", 0): "https://zenodo.org/records/16605299",
    ("demo_coding_vs_intergenomic_seqs", 0): "https://zenodo.org/records/16605299",
    ("demo_human_or_worm", 0): "https://zenodo.org/records/16605299",
    ("drosophila_enhancers_stark", 0): "https://zenodo.org/records/16605299",
    ("dummy_mouse_enhancers_ensembl", 0): "https://zenodo.org/records/16605299",
    ("human_enhancers_cohn", 0): "https://zenodo.org/records/16605299",
    ("human_enhancers_ensembl", 0): "https://zenodo.org/records/16605299",
    ("human_ocr_ensembl", 0): "https://zenodo.org/records/16605299",
    ("human_ensembl_regulatory", 0): "https://zenodo.org/records/16605299"
}

def download_from_cloud_cache(file_key, dest_path, cloud_cache=CLOUD_CACHE, force_download=True):
    """
    Download a file from the cloud cache.
    """
    if file_key not in cloud_cache:
        raise ValueError(f"File ID {file_key} not in the cloud cache.")

    if force_download and Path(dest_path).exists():
        shutil.rmtree(str(dest_path))

    url = f'{cloud_cache[file_key]}/files/{file_key[0]}_v{file_key[1]}.zip?download=1'
    urllib.request.urlretrieve(url, str(dest_path) + ".zip")

    shutil.unpack_archive(str(dest_path) + ".zip", Path(dest_path).parent)
    Path(str(dest_path) + ".zip").unlink()

    return Path(dest_path)
