import shutil
from pathlib import Path

from google_drive_downloader import (
    GoogleDriveDownloader as gdd,  # TODO: use gdown instead of google_drive_downloader
)

CLOUD_CACHE = {
    ("human_nontata_promoters", 0): "1VdUg0Zu8yfLS6QesBXwGz1PIQrTW3Ze4",
    ("demo_coding_vs_intergenomic_seqs", 0): "1cpXg0ULuTGF7h1_HTYvc6p8M-ee43t-v",
    ("demo_human_or_worm", 0): "1Vuc44bXRISqRDXNrxt5lGYLpLsJbrSg8",
    ("demo_mouse_enhancers", 0): "1u3pyaL8smQaJXeOx7YZkjj-Bdpb-jGCM",
    ("human_enhancers_cohn", 0): "176563cDPQ5Y094WyoSBF02QjoVQhWuCh",
    ("human_enhancers_ensembl", 0): "1gZBEV_RGxJE8EON5OObdrp5Tp8JL0Fxb",
}


def download_from_cloud_cache(file_key, dest_path, cloud_cache=CLOUD_CACHE, force_download=True):
    """
    Download a file from the cloud cache.
    """
    if file_key not in cloud_cache:
        raise ValueError(f"File ID {file_key} not in the cloud cache.")

    if force_download and Path(dest_path).exists():
        shutil.rmtree(str(dest_path))

    gdd.download_file_from_google_drive(file_id=cloud_cache[file_key], dest_path=str(dest_path) + ".zip", unzip=True)
    Path(str(dest_path) + ".zip").unlink()

    return Path(dest_path)
