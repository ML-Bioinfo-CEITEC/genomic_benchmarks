import shutil
from pathlib import Path

import shutil
import gdown

# the files can be found in https://drive.google.com/drive/folders/1hV1swtSjun0TVYtuWBERkZVajvMgK-II (ask Petr for permission to write)
# the value is file_id that can be found by right-clicking on the file and selecting "Get link"
CLOUD_CACHE = {
    ("human_nontata_promoters", 0): "1VdUg0Zu8yfLS6QesBXwGz1PIQrTW3Ze4",
    ("demo_coding_vs_intergenomic_seqs", 0): "1cpXg0ULuTGF7h1_HTYvc6p8M-ee43t-v",
    ("demo_human_or_worm", 0): "1JW0-eTB-rJXvFcglqBo3pFZi1kyIWC3X",
    ("dummy_mouse_enhancers_ensembl", 0): "1dz7dwvyM1TVUsCyuxxxn1ZMMFV381sl2",
    ("human_enhancers_cohn", 0): "176563cDPQ5Y094WyoSBF02QjoVQhWuCh",
    ("human_enhancers_ensembl", 0): "1gZBEV_RGxJE8EON5OObdrp5Tp8JL0Fxb",
    ("human_ocr_ensembl", 0): "1y_LInRF2aRXysigpwv_oU3Q67VVxfk18",
    ("human_ensembl_regulatory", 0): "1GefcGAM-tklnmzfzZ9RC91tGDmOzrAXU"
}


def download_from_cloud_cache(file_key, dest_path, cloud_cache=CLOUD_CACHE, force_download=True):
    """
    Download a file from the cloud cache.
    """
    if file_key not in cloud_cache:
        raise ValueError(f"File ID {file_key} not in the cloud cache.")

    if force_download and Path(dest_path).exists():
        shutil.rmtree(str(dest_path))

    gdown.download(
        id = cloud_cache[file_key],
        output = str(dest_path) + ".zip",
    )

    shutil.unpack_archive(str(dest_path) + ".zip", Path(dest_path).parent)
    Path(str(dest_path) + ".zip").unlink()

    return Path(dest_path)
