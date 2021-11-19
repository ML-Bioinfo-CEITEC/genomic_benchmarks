from pathlib import Path

from yarl import URL

# local path where the package stores its data
CACHE_PATH = Path.home() / ".genomic_benchmarks"
# local path where the references are stored
REF_CACHE_PATH = CACHE_PATH / "fasta"
# where the interval datasets are stored (if read from local)
DATASET_DIR_PATH = (Path(__file__).parents[0] / ".." / ".." / ".." / "datasets").resolve()
# where the interval datasets are stored (if read from the internet)
DATASET_URL_PATH = URL("http://raw.githubusercontent.com/ML-Bioinfo-CEITEC/genomic_benchmarks/main/datasets/")
