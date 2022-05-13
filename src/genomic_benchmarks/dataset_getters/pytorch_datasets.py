from genomic_benchmarks.data_check import is_downloaded
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.utils.paths import CACHE_PATH
from torch.utils.data import Dataset


class GenomicClfDataset(Dataset):
    """
    A class to represent generic genomic classification pytorch dataset.
    Instance of this class can be directly wrapped by pytorch DataLoader
    """

    def __init__(self, dset_name, split, force_download=False, version=None):
        """
        Parameters
            dset_name : str
                One of the existing dataset names, list available at TODO
            split : str
                train' or 'test'
            force_download : bool
                Whether to re-download already existing files
            version : int
                Version of the dataset
        """
        base_path = CACHE_PATH / dset_name
        if not is_downloaded(dset_name):
            download_dataset(dset_name, version=version, force_download=force_download)

        if split == "train" or split == "test":
            base_path = base_path / split
        else:
            raise Exception("Define split, train or test")

        self.all_paths = []
        self.all_labels = []
        label_mapper = {}

        for i, x in enumerate(base_path.iterdir()):
            label_mapper[x.stem] = i

        for label_type in label_mapper.keys():
            for x in (base_path / label_type).iterdir():
                self.all_paths.append(x)
                self.all_labels.append(label_mapper[label_type])

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        txt_path = self.all_paths[idx]
        with open(txt_path, "r") as f:
            content = f.read()
        x = content
        y = self.all_labels[idx]
        return x, y


def get_dataset(dataset_name, split, force_download=False, version=None):
    return GenomicClfDataset(dataset_name, split, force_download, version)


def DemoCodingVsIntergenomicSeqs(split, force_download=False, version=None):
    return GenomicClfDataset("demo_coding_vs_intergenomic_seqs", split, force_download, version)


def DemoHumanOrWorm(split, force_download=False, version=None):
    return GenomicClfDataset("demo_human_or_worm", split, force_download, version)


def DemoMouseEnhancers(split, force_download=False, version=None):
    return GenomicClfDataset("dummy_mouse_enhancers_ensembl", split, force_download, version)


def HumanEnhancersCohn(split, force_download=False, version=None):
    return GenomicClfDataset("human_enhancers_cohn", split, force_download, version)


def HumanEnhancersEnsembl(split, force_download=False, version=None):
    return GenomicClfDataset("human_enhancers_ensembl", split, force_download, version)


def HumanNontataPromoters(split, force_download=False, version=None):
    return GenomicClfDataset("human_nontata_promoters", split, force_download, version)

def HumanOcrEnsembl(split, force_download=False, version=None):
    return GenomicClfDataset("human_ocr_ensembl", split, force_download, version)
