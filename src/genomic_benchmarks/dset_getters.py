from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from genomic_benchmarks.loc2seq.with_biopython import download_dataset
from genomic_benchmarks.utils.data_utils import download_tarfile, untar_file


class genomic_clf_dset(Dataset): #TODO inherit mapstyledataset? https://pytorch.org/docs/stable/data.html#dataset-types
    def __init__(self, dset_name, split, force_download=False):
        datasets_folder_path = Path.home() / '.genomic_benchmarks'
        fasta_cache_path = datasets_folder_path / 'fasta'

        dset_path = Path(f'{datasets_folder_path}/{dset_name}')
        download_interval_dset(dest_path=datasets_folder_path, dset_name=dset_name)

        destination_path = Path(f'{datasets_folder_path}/translated_{dset_name}')
        base_path = download_dataset(dset_path, version=None, dest_path=destination_path, cache_path=fasta_cache_path, force_download=force_download)
        
        if(split == 'train' or split=='test'):
          base_path = base_path/split
        else:
          raise Exception('Define split, train or test')

        self.all_paths = []
        self.all_labels = []
        label_mapper = {}

        for i,x in enumerate(base_path.iterdir()):
          label_mapper[x.stem] = i

        for label_type in label_mapper.keys():
            for x in (base_path / label_type).iterdir():
                self.all_paths.append(x)
                self.all_labels.append(label_mapper[label_type])

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        txt_path = self.all_paths[idx]
        with open(txt_path, 'r') as f:
            content = f.read()
        x = content
        y = self.all_labels[idx]
        return x, y

def download_interval_dset(dest_path, dset_name, force_download=False):
  base_path = Path(f'{dest_path}/{dset_name}')
  if (not base_path.exists()) or force_download:
    print('files not found or forced, downloading')
    url = f'https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/raw/main/datasets/{dset_name}.tar.gz'
    file_name = f'{dest_path}/{dset_name}.tar.gz'

    download_tarfile(url, file_name, force_download=force_download)
    untar_file(file_name, dest_path)


def get_dataset(dataset_name, split, force_download=False):
  return genomic_clf_dset(dataset_name, split, force_download)


class demo_coding_vs_intergenomic_seqs_dset(genomic_clf_dset):
  def __init__(self, split, force_download):
    dset_name = 'demo_coding_vs_intergenomic_seqs'
    super().__init__(dset_name, split, force_download=force_download)

  def __len__(self):
    return super().__len__()

  def __getitem__(self, idx):
    return super().__getitem__(idx)


class demo_mouse_enhancers_dset(genomic_clf_dset):
  def __init__(self, split, force_download):
    dset_name = 'demo_mouse_enhancers'
    super().__init__(dset_name, split, force_download=force_download)

  def __len__(self):
    return super().__len__()

  def __getitem__(self, idx):
    return super().__getitem__(idx)