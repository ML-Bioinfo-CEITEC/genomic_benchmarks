from torch.utils.data import Dataset
from pathlib import Path
import genomic_benchmarks
from genomic_benchmarks.loc2seq.with_biopython import download_dataset
from genomic_benchmarks.utils.data_utils import extract_gzip
import os

class genomic_clf_dset(Dataset):
    def __init__(self, dset_name, split, force_download=False):
        translated_datasets_folder_path = Path.home() / '.genomic_benchmarks'
        #TODO resolve path some other way?
        interval_datasets_folder_path = Path(os.path.dirname(genomic_benchmarks.__file__)).parent.parent/'datasets'
        fasta_cache_path = translated_datasets_folder_path / 'fasta'
        dset_path = Path(f'{interval_datasets_folder_path}/{dset_name}')

        for gz_file in dset_path.rglob('*.gz'):
          csv_path = gz_file.parent/(gz_file.stem)
          if not csv_path.exists():
            extract_gzip(gz_file, csv_path)

        destination_path = Path(f'{translated_datasets_folder_path}/translated_{dset_name}')
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

def get_dataset(dataset_name, split, force_download=False):
  return genomic_clf_dset(dataset_name, split, force_download)


class demo_coding_vs_intergenomic_seqs_dset(genomic_clf_dset):
  def __init__(self, split, force_download=False):
    dset_name = 'demo_coding_vs_intergenomic_seqs'
    super().__init__(dset_name, split, force_download=force_download)

  def __len__(self):
    return super().__len__()

  def __getitem__(self, idx):
    return super().__getitem__(idx)


class demo_mouse_enhancers_dset(genomic_clf_dset):
  def __init__(self, split, force_download=False):
    dset_name = 'demo_mouse_enhancers'
    super().__init__(dset_name, split, force_download=force_download)

  def __len__(self):
    return super().__len__()

  def __getitem__(self, idx):
    return super().__getitem__(idx)
