from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from pathlib import Path


class dummy_dset(Dataset):
  def __init__(self):
    self.df = pd.DataFrame()
    for i in range(1000):
        self.df = self.df.append({'column1':i, 'column2':np.random.randint(low=0, high=10, size=100)}, ignore_index=True)

  def __len__(self):
    return self.df.size

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    #TODO parametrize to.cuda?
    x = torch.tensor(row['column2']).to('cuda')
    y = torch.tensor(row['column1']).to('cuda')
    return x,y


class cvsi_dset(Dataset): #TODO inherit mapstyledataset? https://pytorch.org/docs/stable/data.html#dataset-types
    def __init__(self, split):
        #TODO parametrize path
        if(split == 'train'):
          c_path = Path('./datasets/demo_coding_vs_intergenomic_seqs/train/coding_seqs.csv')
          i_path = Path('./datasets/demo_coding_vs_intergenomic_seqs/train/intergenomic_seqs.csv')

        elif(split == 'test'):
          c_path = Path('./datasets/demo_coding_vs_intergenomic_seqs/test/coding_seqs.csv')
          i_path = Path('./datasets/demo_coding_vs_intergenomic_seqs/test/intergenomic_seqs.csv')
        else:
            raise ValueError("Incorrect value of split argument")


        coding_df = pd.read_csv(c_path)
        intergenomic_df = pd.read_csv(i_path)
        # path = Path('../datasets/demo_coding_vs_intergenomic_seqs/train/coding_seqs.csv')
        self.df = pd.concat([coding_df, intergenomic_df])
        # print(len(coding_df))
        # print(len(intergenomic_df))
        # print(len(self.df))


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        start = row['start']
        end = row['end']
        length = end-start
        # length = 100
        # print('L:',length)
        #TODO resolve different lengths with optional padding?
        # dummy_seq = np.random.randint(low=0, high=4, size=length)
        dummy_seq = ''.join(np.random.choice(['A','C','T','G'], size=length))

        dummy_label = np.random.randint(0, 2, size=1)[0]
        # x = torch.tensor(dummy_seq).to('cuda')
        x = dummy_seq

        # y = torch.tensor(dummy_label, dtype=torch.float32).to('cuda')
        y = dummy_label
        return x,y

class cvsi_dset_translated(Dataset): #TODO inherit mapstyledataset? https://pytorch.org/docs/stable/data.html#dataset-types
    def __init__(self, split):
        #TODO check downloaded files
        base_path = Path('/home/martinekvlastimil95/.genomic_benchmarks/demo_coding_vs_intergenomic_seqs')
        if(split == 'train'):
          base_path = base_path/'train'
        elif(split == 'test'):
          base_path = base_path/'test'
        else:
          raise Exception('Define split, train or test')

        self.all_paths = []
        self.all_labels = []
        label_mapper = {
          'coding':0,
          'intergenomic':1,
        }
        for x in (base_path/'coding_seqs').iterdir():
            self.all_paths.append(x)
            self.all_labels.append(label_mapper['coding'])
        for x in (base_path/'intergenomic_seqs').iterdir():
            self.all_paths.append(x)
            self.all_labels.append(label_mapper['intergenomic'])

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        txt_path = self.all_paths[idx]
        with open(txt_path, 'r') as f:
          content = f.read()
        x = content
        y = self.all_labels[idx]
        return x,y