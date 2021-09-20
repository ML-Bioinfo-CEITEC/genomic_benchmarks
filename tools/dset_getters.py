from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from enum import Enum
from pathlib import Path

class Datatype(Enum):
    TRAIN = 1
    TEST = 2
    # VALID = 3

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


class cvsi_dset(Dataset):
    def __init__(self, split):
        #TODO parametrize path
        if(split == 'train'):
          c_path = Path('./datasets/demo_coding_vs_intergenomic_seqs/train/coding_seqs.csv')
          i_path = Path('./datasets/demo_coding_vs_intergenomic_seqs/train/intergenomic_seqs.csv')

        if(split == 'test'):
          c_path = Path('./datasets/demo_coding_vs_intergenomic_seqs/test/coding_seqs.csv')
          i_path = Path('./datasets/demo_coding_vs_intergenomic_seqs/test/intergenomic_seqs.csv')


        coding_df = pd.read_csv(c_path)
        intergenomic_df = pd.read_csv(i_path)
        # path = Path('../datasets/demo_coding_vs_intergenomic_seqs/train/coding_seqs.csv')
        self.df = pd.concat([coding_df, intergenomic_df])
        # print(len(coding_df))
        # print(len(intergenomic_df))
        # print(len(self.df))

    def __len__(self):
        return self.df.size

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        start = row['start']
        end = row['end']
        length = end-start
        # length = 100
        print('L:',length)
        dummy_seq = np.random.randint(low=0, high=4, size=length)
        dummy_label = 0
        x = torch.tensor(dummy_seq, dtype=torch.float32).to('cuda')
        y = torch.tensor(dummy_label, dtype=torch.float32).to('cuda')
        return x,y
