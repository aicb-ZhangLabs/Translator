import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import h5py
import time
from scipy.sparse import load_npz, vstack, csr_matrix

import torch
from torch.utils.data import Dataset, DataLoader

Downsample = 1000

    
class SimDataset(Dataset):
    def __init__(self, peak_name=None, lbl_name=None):
        super(SimDataset, self).__init__()
        file_name = peak_name
        type_name = lbl_name
        tic = time.time()
        self.data = load_npz(file_name)
        self.cell_label = list(pd.read_table(type_name, sep='\t', header=None)[1].values.flatten())
        self.size = self.data.shape[-1]
        self.conv = False
        tok = time.time()
        print(f"Finish loading in {tok-tic}, data size {self.data.shape}")
        self.padto = self.size
        self.depth_data = self.data.copy()
        self.d_mean = np.log(self.depth_data.sum(axis=1)).mean()
        self.d_std = np.log(self.depth_data.sum(axis=1)).std()

    def __getitem__(self, index):
        if self.conv:
            y = np.zeros(self.padto)
            x = np.array(self.data[index].todense()).flatten().astype(np.bool).astype(np.float)
            y[:x.shape[0]] = x
            return y, self.cell_label[index], np.sum(x)
        else:
            x = np.array(self.data[index].todense()).flatten().astype(np.bool).astype(np.float)
            return x, self.cell_label[index], np.sum(x)

    def __len__(self):
        return len(self.cell_label)

class SplitSimDataset(Dataset):
    def __init__(self, peak_name=None, lbl_name=None, peak_name2 = None, lbl_name2 = None):
        super(SplitSimDataset, self).__init__()
        
        tic = time.time()
        self.hd_dataset = SimDataset(peak_name, lbl_name)
        self.ld_dataset = SimDataset(peak_name2, lbl_name2)
        tok = time.time()
        print(f"Finish loading in {tok-tic}, data size {len(self.hd_dataset)}/{len(self.ld_dataset)}")
        self.padto = self.hd_dataset.data.shape[-1]
        self.hd_depth_data = self.hd_dataset.data.copy()
        self.ld_depth_data = self.ld_dataset.data.copy()
        self.d_mean = np.log(vstack([self.hd_depth_data.sum(axis=1), self.ld_depth_data.sum(axis=1)]).todense()).mean()
        self.d_std = np.log(vstack([self.hd_depth_data.sum(axis=1), self.ld_depth_data.sum(axis=1)]).todense()).std()

    def __getitem__(self, index):
        if index < len(self.hd_dataset):
            return self.hd_dataset[index]
        else:
            return self.ld_dataset[index - len(self.hd_dataset)]

    def __len__(self):
        return len(self.hd_dataset) + len(self.ld_dataset)
