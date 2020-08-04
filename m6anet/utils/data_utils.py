import h5py
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from itertools import product


class TrainDS(Dataset):

    def __init__(self, root_dir, mode, data_dir):
        self.data_dir = data_dir
        self.read_info = pd.read_csv(os.path.join(root_dir, mode, "data.csv.gz"))
        self.norm_constant = pd.read_csv(os.path.join(root_dir, "norm_constant.csv")).set_index("0")
        self.labels = self.read_info["modification_status"]
        self.sites = [os.path.join(data_dir, fname) for fname in self.read_info["fnames"].values]
        self.all_kmers = list(["".join(x) for x in product(['A', 'G', 'T'], ['G', 'A'], ['A'], ['C'], ['A', 'C', 'T'])])
        self.kmer_to_int = {self.all_kmers[i]: i for i in range(len(self.all_kmers))}
        self.int_to_kmer =  {i: self.all_kmers[i] for i in range(len(self.all_kmers))}
        self.kmers = np.array([self.kmer_to_int[x.split("_")[2]] for x in self.read_info["fnames"].values])

    def __len__(self):
        return len(self.sites)

    def __getitem__(self, idx):
        kmer = self.kmers[idx]
        norm_info = self.norm_constant.loc[self.int_to_kmer[kmer]].values
        mean, std = norm_info[:3], norm_info[3:]
        if torch.is_tensor(idx):
            idx = idx.tolist()
        f = h5py.File(self.sites[idx], 'r')
        X = (f['X'][:] - mean) / std
        label = self.labels[idx]
        n_reads = len(X)
        f.close()
        return (torch.Tensor(X),
                torch.LongTensor([kmer]).repeat(len(X)),
                torch.Tensor([label]),
                n_reads)


class ValDS(Dataset):

    def __init__(self, norm_constant, data_dir, sites=None):
        self.data_dir = data_dir
        self.norm_constant = norm_constant.set_index("0")
        
        self.all_kmers = list(["".join(x) for x in product(['A', 'G', 'T'], ['G', 'A'], ['A'], ['C'], ['A', 'C', 'T'])])
        self.kmer_to_int = {self.all_kmers[i]: i for i in range(len(self.all_kmers))}
        self.int_to_kmer =  {i: self.all_kmers[i] for i in range(len(self.all_kmers))}
        
        if sites is None:
            all_files = os.listdir(data_dir)
            self.sites = np.array([os.path.join(data_dir, fname) for fname in all_files])
            self.kmers = np.array([self.kmer_to_int[x.split("_")[2]] for x in all_files])

        else:
            self.sites = np.array([os.path.join(data_dir, fname) for fname in sites])
            self.kmers = np.array([self.kmer_to_int[x.split("_")[2]] for x in sites])
            
            
    def __len__(self):
        return len(self.sites)

    def __getitem__(self, idx):
        kmer = self.kmers[idx]
        norm_info = self.norm_constant.loc[self.int_to_kmer[kmer]].values
        mean, std = norm_info[:3], norm_info[3:]
        if torch.is_tensor(idx):
            idx = idx.tolist()
        f = h5py.File(self.sites[idx], 'r')
        X = (f['X'][:] - mean) / std
        n_reads = len(X)
        f.close()
        return (torch.Tensor(X),
                torch.LongTensor([kmer]).repeat(len(X)),
                n_reads)


def assign_group_to_index(batch):
    curr_idx = 0
    idx_per_group = []
    for i in range(len(batch)):
        num_reads = batch[i][3]
        idx_per_group.append(np.arange(curr_idx, curr_idx + num_reads))
        curr_idx += num_reads
    return np.array(idx_per_group)


def assign_group_to_index_val(batch):
    curr_idx = 0
    idx_per_group = []
    for i in range(len(batch)):
        num_reads = batch[i][2]
        idx_per_group.append(np.arange(curr_idx, curr_idx + num_reads))
        curr_idx += num_reads
    return np.array(idx_per_group)


def custom_collate(batch):
    return (torch.cat([item[0] for item in batch]),
            torch.cat([item[1] for item in batch]),
            assign_group_to_index(batch),
            torch.cat([item[2] for item in batch])
            )


def custom_collate_val(batch):
    return (torch.cat([item[0] for item in batch]),
            torch.cat([item[1] for item in batch]),
            assign_group_to_index_val(batch))
