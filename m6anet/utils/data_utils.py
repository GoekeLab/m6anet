import os
import pandas as pd
import numpy as np
import torch
import json
import joblib
from ..scripts.constants import NUM_NEIGHBORING_FEATURES, KMER_TO_INT
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from itertools import product


class NanopolishDS(Dataset):

    def __init__(self, root_dir, min_reads, norm_path):
        self.data_info = self.initialize_data_info(root_dir, min_reads)
        self.data_dir = os.path.join(root_dir, "data.json")
        self.min_reads = min_reads
        self.norm_dict = joblib.load(norm_path)

    def initialize_data_info(self, fpath, min_reads):
        data_index = pd.read_csv(os.path.join(fpath ,"data.index"))
        read_count = pd.read_csv(os.path.join(fpath, "data.readcount"))
        data_info = data_index.merge(read_count, on=["transcript_id", "transcript_position"])
        return data_info[data_info["n_reads"] >= min_reads].reset_index(drop=True)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        with open(self.data_dir, 'r') as f:
            tx_id, tx_pos, start_pos, end_pos = self.data_info.iloc[idx][["transcript_id", "transcript_position",
                                                                        "start", "end"]]
            f.seek(start_pos, 0)
            json_str = f.read(end_pos - start_pos)
            pos_info = json.loads(json_str)[tx_id][str(tx_pos)]

            assert(len(pos_info.keys()) == 1)

            kmer, features = list(pos_info.items())[0]
            
            # Repeating kmer to the number of reads sampled
            kmer = [kmer[i:i+5] for i in range(2 * NUM_NEIGHBORING_FEATURES + 1)]
            mean, std = self.get_norm_factor(kmer)
            kmer = np.repeat(np.array([KMER_TO_INT[kmer] for kmer in kmer])\
                        .reshape(-1, 2 * NUM_NEIGHBORING_FEATURES + 1), self.min_reads, axis=0)
            kmer = torch.Tensor(kmer)

            features = np.array(features)
            features = features[np.random.choice(len(features), self.min_reads, replace=False), :]
            features = torch.Tensor((features - mean) / std)
            return features, kmer

    def get_norm_factor(self, list_of_kmers):
        norm_mean, norm_std = [], []
        for kmer in list_of_kmers:
            mean, std = self.norm_dict[kmer]
            norm_mean.append(mean)
            norm_std.append(std)
        return np.concatenate(norm_mean, axis=1), np.concatenate(norm_std, axis=1)

def kmer_collate(batch):
    return {key: batch for key, batch 
            in zip (['X', 'kmer'], default_collate(batch))}
