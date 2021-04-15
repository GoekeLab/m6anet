import os
import pandas as pd
import numpy as np
import torch
import json
import joblib
from ..scripts.compute_normalization_factors import annotate_kmer_information, create_kmer_mapping_df, create_norm_dict
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from itertools import product


class NanopolishDS(Dataset):

    def __init__(self, root_dir, min_reads, norm_path=None, site_info=None,
                 num_neighboring_features=1, mode='Inference', site_mode=False,
                 n_processes=1):
        allowed_mode = ('Train', 'Test', 'Val', 'Inference')
        
        if mode not in allowed_mode:
            raise ValueError("Invalid mode passed to dataset, must be one of {}".format(allowed_mode))
        
        self.mode = mode
        self.site_info = site_info
        self.data_info = self.initialize_data_info(root_dir, min_reads)
        self.data_fpath = os.path.join(root_dir, "data.json")
        self.min_reads = min_reads
        self.site_mode = site_mode

        if norm_path is not None:
            self.norm_dict = joblib.load(norm_path)
        else:
            self.norm_dict = self.compute_norm_factors(n_processes)

        if num_neighboring_features > 5:
            raise ValueError("Invalid neighboring features number {}".format(num_neighboring_features))

        self.num_neighboring_features = num_neighboring_features

        center_motifs = [['A', 'G', 'T'], ['G', 'A'], ['A'], ['C'], ['A', 'C', 'T']]
        flanking_motifs = [['G', 'A', 'C', 'T'] for i in range(self.num_neighboring_features)]
        all_kmers = list(["".join(x) for x in product(*(flanking_motifs + center_motifs + flanking_motifs))])
        
        self.all_kmers = np.unique(np.array(list(map(lambda x: [x[i:i+5] for i in range(len(x) -4)], 
                                            all_kmers))).flatten())
        self.kmer_to_int = {self.all_kmers[i]: i for i in range(len(self.all_kmers))}
        self.int_to_kmer =  {i: self.all_kmers[i] for i in range(len(self.all_kmers))}

        # Inferring total number of neighboring features extracted during dataprep step

        kmer, _ = self._load_data(0)
        self.total_neighboring_features = (len(kmer) - 5) // 2
        left_idx = [(self.total_neighboring_features - num_neighboring_features + j) * 3 + i 
                    for j in range(num_neighboring_features) for i in range(3)]
        center_idx = [self.total_neighboring_features * 3 + i for i in range(3)]
        right_idx = [(self.total_neighboring_features + j) * 3 + i for j in range(1, num_neighboring_features + 1) 
                        for i in range(3)]

        self.indices = np.concatenate([left_idx, center_idx, right_idx]).astype('int')

        if self.mode != 'Inference':
            self.labels = self.data_info["modification_status"].values


    def initialize_data_info(self, fpath, min_reads):
        data_index = pd.read_csv(os.path.join(fpath ,"data.index"))            
        if self.mode == 'Inference':
            read_count = pd.read_csv(os.path.join(fpath, "data.readcount"))
        else:
            if self.site_info is None:
                read_count = pd.read_csv(os.path.join(fpath, "data.readcount.labelled"))
            else:
                read_count = pd.read_csv(os.path.join(self.site_info, "data.readcount.labelled"))
            
            read_count = read_count[read_count["set_type"] == self.mode].reset_index(drop=True)

        data_info = data_index.merge(read_count, on=["transcript_id", "transcript_position"])
        return data_info[data_info["n_reads"] >= min_reads].reset_index(drop=True)

    def __len__(self):
        return len(self.data_info)

    def _load_data(self, idx):
        with open(self.data_fpath, 'r') as f:
            tx_id, tx_pos, start_pos, end_pos = self.data_info.iloc[idx][["transcript_id", "transcript_position",
                                                                          "start", "end"]]
            f.seek(start_pos, 0)
            json_str = f.read(end_pos - start_pos)
            pos_info = json.loads(json_str)[tx_id][str(tx_pos)]

            assert(len(pos_info.keys()) == 1)

            kmer, features = list(pos_info.items())[0]
        return kmer, np.array(features)

    def __getitem__(self, idx):
        kmer, features = self._load_data(idx)
        # Repeating kmer to the number of reads sampled
        kmer = self._retrieve_full_sequence(kmer, self.num_neighboring_features)
        kmer = [kmer[i:i+5] for i in range(2 * self.num_neighboring_features + 1)]

        features = features[np.random.choice(len(features), self.min_reads, replace=False), :]
        features = features[:, self.indices]
        
        if self.norm_dict is not None:
            mean, std = self.get_norm_factor(kmer)
            features = torch.Tensor((features - mean) / std)
        else:
            features = torch.Tensor((features))

        if not self.site_mode:
            kmer = np.repeat(np.array([self.kmer_to_int[kmer] for kmer in kmer])\
                        .reshape(-1, 2 * self.num_neighboring_features + 1), self.min_reads, axis=0)
            kmer = torch.Tensor(kmer)
        else:
            kmer = torch.LongTensor([self.kmer_to_int[kmer] for kmer in kmer])
        if self.mode == 'Inference':
            return features, kmer
        else:
            return features, kmer, self.data_info.iloc[idx]["modification_status"]

    def get_norm_factor(self, list_of_kmers):
        norm_mean, norm_std = [], []
        for kmer in list_of_kmers:
            mean, std = self.norm_dict[kmer]
            norm_mean.append(mean)
            norm_std.append(std)
        return np.concatenate(norm_mean), np.concatenate(norm_std)

    def compute_norm_factors(self, n_processes):
        if "kmer" not in self.data_info.columns:
            print("k-mer information is not present in column, annotating k-mer information in data info")
            self.data_info = annotate_kmer_information(self.data_fpath, self.data_info, n_processes)
        kmer_mapping_df = create_kmer_mapping_df(self.data_info)
        norm_dict = create_norm_dict(kmer_mapping_df, self.data_fpath, n_processes)
        return norm_dict

    def _retrieve_full_sequence(self, kmer, n_neighboring_features=0):
        if n_neighboring_features < self.total_neighboring_features:
            return kmer[self.total_neighboring_features - n_neighboring_features:2 * self.total_neighboring_features + n_neighboring_features]
        else:
            return kmer

    def _retrieve_sequence(self, sequence, n_neighboring_features=0):
        return [sequence[i : i+5] for i in range(len(sequence) - 4)]


class ImbalanceUnderSampler(torch.utils.data.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.class_counts = np.unique(self.data_source.labels, return_counts=True)[1]
        self.minority_class, self.majority_class = np.argmin(self.class_counts), np.argmax(self.class_counts)
        self.minority_class_idx = np.argwhere(self.data_source.labels == self.minority_class).flatten()
        self.majority_class_idx = np.argwhere(self.data_source.labels == self.majority_class).flatten()

    def __iter__(self):
        idx = np.append(self.minority_class_idx, np.random.choice(self.majority_class_idx,
                                                                  len(self.minority_class_idx), replace=False))
        np.random.shuffle(idx)
        return iter(idx)

    def __len__(self):
        return 2 * len(self.minority_class_idx)


class ImbalanceOverSampler(torch.utils.data.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.class_counts = np.unique(self.data_source.labels, return_counts=True)[1]
        self.minority_class, self.majority_class = np.argmin(self.class_counts), np.argmax(self.class_counts)
        self.minority_class_idx = np.argwhere(self.data_source.labels == self.minority_class).flatten()
        self.majority_class_idx = np.argwhere(self.data_source.labels == self.majority_class).flatten()

    def __iter__(self):
        idx = np.append(self.majority_class_idx, np.random.choice(self.minority_class_idx,
                                                                  len(self.majority_class_idx), replace=True))
        np.random.shuffle(idx)
        return iter(idx)

    def __len__(self):
        return 2 * len(self.majority_class_idx)


def inference_collate(batch):
    return {key: batch for key, batch 
            in zip (['X', 'kmer'], default_collate(batch))}


def train_collate(batch):
    return {key: batch for key, batch 
            in zip (['X', 'kmer', 'y'], default_collate(batch))}
