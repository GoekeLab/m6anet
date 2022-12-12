import os
import pandas as pd
import numpy as np
import torch
import json
import joblib
from ast import literal_eval
from ..scripts.compute_normalization_factors import annotate_kmer_information, create_kmer_mapping_df, create_norm_dict, read_kmer, get_mean_std_replicates
from multiprocessing import Pool
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from itertools import product


class NanopolishDS(Dataset):

    def __init__(self, root_dir=None, min_reads=20, norm_path=None, site_info=None,
                 num_neighboring_features=1, mode='Inference',
                 n_processes=1, data_info=None):
        allowed_mode = ('Train', 'Test', 'Val', 'Inference')

        if mode not in allowed_mode:
            raise ValueError("Invalid mode passed to dataset, must be one of {}".format(allowed_mode))

        if (root_dir is None) and (data_info is None):
            raise ValueError("Either root directory or data info must be given")

        self.mode = mode
        self.site_info = site_info
        self.min_reads = min_reads

        if data_info is None:
            self.initialize_data_info(root_dir, min_reads)
        else:
            self.prepare_data_info(data_info)

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

        self.set_feature_indices()

        if self.mode != 'Inference':
            self.labels = self.data_info["modification_status"].values
            self.inference = False
        else:
            self.inference = True

    def set_feature_indices(self):
        self.total_neighboring_features = self.get_total_neighboring_features()
        left_idx = [(self.total_neighboring_features - self.num_neighboring_features + j) * 3 + i
                    for j in range(self.num_neighboring_features) for i in range(3)]
        center_idx = [self.total_neighboring_features * 3 + i for i in range(3)]
        right_idx = [(self.total_neighboring_features + j) * 3 + i
                     for j in range(1, self.num_neighboring_features + 1)
                     for i in range(3)]
        self.indices = np.concatenate([left_idx, center_idx, right_idx]).astype('int')

    def prepare_data_info(self, data_info):

        fpath = os.path.dirname(data_info)
        self.data_fpath = os.path.join(fpath, "data.json")
        data_info = pd.read_csv(data_info)
        if self.mode != 'Inference':
            data_info = data_info[data_info["set_type"] == self.mode].reset_index(drop=True)

        data_index = pd.read_csv(os.path.join(fpath ,"data.index"))
        data_info = data_index.merge(data_info, on=["transcript_id", "transcript_position"])
        self.data_info = data_info

    def initialize_data_info(self, fpath, min_reads):

        if self.mode == 'Inference':
            data_info = pd.read_csv(os.path.join(fpath, "data.info"))
        else:
            if self.site_info is None:
                data_info = pd.read_csv(os.path.join(fpath, "data.info.labelled"))
            else:
                data_info = pd.read_csv(os.path.join(self.site_info, "data.info.labelled"))

            data_info = data_info[data_info["set_type"] == self.mode].reset_index(drop=True)

        self.data_fpath = os.path.join(fpath, "data.json")
        self.data_info = data_info[data_info["n_reads"] >= min_reads].reset_index(drop=True)

    def __len__(self):
        return len(self.data_info)

    def get_total_neighboring_features(self):
        tx_id, tx_pos, start_pos, end_pos = self.data_info.iloc[0][["transcript_id", "transcript_position",
                                                                      "start", "end"]]
        kmer, _ = self._load_data(self.data_fpath, tx_id, tx_pos, start_pos, end_pos)
        return (len(kmer) - 5) // 2

    def load_data(self, idx):
        tx_id, tx_pos, start_pos, end_pos = self.data_info.iloc[idx][["transcript_id", "transcript_position",
                                                                      "start", "end"]]
        kmer, features = self._load_data(self.data_fpath, tx_id, tx_pos, start_pos, end_pos)

        read_ids, features = features[:, -1], features[:, self.indices]
        return tx_id, tx_pos, read_ids, features, kmer

    def _load_data(self, fpath, tx_id, tx_pos, start_pos, end_pos):
        with open(fpath, 'r') as f:
            f.seek(start_pos, 0)
            json_str = f.read(end_pos - start_pos)
            pos_info = json.loads(json_str)[tx_id][str(tx_pos)]

            assert(len(pos_info.keys()) == 1)

            kmer, features = list(pos_info.items())[0]
        return kmer, np.array(features)

    def __getitem__(self, idx):
        tx_id, tx_pos, read_ids, features, kmer = self.load_data(idx)

        kmer = self._retrieve_full_sequence(kmer, self.num_neighboring_features)
        kmer = [kmer[i:i+5] for i in range(2 * self.num_neighboring_features + 1)]

        if not self.inference:
            features = features[np.random.choice(len(features), self.min_reads, replace=False), :]

        if self.norm_dict is not None:
            mean, std = self.get_norm_factor(kmer)
            features = torch.Tensor((features - mean) / std)
        else:
            features = torch.Tensor((features))

        # Repeating kmer to the number of reads
        kmer = torch.LongTensor(np.repeat(np.array([self.kmer_to_int[kmer] for kmer in kmer])\
                    .reshape(-1, 2 * self.num_neighboring_features + 1), len(features), axis=0))

        if self.inference:
            tx_id = np.repeat(tx_id, len(features))
            tx_pos = np.repeat(tx_pos, len(features))
            return features, kmer, tx_id, tx_pos, read_ids
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

    def _retrieve_sequence(self, sequence):
        return [sequence[i : i+5] for i in range(len(sequence) - 4)]


class NanopolishReplicateDS(NanopolishDS):

    def __init__(self, root_dirs=None, min_reads=20, norm_path=None, site_info=None,
                 num_neighboring_features=1, mode='Inference',
                 data_info=None,
                 n_processes=1):
        super().__init__(root_dirs, min_reads, norm_path, site_info,
                         num_neighboring_features, mode,
                         n_processes, data_info)

    def prepare_data_info(self, data_info):
        data_info = pd.read_csv(data_info)
        data_info["coords"] = data_info["coords"].apply(lambda x: literal_eval(x))
        data_info["fpath"] = data_info["fpath"].apply(lambda x: [y.replace("'", '').strip(" ") for y in x[1:-1].split(",")])
        data_info["n_reads"] = data_info["n_reads"].astype('int')
        data_info = data_info[data_info["n_reads"] >= self.min_reads].reset_index(drop=True)

        if not self.inference:
            data_info = data_info[data_info["set_type"] == self.mode].reset_index(drop=True)

        self.data_info = data_info


    def initialize_data_info(self, root_dirs, min_reads):
        all_read_info = [pd.read_csv(os.path.join(root_dir, "data.info"))\
                            .assign(fpath=root_dir)\
                            .set_index(["transcript_id", "transcript_position"])
                        for root_dir in root_dirs]
        read_info = pd.concat(all_read_info, axis=1)
        total_reads = read_info["n_reads"].sum(axis=1).reset_index(drop=True)
        start = read_info["start"].apply(lambda x: [int(col) for col in x if col == col], axis=1)
        end = read_info["end"].apply(lambda x: [int(col) for col in x if col == col], axis=1)
        fpath = read_info["fpath"].apply(lambda x: [col for col in x if col == col], axis=1).reset_index(drop=True)
        coord = pd.concat([start, end], axis=1).apply(lambda x: [(x, y) for x, y in zip(x[0],x[1])], axis=1)\
            .reset_index(drop=True)
        read_info = read_info.reset_index()[["transcript_id", "transcript_position"]]
        read_info["n_reads"] = total_reads.astype('int')
        read_info["coords"] = coord
        read_info["fpath"] = fpath
        self.data_info = read_info[read_info["n_reads"] >= min_reads].reset_index(drop=True)
        self.fpath_mapping = {root_dir: num_rep for num_rep, root_dir in enumerate(root_dirs)}

    def get_total_neighboring_features(self):
        tx_id, tx_pos, coords, fpaths = self.data_info.iloc[0][["transcript_id", "transcript_position",
                                                               "coords", "fpath"]]
        start_pos, end_pos = coords[0]
        fpath = os.path.join(fpaths[0], "data.json")

        kmer, _ = self._load_data(fpath, tx_id, tx_pos, start_pos, end_pos)
        return (len(kmer) - 5) // 2

    def load_data(self, idx):
        tx_id, tx_pos, coords, fpaths = self.data_info.iloc[idx][["transcript_id", "transcript_position",
                                                                  "coords", "fpath"]]
        all_features = []
        all_read_ids = []
        all_kmer = None
        for coord, fpath in zip(coords, fpaths):
            start_pos, end_pos = coord
            read_suffix = self.fpath_mapping[fpath]
            fpath = os.path.join(fpath, "data.json")
            kmer, features = self._load_data(fpath, tx_id, tx_pos, start_pos, end_pos)

            if all_kmer is None:
                all_kmer = kmer
            else:
                assert(all_kmer == kmer)

            read_ids, features = features[:, -1].astype('int'), features[:, self.indices]
            all_features.append(features)
            all_read_ids.append(["{}_{}".format(read_id, read_suffix) for read_id in read_ids])

        all_features = np.concatenate(all_features)
        all_read_ids = np.concatenate(all_read_ids)
        return tx_id, tx_pos, all_read_ids, all_features, all_kmer

    def compute_norm_factors(self, n_processes):
        kmer_mapping_df = self.annotate_kmer_information(self.data_info, n_processes)
        kmer_mapping_df = self.create_kmer_mapping_df(kmer_mapping_df)
        norm_dict = self.create_norm_dict(kmer_mapping_df, n_processes)
        return norm_dict

    def annotate_kmer_information(self, index_df, n_processes):
        tasks = ((os.path.join(fpaths[0], "data.json"), tx_id, tx_pos, coords[0][0], coords[0][1]) for fpaths, tx_id, tx_pos, coords in
                    zip(index_df["fpath"], index_df["transcript_id"], index_df["transcript_position"], index_df["coords"]))

        with Pool(n_processes) as p:
            kmer_info = [x for x in tqdm(p.imap(read_kmer, tasks), total=len(index_df))]
        index_df["kmer"] = kmer_info
        return index_df

    def create_kmer_mapping_df(self, merged_df):
        kmer_mapping_df = []
        for _, row in merged_df.iterrows():
            tx, tx_position, sequence, coords, fpaths, n_reads = row["transcript_id"], \
                    row["transcript_position"], row["kmer"], row["coords"], row["fpath"], row["n_reads"]
            kmers = [sequence[i:i+5] for i in range(len(sequence) - 4)]
            for i in range(len(kmers)):
                kmer_mapping_df += [(tx, tx_position, coords, fpaths, n_reads, kmers[i], i)]
        kmer_mapping_df = pd.DataFrame(kmer_mapping_df,
                                    columns=["transcript_id", "transcript_position", "coords",
                                             "fpath", "n_reads", "kmer", "segment_number"])
        return kmer_mapping_df

    def create_norm_dict(self, kmer_mapping_df, n_processes):
        tasks = ((kmer, df) for kmer, df in kmer_mapping_df.groupby("kmer"))
        with Pool(n_processes) as p:
            norm_dict = [x for x in tqdm(p.imap_unordered(get_mean_std_replicates, tasks),
                                         total=len(kmer_mapping_df["kmer"].unique()))]
        return {tup[0]: (tup[1], tup[2]) for tup in norm_dict}


def inference_collate(batch):
    n_reads = torch.LongTensor([len(item[0]) for item in batch])
    features = torch.cat([item[0] for item in batch])
    kmers = torch.cat([item[1] for item in batch])
    tx_ids = np.concatenate([item[2] for item in batch])
    tx_pos = np.concatenate([item[3] for item in batch])
    read_ids = np.concatenate([item[4] for item in batch])

    return features, kmers, n_reads, tx_ids, tx_pos, read_ids


def train_collate(batch):
    return {key: batch for key, batch
            in zip (['X', 'kmer', 'y'], default_collate(batch))}
