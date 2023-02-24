r"""
This module contains PyTorch dataset classes used during training and inference to load nanopolish-preprocessed data
"""
import os
import pandas as pd
import numpy as np
import torch
import json
import joblib
from ast import literal_eval
from .norm_utils import annotate_kmer_information, create_kmer_mapping_df, create_norm_dict, read_kmer, get_mean_std_replicates
from multiprocessing import Pool
from typing import Dict, List, Tuple, Union, Optional
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from itertools import product


class NanopolishDS(Dataset):
    r"""
    A PyTorch Dataset class for a single dataprep-preprocessed nanopolish eventalign.txt file. This class is used
    for both training and inference, allowing indexed access of features of RNA reads from each transcriptomic site

    ...

    Attributes
    -----------
    allowed_mode (tuple[str]): A class attribute containing allowed mode of initialization
    root_dir (str): A filepath to directory containing data.json and data.info or data.info.labelled file preprocessed using m6anet dataprep functionality
    min_reads (int): Minimum number of reads for a site to be considered for inference
    mode (str): The running mode for the dataset instance, must be one of Train, Val, or Test
    data_info (pd.DataFrame): pd.Dataframe object containing indexing information of data.json for faster file access and the number of reads for each DRACH positions in eventalign.txt
    norm_dict (Dict): Dictionary containing normalization factors for each 5-mer motifs
    num_neighboring_features (int): Number of flanking positions around the target site involved during features extraction
    all_kmers (list[str]): List containing unique motifs from all positions and flanking positions considered during features extraction
    kmer_to_int (Dict): A dictionary containing mapping between 5-mer motifs and their assigned integers
    int_to_kmer (Dict): A dictionary containing mapping between integers and their assigned 5-mer motifs
    indices (np.array): A NumPy array containing information of indices to be extracted from features in data.json preprocessed file from m6anet dataprep functionality
    """

    allowed_mode = ('Train', 'Test', 'Val', 'Inference')

    def __init__(self, root_dir: str,
                 min_reads: Optional[int] = 20,
                 norm_path: Optional[str] = None,
                 num_neighboring_features: Optional[int] =1,
                 mode: Optional[str] = 'Inference',
                 n_processes: Optional[int] = 1):
        r'''
        Initialization function for the class

                Args:
                        root_dir (str): A filepath to directory containing data.json and data.info or data.info.labelled file preprocessed using m6anet dataprep functionality
                        min_reads (int): Minimum number of reads for a site to be considered for inference
                        norm_path (str): A filepath to a joblib object containing normalization factors
                        num_neighboring_features (int): Number of flanking positions around the target site involved during features extraction
                        mode (str): The running mode for the dataset instance, must be one of Train, Val, or Test
                        n_processes (int): Number of processes used to compute normalization factors if needed

                Returns:
                        None

                Raises:
                    ValueError: Raises an exception when some of the listed arguments do not follow the allowed conventions
        '''
        if mode not in self.allowed_mode:
            raise ValueError("Invalid mode passed to dataset, must be one of {}".format(self.allowed_mode))

        if root_dir is None:
            raise ValueError("Either root directory or data info must be given")

        self.root_dir = root_dir
        self.min_reads = min_reads
        self.mode = mode

        self.initialize_data_info()

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

    def set_feature_indices(self):
        r'''
        Instance method for the class to initialize indices corresponding to features to be extracted from data.json file
        '''
        self.total_neighboring_features = self.get_total_neighboring_features()
        left_idx = [(self.total_neighboring_features - self.num_neighboring_features + j) * 3 + i
                    for j in range(self.num_neighboring_features) for i in range(3)]
        center_idx = [self.total_neighboring_features * 3 + i for i in range(3)]
        right_idx = [(self.total_neighboring_features + j) * 3 + i
                     for j in range(1, self.num_neighboring_features + 1)
                     for i in range(3)]
        self.indices = np.concatenate([left_idx, center_idx, right_idx]).astype('int')

    def initialize_data_info(self):
        r'''
        Instance method for the class to initialize indices corresponding to features to be extracted from data.json file
        '''
        if self.mode == 'Inference':
            data_info = pd.read_csv(os.path.join(self.root_dir, "data.info"))
        else:
            data_info = pd.read_csv(os.path.join(self.root_dir, "data.info.labelled"))
            data_info = data_info[data_info["set_type"] == self.mode].reset_index(drop=True)

        self.data_fpath = os.path.join(self.root_dir, "data.json")
        self.data_info = data_info[data_info["n_reads"] >= self.min_reads].reset_index(drop=True)

    def __len__(self) -> int:
        r'''
        Number of sites processed by the instance of this class
        '''
        return len(self.data_info)

    def get_total_neighboring_features(self) -> int:
        r'''
        Instance method for the class to infer the total number of neighboring positions present in data.json

                Args:
                        None

                Returns:
                        Integer representing the total number of neighboring positions in data.json
        '''
        tx_id, tx_pos, start_pos, end_pos = self.data_info.iloc[0][["transcript_id", "transcript_position",
                                                                      "start", "end"]]
        kmer, _ = self._load_data(self.data_fpath, tx_id, tx_pos, start_pos, end_pos)
        return (len(kmer) - 5) // 2

    def load_data(self, idx: int) -> Tuple[str, int, np.ndarray, np.ndarray, str]:
        r'''
        Instance method for the class to load features from reads belonging to the idx-th site in data_info attribute
                Args:
                        idx (int): Integer representing the index position of the sites to load

                Returns:
                        tuple(int, int, np.ndarray, np.ndarray, str): Tuple containing transcript_id information, transcript_position information, read_ids, features,
                                                                      and 5-mer motifs information from reads belonging to the idx-th site in data info
        '''
        tx_id, tx_pos, start_pos, end_pos = self.data_info.iloc[idx][["transcript_id", "transcript_position",
                                                                      "start", "end"]]
        kmer, features = self._load_data(self.data_fpath, tx_id, tx_pos, start_pos, end_pos)

        read_ids, features = features[:, -1], features[:, self.indices]
        return tx_id, tx_pos, read_ids, features, kmer

    def _load_data(self, data_fpath: str, tx_id: str, tx_pos: int, start_pos: int, end_pos: int) -> Tuple[str, np.ndarray]:
        r'''
        Instance method for the class to read features from specified positions within data.json
                Args:
                        data_fpath (str): Filepath to data.json
                        tx_id (str): Transcript id information of the query position
                        tx_pos (int): Transcript position information of the query position
                        start_pos (int): Starting index corresponding to the query position within data.json
                        end_pos (int): Ending index corresponding to the query position within data.json

                Returns:
                        tuple(str, np.ndarray): Tuple containing the 5-mer motif and numpy ndarray of the features from reads belonging to the position
        '''
        with open(data_fpath, 'r', encoding='utf-8') as f:
            f.seek(start_pos, 0)
            json_str = f.read(end_pos - start_pos)
            pos_info = json.loads(json_str)[tx_id][str(tx_pos)]

            assert(len(pos_info.keys()) == 1)

            kmer, features = list(pos_info.items())[0]
        return kmer, np.array(features)

    def __getitem__(self, idx: int) -> Union[Tuple[np.ndarray, str, str, int, np.ndarray],
                                             Tuple[np.ndarray, str, int]]:
        r'''
        Instance method to access features from reads belonging to the idx-th site in data_info attribute
                Args:
                        idx (int): Integer representing the index position of the sites to load

                Returns:
                        inference:
                            tuple(np.ndarray, str, str, int, np.ndarray): Tuple containing features information, 5-mer motifs, transcript_id, transcript_position and read_ids
                                                                          from reads belonging to the idx-th site in data info during inference
                        Training:
                            tuple(np.ndarray, str, int): Tuple containing features information, 5-mer motifs, and modification status from reads belonging to
                                                         the idx-th site in data info during training

        '''
        tx_id, tx_pos, read_ids, features, kmer = self.load_data(idx)

        kmer = self._retrieve_full_sequence(kmer, self.num_neighboring_features)
        kmer = [kmer[i:i+5] for i in range(2 * self.num_neighboring_features + 1)]

        if self.mode != 'Inference':
            features = features[np.random.choice(len(features), self.min_reads, replace=False), :]

        if self.norm_dict is not None:
            mean, std = self.get_norm_factor(kmer)
            features = torch.Tensor((features - mean) / std)
        else:
            features = torch.Tensor((features))

        # Repeating kmer to the number of reads
        kmer = torch.LongTensor(np.repeat(np.array([self.kmer_to_int[kmer] for kmer in kmer])\
                    .reshape(-1, 2 * self.num_neighboring_features + 1), len(features), axis=0))

        if self.mode == 'Inference':
            tx_id = np.repeat(tx_id, len(features))
            tx_pos = np.repeat(tx_pos, len(features))
            return features, kmer, tx_id, tx_pos, read_ids
        else:
            return features, kmer, self.data_info.iloc[idx]["modification_status"]

    def get_norm_factor(self, list_of_kmers) -> Tuple[np.ndarray, np.ndarray]:
        r'''
        Instance method to retrieve normalization factors corresponding to the provided list of 5-mer motifs
                Args:
                        list_of_kmers (list[str]): List of string 5-mer motifs to query the normalization factors of

                Returns:
                        tuple(np.ndarray, str, str, int, np.ndarray): Tuple containing features information, 5-mer motifs, transcript_id, transcript_position and read_ids
                                                                          from reads belonging to the idx-th site in data info during inference
        '''
        norm_mean, norm_std = [], []
        for kmer in list_of_kmers:
            mean, std = self.norm_dict[kmer]
            norm_mean.append(mean)
            norm_std.append(std)
        return np.concatenate(norm_mean), np.concatenate(norm_std)

    def compute_norm_factors(self, n_processes:int) -> Dict:
        r'''
        Instance method to compute normalization factors for each 5-mer motif present in the data
                Args:
                        n_processes (int): Number of CPU processes to be allocated for th processing

                Returns:
                        norm_dict (dict): Dictionary containing normalization factors for each 5-mer motif present in the data
        '''
        if "kmer" not in self.data_info.columns:
            print("k-mer information is not present in column, annotating k-mer information in data info")
            self.data_info = annotate_kmer_information(self.data_fpath, self.data_info, n_processes)
        kmer_mapping_df = create_kmer_mapping_df(self.data_info)
        norm_dict = create_norm_dict(kmer_mapping_df, self.data_fpath, n_processes)
        return norm_dict

    def _retrieve_full_sequence(self, kmer: str, n_neighboring_features: Optional[int] = 1) -> str:
        r'''
        Instance method to retrieve sequence information corresponding to the specified number of neighboring features from data.json sequence information
                Args:
                        kmer (str): Sequence information from data.json
                        n_neighboring_features (int): Number of neighboring flanking positions to extract the sequence information from

                Returns:
                        kmer (str): Sequence information corresponding to the flanking region of the middle nucleotide specified in the kmer input
        '''
        if n_neighboring_features < self.total_neighboring_features:
            return kmer[self.total_neighboring_features - n_neighboring_features:2 * self.total_neighboring_features + n_neighboring_features]
        else:
            return kmer

    def _retrieve_sequence(self, sequence: str) -> List[str]:
        r'''
        Instance method to convert a nucleotide sequence into a set of consecutive 5-mer motifs
                Args:
                        sequence (str): Sequence information corresponding to the neighboring flanking positions of its middle nucleotide

                Returns:
                        list[str]: A list containing 5-mer motifs corresponding to the flanking positions of the original input
        '''
        return [sequence[i : i+5] for i in range(len(sequence) - 4)]


class NanopolishReplicateDS(NanopolishDS):
    r"""
    A PyTorch Dataset class for multiple dataprep-preprocessed nanopolish eventalign.txt files. This class is used
    for both training and inference, allowing indexed access of features of RNA reads from each transcriptomic site

    ...

    Attributes
    -----------
    allowed_mode (tuple[str]): A class attribute containing allowed mode of initialization
    root_dir (str): A filepath to directory containing data.json and data.info or data.info.labelled file preprocessed using m6anet dataprep functionality
    min_reads (int): Minimum number of reads for a site to be considered for inference
    mode (str): The running mode for the dataset instance, must be one of Train, Val, or Test
    data_info (pd.DataFrame): pd.Dataframe object containing indexing information of data.json for faster file access and the number of reads for each DRACH positions in eventalign.txt
    norm_dict (Dict): Dictionary containing normalization factors for each 5-mer motifs
    num_neighboring_features (int): Number of flanking positions around the target site involved during features extraction
    all_kmers (list[str]): List containing unique motifs from all positions and flanking positions considered during features extraction
    kmer_to_int (Dict): A dictionary containing mapping between 5-mer motifs and their assigned integers
    int_to_kmer (Dict): A dictionary containing mapping between integers and their assigned 5-mer motifs
    indices (np.array): A NumPy array containing information of indices to be extracted from features in data.json preprocessed file from m6anet dataprep functionality
    """
    def __init__(self, root_dir: List[str],
                 min_reads: Optional[int] = 20,
                 norm_path: Optional[str] = None,
                 num_neighboring_features: Optional[int] = 1,
                 mode: Optional[str] = 'Inference',
                 n_processes: Optional[int] = 1):
        r'''
        Initialization function for the class

                Args:
                        root_dir (list[str]): A list of string filepaths, each one referring to a directory with data.json and data.info or data.info.labelled files from m6anet dataprep functionality
                        min_reads (int): Minimum number of reads for a site to be considered for inference
                        norm_path (str): A filepath to a joblib object containing normalization factors
                        num_neighboring_features (int): Number of flanking positions around the target site involved during features extraction
                        mode (str): The running mode for the dataset instance, must be one of Train, Val, or Test
                        n_processes (int): Number of processes used to compute normalization factors if needed

                Returns:
                        None

                Raises:
                    ValueError: Raises an exception when some of the listed arguments do not follow the allowed conventions
        '''
        super().__init__(root_dir, min_reads, norm_path,
                         num_neighboring_features, mode,
                         n_processes)

    def initialize_data_info(self):
        r'''
        Instance method for the class to initialize indices corresponding to features to be extracted from data.json file
        '''
        if self.mode == 'Inference':
            suffix = "data.info"
            indices = ["transcript_id", "transcript_position"]
        else:
            suffix = "data.info.labelled"
            indices = ["transcript_id", "transcript_position", "modification_status", "set_type"]

        all_read_info = [pd.read_csv(os.path.join(_root_dir, suffix))\
                         .assign(fpath=_root_dir)\
                         .set_index(indices)
                        for _root_dir in self.root_dir]

        read_info = pd.concat(all_read_info, axis=1)
        total_reads = read_info["n_reads"].sum(axis=1).reset_index(drop=True)
        start = read_info["start"].apply(lambda x: [int(col) for col in x if col == col], axis=1)
        end = read_info["end"].apply(lambda x: [int(col) for col in x if col == col], axis=1)
        fpath = read_info["fpath"].apply(lambda x: [col for col in x if col == col], axis=1).reset_index(drop=True)
        coord = pd.concat([start, end], axis=1).apply(lambda x: [(x, y) for x, y in zip(x[0],x[1])], axis=1)\
            .reset_index(drop=True)

        read_info = read_info.reset_index()[indices]
        read_info["n_reads"] = total_reads.astype('int')
        read_info["coords"] = coord
        read_info["fpath"] = fpath

        if self.mode != 'Inference':
            read_info = read_info[read_info["set_type"] == self.mode]

        self.data_info = read_info[read_info["n_reads"] >= self.min_reads].reset_index(drop=True)

        self.fpath_mapping = {_root_dir: num_rep for num_rep, _root_dir in enumerate(self.root_dir)}

    def get_total_neighboring_features(self) -> int:
        r'''
        Instance method for the class to infer the total number of neighboring positions present in data.json

                Args:
                        None

                Returns:
                        Integer representing the total number of neighboring positions in data.json
        '''
        tx_id, tx_pos, coords, fpaths = self.data_info.iloc[0][["transcript_id", "transcript_position",
                                                               "coords", "fpath"]]
        start_pos, end_pos = coords[0]
        fpath = os.path.join(fpaths[0], "data.json")

        kmer, _ = self._load_data(fpath, tx_id, tx_pos, start_pos, end_pos)
        return (len(kmer) - 5) // 2

    def load_data(self, idx: int):
        r'''
        Instance method for the class to load features from reads belonging to the idx-th site in data_info attribute
                Args:
                        idx (int): Integer representing the index position of the sites to load

                Returns:
                        tuple(int, int, np.ndarray, np.ndarray, str): Tuple containing transcript_id information, transcript_position information, read_ids, features,
                                                                      and 5-mer motifs information from reads belonging to the idx-th site in data info
        '''
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

    def compute_norm_factors(self, n_processes: int) -> Dict:
        r'''
        Instance method to compute normalization factors for each 5-mer motif present in the data
                Args:
                        n_processes (int): Number of CPU processes to be allocated for th processing

                Returns:
                        norm_dict (dict): Dictionary containing normalization factors for each 5-mer motif present in the data
        '''
        kmer_mapping_df = self.annotate_kmer_information(self.data_info, n_processes)
        kmer_mapping_df = self.create_kmer_mapping_df(kmer_mapping_df)
        norm_dict = self.create_norm_dict(kmer_mapping_df, n_processes)
        return norm_dict

    def annotate_kmer_information(self, data_info: pd.DataFrame, n_processes: int) -> pd.DataFrame:
        r'''
        Instance method to annotate kmer-information for each entry in the data.info table
                Args:
                        data_info (pd.DataFrame): pd.DataFrame object containing filepath, transcript_id, transcript_position, and coords of reads to load
                        n_processes (int): Number of CPU processes to be allocated for th processing

                Returns:
                        data_info (pd.DataFrame): pd.DataFrame object with 5-mer motifs annotations
        '''
        tasks = ((os.path.join(fpaths[0], "data.json"), tx_id, tx_pos, coords[0][0], coords[0][1]) for fpaths, tx_id, tx_pos, coords in
                    zip(data_info["fpath"], data_info["transcript_id"], data_info["transcript_position"], data_info["coords"]))
        with Pool(n_processes) as p:
            kmer_info = [x for x in tqdm(p.imap(read_kmer, tasks), total=len(data_info))]
        data_info["kmer"] = kmer_info
        return data_info

    def create_kmer_mapping_df(self, kmer_annotated_df: pd.DataFrame) -> pd.DataFrame:
        r'''
        Instance method to associate 5-mer motif with corresponding indices in features array from each transcriptomic position in data.json
                Args:
                        kmer_annotated_df (pd.DataFrame): pd.DataFrame object containing filepath, transcript_id, transcript_position, coords of reads to load, and 5-mer motifs annotation

                Returns:
                        kmer_mapping_df (pd.DataFrame): pd.DataFrame object with 5-mer motifs annotations and corresponding feature indices information
        '''
        kmer_mapping_df = []
        for _, row in kmer_annotated_df.iterrows():
            tx, tx_position, sequence, coords, fpaths, n_reads = row["transcript_id"], \
                    row["transcript_position"], row["kmer"], row["coords"], row["fpath"], row["n_reads"]
            kmers = [sequence[i:i+5] for i in range(len(sequence) - 4)]
            for i in range(len(kmers)):
                kmer_mapping_df += [(tx, tx_position, coords, fpaths, n_reads, kmers[i], i)]
        kmer_mapping_df = pd.DataFrame(kmer_mapping_df,
                                    columns=["transcript_id", "transcript_position", "coords",
                                             "fpath", "n_reads", "kmer", "segment_number"])
        return kmer_mapping_df

    def create_norm_dict(self, kmer_mapping_df: pd.DataFrame, n_processes: int) -> Dict:
        r'''
        Instance method to associate 5-mer motif with corresponding indices in features array from each transcriptomic position in data.json
                Args:
                        kmer_mapping_df (pd.DataFrame): pd.DataFrame object with 5-mer motifs annotations and corresponding feature indices information
                        n_processes (int): Number of CPU processes to be allocated for th processing
                Returns:
                        norm_dict (Dict): Dictionary containing normalization factors for each 5-mer motif
        '''
        tasks = ((kmer, df) for kmer, df in kmer_mapping_df.groupby("kmer"))
        with Pool(n_processes) as p:
            norm_dict = [x for x in tqdm(p.imap_unordered(get_mean_std_replicates, tasks),
                                         total=len(kmer_mapping_df["kmer"].unique()))]
        norm_dict = {tup[0]: (tup[1], tup[2]) for tup in norm_dict}
        return norm_dict


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
