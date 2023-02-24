r"""
This module is a collection of functions used to compute normalization factors from data.json
"""
import os
import json
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple


def get_mean_std(task: Tuple[str, pd.DataFrame, str]) -> Tuple[str, np.ndarray, np.ndarray]:
    r'''
    Function to calculate mean and standard deviation for each feature corresponding to a specific 5-mer motif from a single replicate

            Args:
                    task (tuple[str, pd.DataFrame, str]): A tuple containing 5-mer motif information, dataframe containing indices of reads within data.json,
                                                          and a filepath string to data.json
            Returns:
                    (tuple[str, np.ndarray, np.ndarray]): A tuple containing the 5-mer motif, the mean normalization factors, and standard deviation
                                                          normalization factors
    '''
    kmer, site_df, data_fpath = task
    n_reads = 0
    sum_X = []
    sum_X2 = []
    for _, row in site_df.iterrows():
        tx_id, tx_pos = row["transcript_id"], row["transcript_position"]
        start_pos, end_pos, segment_number = row["start"], row["end"], row["segment_number"]
        features = read_features(data_fpath, tx_id, tx_pos, start_pos, end_pos)
        indices = np.arange(3 * segment_number, 3 * (segment_number + 1))
        n_reads += row["n_reads"]
        signals = features[:, indices]
        sum_X.append(np.sum(signals, axis=0))
        sum_X2.append(np.sum(np.square(signals), axis=0))

    mean = np.sum(sum_X, axis=0) / n_reads
    stds = np.sqrt((np.sum(sum_X2, axis=0) / n_reads) - mean ** 2)
    return kmer, mean, stds


def get_mean_std_replicates(task) -> Tuple[str, np.ndarray, np.ndarray]:
    r'''
    Function to calculate mean and standard deviation for each feature corresponding to a specific 5-mer motif from multiple replicates

            Args:
                    task (tuple[str, pd.DataFrame]): A tuple containing 5-mer motif information, dataframe containing indices of reads within several replicates of data.json
            Returns:
                    (tuple[str, np.ndarray, np.ndarray]): A tuple containing the 5-mer motif, the mean normalization factors, and standard deviation
                                                          normalization factors
    '''
    kmer, site_df = task
    n_reads = 0
    sum_X = []
    sum_X2 = []
    for _, row in site_df.iterrows():
        tx_id, tx_pos = row["transcript_id"], row["transcript_position"]
        coords, fpaths, segment_number = row["coords"], row["fpath"], \
                row["segment_number"]
        for coord, fpath in zip(coords, fpaths):
            start_pos, end_pos = coord
            features = read_features(os.path.join(fpath, "data.json"),
                                     tx_id, tx_pos, start_pos, end_pos)
            indices = np.arange(3 * segment_number,
                                3 * (segment_number + 1))
            signals = features[:, indices]
            sum_X.append(np.sum(signals, axis=0))
            sum_X2.append(np.sum(np.square(signals), axis=0))

        n_reads += row["n_reads"]

    mean = np.sum(sum_X, axis=0) / n_reads
    stds = np.sqrt((np.sum(sum_X2, axis=0) / n_reads) - mean ** 2)
    return kmer, mean, stds


def read_kmer(task: Tuple[str, str, int, int, int]) -> str:
    r'''
    Function to read 5-mer motif information from a specific transcript coordinate within data.json

            Args:
                    task (tuple[str, str, int, int, int]): A tuple containing file path to data.json as well as transcript coordinates and their indices within data.json
            Returns:
                    kmer (str): A string representing the sequence motif of the query coordinates
    '''
    data_fpath, tx_id, tx_pos, start_pos, end_pos = task
    with open(data_fpath, encoding='utf-8') as f:
        f.seek(start_pos, 0)
        json_str = f.read(end_pos - start_pos)
        pos_info = json.loads(json_str)[tx_id][str(tx_pos)]

        assert(len(pos_info.keys()) == 1)

        kmer, _ = list(pos_info.items())[0]
        return kmer


def read_features(data_fpath: str, tx_id:int , tx_pos: int, start_pos: int, end_pos:int) -> np.ndarray:
    r'''
    Function to read features information from a specific transcript coordinate within data.json

            Args:
                    data_fpath (str): File path to data.json
                    tx_id (str): Transcript id of the query coordinate
                    tx_pos (int): Transcript position of the query coordinate
                    start_pos (int): Start position within data.json file that corresponds to the query coordinate
                    end_pos (int): End position within data.json file that corresponds to the query coordinate
            Returns:
                    features (np.ndarray): A NumPy array representing features from the query coordinate
    '''
    with open(data_fpath, encoding='utf-8') as f:
        f.seek(start_pos, 0)
        json_str = f.read(end_pos - start_pos)
        pos_info = json.loads(json_str)[tx_id][str(tx_pos)]

        assert(len(pos_info.keys()) == 1)

        _, features = list(pos_info.items())[0]
        features = np.array(features)
        return features


def annotate_kmer_information(data_fpath: str, data_info: pd.DataFrame, n_processes: int) -> pd.DataFrame:
    r'''
    Function to annotate kmer-information for each entry in the data.info table
            Args:
                    data_fpath (str): String filepath to data.json
                    data_info (pd.DataFrame): pd.DataFrame object containing start and end position within data.json for each transcript coordinate
                    n_processes (int): Number of CPU processes to be allocated for th processing

            Returns:
                    data_info (pd.DataFrame): pd.DataFrame object with 5-mer motifs annotations
    '''
    tasks = ((data_fpath, tx_id, tx_pos, start, end) for tx_id, tx_pos, start, end in zip(data_info["transcript_id"], data_info["transcript_position"],
                                                                                          data_info["start"], data_info["end"]))
    with Pool(n_processes) as p:
        kmer_info = [x for x in tqdm(p.imap(read_kmer, tasks),
                                     total=len(data_info))]
    data_info["kmer"] = kmer_info
    return data_info


def create_kmer_mapping_df(kmer_annotated_df) -> pd.DataFrame:
    r'''
    Function to associate 5-mer motif with corresponding indices in features array from each transcriptomic position in data.json
            Args:
                    kmer_annotated_df (pd.DataFrame): pd.DataFrame object containing transcript_id, transcript_position, coords of reads to load, number of reads, and 5-mer motifs annotation

            Returns:
                    kmer_mapping_df (pd.DataFrame): pd.DataFrame object with 5-mer motifs annotations and corresponding feature indices information
    '''
    kmer_mapping_df = []
    for _, row in kmer_annotated_df.iterrows():
        tx, tx_position, sequence, start, end, n_reads = row["transcript_id"], row["transcript_position"], row["kmer"], row["start"], row["end"], row["n_reads"]
        kmers = [sequence[i:i+5] for i in range(len(sequence) - 4)]
        for i in range(len(kmers)):
            kmer_mapping_df += [(tx, tx_position, start, end, n_reads, kmers[i], i)]

    kmer_mapping_df = pd.DataFrame(kmer_mapping_df,
                                   columns=["transcript_id", "transcript_position", "start", "end", "n_reads", "kmer", "segment_number"])
    return kmer_mapping_df


def create_norm_dict(kmer_mapping_df: pd.DataFrame, data_fpath: str, n_processes: int):
    r'''
    Function to associate 5-mer motif with corresponding indices in features array from each transcriptomic position in data.json
            Args:
                    kmer_mapping_df (pd.DataFrame): pd.DataFrame object with 5-mer motifs annotations and corresponding feature indices information
                    data_fpath (str): String filepath to data.json
                    n_processes (int): Number of CPU processes to be allocated for th processing
            Returns:
                    norm_dict (Dict): Dictionary containing normalization factors for each 5-mer motif
    '''
    tasks = ((kmer, df, data_fpath) for kmer, df in kmer_mapping_df.groupby("kmer"))
    with Pool(n_processes) as p:
        norm_dict = [x for x in tqdm(p.imap_unordered(get_mean_std, tasks),
                                     total=len(kmer_mapping_df["kmer"]\
                                            .unique()))]
    return {tup[0]: (tup[1], tup[2]) for tup in norm_dict}
