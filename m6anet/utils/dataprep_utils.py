r"""
This module is a collection of functions needed to run m6anet dataprep
"""
import os
import multiprocessing
from io import StringIO
from itertools import groupby
from collections import defaultdict
from operator import itemgetter
from math import floor, log10
import numpy as np
import pandas as pd
import ujson
from typing import List, Tuple, Dict, Union
from . import helper
from ..utils.constants import M6A_KMERS


def filter_by_kmer(partition: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                   kmers: List[str], window_size: int) -> \
                        Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], List]:
    r'''
    Function to filter features information from each segment from each individual read within eventalign.txt file for DRACH 5-mer motif

            Args:
                    partition (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]): A tuple containing extracted features, 5-mer motifs,
                                                                                                   transcript id, transcript position, read_index within
                                                                                                   each segment and its flanking neighbors
                    kmers (list[str]): 5-mer motif information for each segment

            Returns:
                    (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]): A tuple containing extracted features from neighboring positions, 5-mer motifs,
                                                                                         transcript id, transcript position, read_index after being filtered for DRACH
                                                                                         segments
    '''
    feature_arr, kmer_arr, tx_id_arr, tx_pos_arr = partition[:4]
    kmers_5 = kmer_arr[:, (2 * window_size + 1) // 2]
    mask = np.isin(kmers_5, kmers)
    filtered_feature_arr = feature_arr[mask, :]
    filtered_kmer_arr = kmer_arr[mask, :]
    filtered_tx_pos_arr = tx_pos_arr[mask]
    filtered_tx_id_arr = tx_id_arr[mask]
    filtered_read_id_arr = partition[-1][mask]

    if len(filtered_kmer_arr) == 0:
        return []
    else:
        return filtered_feature_arr, filtered_kmer_arr, filtered_tx_id_arr, filtered_tx_pos_arr, filtered_read_id_arr


def roll(to_roll: np.ndarray, window_size: int) -> np.ndarray:
    r'''
    Function to extract information from window_size flanking segments. Each information from each position in to_roll is presented in
    a row-wise fashion and this function converts it into columnar format

            Args:
                    to_roll (np.ndarray): A NumPy array containing information from consecutive segments in a row-wise manner
                    window_size (int): The number of neighboring segments to be included in the feature extraction process

            Returns:
                    combined (np.ndarray): A NumPy array containing combined information in a column-wise manner
    '''
    nex = np.concatenate([np.roll(to_roll, i, axis=0) for i in range(-1, - window_size - 1, -1)],
                          axis=1)
    prev = np.concatenate([np.roll(to_roll, i, axis=0) for i in range(window_size, 0, -1)], axis=1)
    combined = np.concatenate((prev, to_roll, nex), axis=1)[window_size: -window_size, :]
    return combined


def create_windowed_features(partition: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], window_size: int) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r'''
    Function to extract features from neighboring positions within a stretch of continuous segments from a single read in eventalign.txt file

            Args:
                    partitions (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]): A tuple containing extracted features, 5-mer motifs, transcript_id, transcript_positions,
                                                                                                    read_indices from consecutive segments from a single read from eventalign.txt file
                    window_size (int): The number of neighboring segments to be included in the feature extraction process

            Returns:
                    windowed_results (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]): A tuple containing extracted features, 5-mer motifs,
                                                                                                          transcript id, transcript position, read_index within each segment and its
                                                                                                          flanking neighbors
    '''
    float_arr, kmer_arr, tx_id_arr, tx_pos_arr = partition[:4]
    windowed_results = roll(float_arr, window_size), roll(kmer_arr, window_size), \
            tx_id_arr[window_size: -window_size], tx_pos_arr[window_size: -window_size], \
            partition[-1][window_size: -window_size]
    return windowed_results


def process_partitions(partitions: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                       window_size: int, kmers: List[str]):
    r'''
    Function to extract features from list of partitions and filter for DRACH 5-mer motifs 5-mer motifs

            Args:
                    partitions (list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]): A list of tuples containing extracted features from neighboring positions, 5-mer motifs,
                                                                                                          transcript id, transcript position, read_index after being filtered to ensure only DRACH
                                                                                                          segments with window_size flanking segments being present are selected from the read
                    window_size (int): The number of neighboring segments to be included in the feature extraction process
                    kmers (list[str]): 5-mer motif information for each segment

            Returns:
                    final_partitions (list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]): A list of tuples containing extracted features from neighboring positions, 5-mer motifs,
                                                                                                                transcript id, transcript position, read_index after being filtered to ensure only DRACH
                                                                                                                segments with window_size flanking segments being present are selected from the read
    '''
    windowed_partition = [create_windowed_features(partition, window_size) for partition in partitions]
    filtered_by_kmers = [filter_by_kmer(partition, kmers, window_size) for partition in windowed_partition]
    final_partitions = [x for x in filtered_by_kmers if len(x) > 0]
    return final_partitions


def partition_into_continuous_positions(arr: np.recarray, window_size: int) -> \
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    r'''
    Function to filter and partition a portion of eventalign.txt file belonging to a single read into continuous segments of features. Only partitions
    with at least window_size flanking positions being present for each position are selected

            Args:
                    events (np.recarray): A NumPy record array object containing transcript information, read index, position, 5-mer motif, mean current,
                                          standard deviation of current and dwelling time for each segment within a single read
                    window_size (int): The number of neighboring segments to be included in the feature extraction process

            Returns:
                    partitions (list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]): A list of tuples containing extracted features from neighboring positions, 5-mer motifs, transcript id,
                                                                                                          transcript position, read_index after being filtered to ensure only segments with at least window_size flanking
                                                                                                          positions to be present from each read
    '''
    arr = arr[np.argsort(arr["transcriptomic_position"])]
    float_features = ['dwell_time', 'norm_std', 'norm_mean']
    float_dtypes = [('norm_mean', '<f8'), ('norm_std', '<f8'), ('dwell_time', '<f8')]

    float_arr = arr[float_features].astype(float_dtypes).view('<f8').reshape(-1, 3)
    kmer_arr = arr["reference_kmer"].reshape(-1, 1)
    tx_pos_arr = arr["transcriptomic_position"]
    tx_id_arr = arr["transcript_id"]
    read_indices = arr["read_index"]

    partitions = [list(map(itemgetter(0), g)) for k, g in groupby(enumerate(tx_pos_arr),
                                                                  lambda x: x[0] - x[1])]
    partitions = [(float_arr[partition],
                   kmer_arr[partition], tx_id_arr[partition], tx_pos_arr[partition],
                   read_indices[partition])
                   for partition in partitions if len(partition) >= 2 * window_size + 1]
    return partitions


def filter_events(events: np.recarray, window_size: int, kmers: List[str]) -> \
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    r'''
    Function to combine 5-mer motifs from consecutive positions into a single sequence of nucleotide

            Args:
                    events (np.recarray): A NumPy record array object containing transcript information, read index, position, 5-mer motif, mean current,
                                          standard deviation of current and dwelling time for each segment within a single read
                    window_size (int): The number of neighboring segments to be included in the feature extraction process
                    kmers (list[str]): 5-mer motif information for each segment

            Returns:
                    events (list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]): A list of tuples containing extracted features from neighboring positions, 5-mer motifs, transcript id,
                                                                                                      transcript position, read_index after being filtered to ensure only DRACH segments with window_size flanking
                                                                                                      segments being present are selected from the read
    '''
    events = partition_into_continuous_positions(events, window_size)
    events = process_partitions(events, window_size, kmers)
    return events


def combine_sequence(kmers: List[str]) -> str:
    r'''
    Function to combine 5-mer motifs from consecutive positions into a single sequence of nucleotide

            Args:
                    kmers (list[str]): A list containing 5-mer motif strings from consecutive transcriptomic position

            Returns:
                    kmer (str): A string representing combined sequence of nucleotide corresponding to the consecutive positions in the list of 5-mers
    '''
    kmer = kmers[0]
    for _kmer in kmers[1:]:
        kmer += _kmer[-1]
    return kmer


def index(eventalign_result: pd.DataFrame, pos_start: int, out_paths: Dict, locks: Dict):
    r'''
    Function to index the position of a specific read features within eventalign.txt

            Args:
                    eventalign_result (pd.DataFrame): A pd.DataFrame object containing a portion of eventalign.txt to be indexed
                    pos_start (int): An index position within eventalign.txt that corresponds to the start of the eventalign_result portion within eventalign.txt file
                    out_paths (Dict): A dictionary containing filepath for all the output files produced by the index function
                    locks (Dict): A lock object from multiprocessing library that ensures only one process write to the output file at any given time

            Returns:
                    None
    '''
    eventalign_result = eventalign_result.set_index(['contig','read_index'])
    pos_end=pos_start
    with locks['index'], open(out_paths['index'],'a', encoding='utf-8') as f_index:
        for _index in list(dict.fromkeys(eventalign_result.index)):
            transcript_id,read_index = _index
            pos_end += eventalign_result.loc[_index]['line_length'].sum()
            f_index.write('%s,%d,%d,%d\n' %(transcript_id,read_index,pos_start,pos_end))
            pos_start = pos_end


def parallel_index(eventalign_filepath: str, chunk_size:int , out_dir:str, n_processes:int):
    r'''
    Function to index every read within eventalign.txt file for faster access later

            Args:
                    eventalign_filepath (str): String filepath to the eventalign.txt file
                    chunk_size (int): Chunksize argument for pd.read_csv function
                    out_dir (str):  String filepath to the output directory of the indexing function
                    n_processes (int): Number of processes used for indexing

            Returns:
                    None
    '''
    # Create output paths and locks.
    out_paths, locks = dict(), dict()
    for out_filetype in ['index']:
        out_paths[out_filetype] = os.path.join(out_dir,'eventalign.%s'  %out_filetype)
        locks[out_filetype] = multiprocessing.Lock()

    with open(out_paths['index'],'w', encoding='utf-8') as f:
        f.write('transcript_id,read_index,pos_start,pos_end\n') # header

    # Create communication queues.
    task_queue = multiprocessing.JoinableQueue(maxsize=n_processes * 2)

    # Create and start consumers.p
    consumers = [helper.Consumer(task_queue=task_queue,task_function=index,locks=locks) for i in range(n_processes)]
    for process in consumers:
        process.start()

    ## Load tasks into task_queue. A task is eventalign information of one read.
    eventalign_file = open(eventalign_filepath,'r', encoding='utf-8')
    pos_start = len(eventalign_file.readline()) #remove header
    chunk_split = None
    index_features = ['contig','read_index','line_length']
    for chunk in pd.read_csv(eventalign_filepath, chunksize=chunk_size,sep='\t'):
        chunk_complete = chunk[chunk['read_index'] != chunk.iloc[-1]['read_index']]
        chunk_concat = pd.concat([chunk_split,chunk_complete])
        chunk_concat_size = len(chunk_concat.index)
        ## read the file at where it left off because the file is opened once ##
        lines = [len(eventalign_file.readline()) for i in range(chunk_concat_size)]
        chunk_concat.loc[:, 'line_length'] = np.array(lines)
        task_queue.put((chunk_concat[index_features], pos_start, out_paths))
        pos_start += sum(lines)
        chunk_split = chunk[chunk['read_index'] == chunk.iloc[-1]['read_index']].copy()

    ## the loop above leaves off w/o adding the last read_index to eventalign.index
    chunk_split_size = len(chunk_split.index)
    lines = [len(eventalign_file.readline()) for i in range(chunk_split_size)]
    chunk_split.loc[:, 'line_length'] = np.array(lines)
    task_queue.put((chunk_split[index_features], pos_start, out_paths))

    # Put the stop task into task_queue.
    task_queue = helper.end_queue(task_queue,n_processes)

    # Wait for all of the tasks to finish.
    task_queue.join()


def combine(events_str: str) -> np.recarray:
    r'''
    Function to aggregate features from eventalign.txt on a single transcript id and extract mean current level, dwelling time, and current standard deviation from each position in each read
            Args:
                    events_str (str): String corresponding to portion within eventalign.txt to be processed that can be read as pd.DataFrame object

            Returns
                    np_events (np.recarray): A NumPy record array object that contains the extracted features
    '''
    f_string = StringIO(events_str)
    eventalign_result = pd.read_csv(f_string,delimiter='\t',
                                    names=['contig','position','reference_kmer','read_index','strand','event_index',
                                           'event_level_mean','event_stdv','event_length','model_kmer','model_mean',
                                           'model_stdv','standardized_level','start_idx','end_idx'])
    f_string.close()
    cond_successfully_eventaligned = eventalign_result['reference_kmer'] == eventalign_result['model_kmer']

    if cond_successfully_eventaligned.sum() != 0:

        eventalign_result = eventalign_result[cond_successfully_eventaligned]

        keys = ['read_index','contig','position','reference_kmer'] # for groupby
        eventalign_result.loc[:, 'length'] = pd.to_numeric(eventalign_result['end_idx']) - \
                pd.to_numeric(eventalign_result['start_idx'])
        eventalign_result.loc[:, 'sum_norm_mean'] = pd.to_numeric(eventalign_result['event_level_mean']) \
                * eventalign_result['length']
        eventalign_result.loc[:, 'sum_norm_std'] = pd.to_numeric(eventalign_result['event_stdv']) \
                * eventalign_result['length']
        eventalign_result.loc[:, 'sum_dwell_time'] = pd.to_numeric(eventalign_result['event_length']) \
                * eventalign_result['length']

        eventalign_result = eventalign_result.groupby(keys)
        sum_norm_mean = eventalign_result['sum_norm_mean'].sum()
        sum_norm_std = eventalign_result["sum_norm_std"].sum()
        sum_dwell_time = eventalign_result["sum_dwell_time"].sum()

        start_idx = eventalign_result['start_idx'].min()
        end_idx = eventalign_result['end_idx'].max()
        total_length = eventalign_result['length'].sum()

        eventalign_result = pd.concat([start_idx,end_idx],axis=1)
        eventalign_result['norm_mean'] = (sum_norm_mean/total_length).round(1)
        eventalign_result["norm_std"] = sum_norm_std / total_length
        eventalign_result["dwell_time"] = sum_dwell_time / total_length
        eventalign_result = eventalign_result.reset_index()

        eventalign_result['transcript_id'] = eventalign_result['contig']    #### CHANGE MADE ####
        eventalign_result['transcriptomic_position'] = \
                pd.to_numeric(eventalign_result['position']) + 2 # the middle position of 5-mers.
        features = ['transcript_id', 'read_index',
                    'transcriptomic_position', 'reference_kmer',
                    'norm_mean', 'norm_std', 'dwell_time']
        df_events = eventalign_result[features]
        np_events = np.rec.fromrecords(df_events, names=[*df_events])
        return np_events

    return np.array([])


def parallel_preprocess_tx(eventalign_filepath: str, out_dir:str, n_processes:int, readcount_min: int,
                           readcount_max:int, n_neighbors:int, min_segment_count:int, compress:bool):
    r'''
    Function to aggregate segments from the same position within individual read and extract mean current level, dwelling time, and current standard deviation from each DRACH segment
    and its neighboring segments from each in eventalign.txt that passes certain count requirements as specified by the user for the purpose of m6anet inference or training

            Args:
                    eventalign_filepath (str): String filepath to the eventalign.txt file
                    out_dir (str): String filepath to the output directory of the indexing function
                    n_processes (int): Number of processes used for indexing
                    readcount_min (int): Minimum required number of reads expressed by each transcript to be considered for feature extraction
                    readcount_max (int): Maximum reads expressed by a transcript that will be processed for feature extraction
                    n_neighbors (int): The number of neighboring segments to be included in the feature extraction process
                    min_segment_count (int): Minimum required number of reads that cover a segment o be considered for feature extraction

            Returns:
                    None
    '''
    # Create output paths and locks.
    out_paths,locks = dict(),dict()
    for out_filetype in ['json','info','log']:
        out_paths[out_filetype] = os.path.join(out_dir, 'data.%s' % out_filetype)
        locks[out_filetype] = multiprocessing.Lock()

    # Writing the starting of the files.

    open(out_paths['json'], 'w', encoding='utf-8').close()

    with open(out_paths['info'], 'w', encoding='utf-8') as f:
        f.write('transcript_id,transcript_position,start,end,n_reads\n') # header

    open(out_paths['log'], 'w', encoding='utf-8').close()

    # Create communication queues.
    task_queue = multiprocessing.JoinableQueue(maxsize=n_processes * 2)

    # Create and start consumers.
    consumers = [helper.Consumer(task_queue=task_queue, task_function=preprocess_tx, locks=locks) for i in range(n_processes)]

    for process in consumers:
        process.start()

    df_eventalign_index = pd.read_csv(os.path.join(out_dir,'eventalign.index'))
    df_eventalign_index['transcript_id'] = df_eventalign_index['transcript_id']
    tx_ids = df_eventalign_index['transcript_id'].values.tolist()
    tx_ids = list(dict.fromkeys(tx_ids))
    df_eventalign_index = df_eventalign_index.set_index('transcript_id')
    with open(eventalign_filepath, 'r', encoding='utf-8') as eventalign_result:
        for tx_id in tx_ids:
            data_dict = dict()
            readcount = 0
            for _,row in df_eventalign_index.loc[[tx_id]].iterrows():
                read_index,pos_start,pos_end = row['read_index'],row['pos_start'],row['pos_end']
                eventalign_result.seek(pos_start,0)
                events_str = eventalign_result.read(pos_end-pos_start)
                data = combine(events_str)
                if data.size > 1:
                    data_dict[read_index] = data
                readcount += 1
                if readcount > readcount_max:
                    break
            if readcount>=readcount_min:
                task_queue.put((tx_id,data_dict, n_neighbors, min_segment_count, out_paths, compress))

    # Put the stop task into task_queue.
    task_queue = helper.end_queue(task_queue, n_processes)

    # Wait for all of the tasks to finish.
    task_queue.join()


def preprocess_tx(tx_id: str, data_dict: Dict, n_neighbors: int, min_segment_count: int, out_paths: Dict, compress: bool, locks: Dict):
    r'''
    Function to extract features from eventalign.txt on a single transcript id

            Args:
                    tx_id (str): Transcript id of the portion of eventalign.txt file to be processed
                    data_dict (Dict): Dictionary containing events for each read.
                    n_neighbors (int): The number of neighboring segments to be included in the feature extraction process
                    min_segment_count (int): Minimum required number of reads that cover a segment o be considered for feature extraction
                    out_paths (Dict): A dictionary containing filepath for all the output files produced by the index function
                    compress (bool): A boolean variable indicating whether to compress the output data.json features
                    locks (Dict): A lock object from multiprocessing library that ensures only one process write to the output file at any given time

            Returns
                    None
    '''
    if len(data_dict) == 0:
        return

    features_arrays = []
    reference_kmer_arrays = []
    transcriptomic_positions_arrays = []
    read_ids = []

    for _,events_per_read in data_dict.items():
        events_per_read = filter_events(events_per_read, n_neighbors, M6A_KMERS)
        for event_per_read in events_per_read:
            features_arrays.append(event_per_read[0])
            reference_kmer_arrays.append([combine_sequence(kmer) for kmer in event_per_read[1]])
            transcriptomic_positions_arrays.append(event_per_read[3])
            read_ids.append(event_per_read[4])

    if len(features_arrays) == 0:
        return

    features_arrays = np.concatenate(features_arrays)
    reference_kmer_arrays = np.concatenate(reference_kmer_arrays)
    transcriptomic_positions_arrays = np.concatenate(transcriptomic_positions_arrays)
    read_ids = np.concatenate(read_ids)

    assert(len(features_arrays) == len(reference_kmer_arrays) == \
            len(transcriptomic_positions_arrays) == len(read_ids))

    # Sort and split

    idx_sorted = np.argsort(transcriptomic_positions_arrays)
    positions, indices = np.unique(transcriptomic_positions_arrays[idx_sorted],
                                   return_index = True,axis=0) #'chr',

    features_arrays = np.split(features_arrays[idx_sorted], indices[1:])
    reference_kmer_arrays = np.split(reference_kmer_arrays[idx_sorted], indices[1:])
    read_ids = np.split(read_ids[idx_sorted], indices[1:])

    # Prepare
    data = defaultdict(dict)

    # for each position, make it ready for json dump
    for position, features_array, reference_kmer_array, read_id in \
            zip(positions, features_arrays, reference_kmer_arrays, read_ids):
        kmer = set(reference_kmer_array)

        if compress:
                features_array = features_array.round(decimals=3)

        features_array = np.concatenate([features_array, read_id.reshape(-1, 1)], axis=1)
        assert(len(kmer) == 1)
        if (len(set(reference_kmer_array)) == 1) and ('XXXXX' in set(reference_kmer_array)) \
                or (len(features_array) == 0):
            continue
        if len(features_array) >= min_segment_count:
            data[int(position)] = {kmer.pop(): features_array.tolist()}

    # write to file.
    log_str = '%s: Data preparation ... Done.' %(tx_id)
    with locks['json'], open(out_paths['json'],'a', encoding='utf-8') as f:
        with locks['info'], open(out_paths['info'],'a', encoding='utf-8') as g:
            for pos, dat in data.items():
                pos_start = f.tell()
                f.write('{')
                f.write('"%s":{"%d":' %(tx_id,pos))
                ujson.dump(dat, f)
                f.write('}}\n')
                pos_end = f.tell()
                n_reads = 0
                for kmer, features in dat.items():
                    n_reads += len(features)
                g.write('%s,%d,%d,%d,%d\n' %(tx_id,pos,pos_start,pos_end,n_reads))

    with locks['log'], open(out_paths['log'],'a', encoding='utf-8') as f:
        f.write(log_str + '\n')
