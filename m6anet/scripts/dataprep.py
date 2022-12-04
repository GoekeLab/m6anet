import argparse
from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
import os
import multiprocessing
import ujson
import warnings
from operator import itemgetter
from collections import defaultdict
from itertools import groupby
from io import StringIO

from . import helper
from .constants import M6A_KMERS, NUM_NEIGHBORING_FEATURES
from ..utils import misc


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--eventalign', dest='eventalign',
                        help='eventalign filepath, the output from nanopolish.',required=True)
    parser.add_argument('--out_dir', dest='out_dir',
                        help='output directory.',required=True)
    parser.add_argument('--n_processes', dest='n_processes',
                        help='number of processes to run.',
                        type=int, default=1)
    parser.add_argument('--chunk_size', dest='chunk_size',
                        help='number of lines from nanopolish eventalign.txt for processing.',
                        type=int, default=1000000)
    parser.add_argument('--readcount_min', dest='readcount_min',
                        help='minimum read counts per gene.',
                        type=int, default=1)
    parser.add_argument('--readcount_max', dest='readcount_max',
                        help='maximum read counts per gene.',
                        type=int, default=1000)
    parser.add_argument('--min_segment_count', dest='min_segment_count',
                        help='minimum read counts per candidate segment.',
                        type=int, default=20)
    parser.add_argument('--skip_index', dest='skip_index',
                        help='with this argument the program will skip indexing eventalign.txt first.',
                        default=False, action='store_true')
    parser.add_argument('--n_neighbors', dest='n_neighbors',
                        help='number of neighboring features to extract.',type=int,
                        default=NUM_NEIGHBORING_FEATURES)
    return parser.parse_args()


def partition_into_continuous_positions(arr, window_size):
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
    return [(float_arr[partition],
            kmer_arr[partition], tx_id_arr[partition], tx_pos_arr[partition],
            read_indices[partition])
            for partition in partitions if len(partition) >= 2 * window_size + 1]


def filter_by_kmer(partition, kmers, window_size):
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


def filter_partitions(partitions, window_size, kmers):
    windowed_partition = [create_features(partition, window_size) for partition in partitions]
    filtered_by_kmers = [filter_by_kmer(partition, kmers, window_size) for partition in windowed_partition]
    final_partitions = [x for x in filtered_by_kmers if len(x) > 0]
    return final_partitions


def roll(to_roll, window_size):
    nex = np.concatenate([np.roll(to_roll, i, axis=0) for i in range(-1, - window_size - 1, -1)],
                          axis=1)
    prev = np.concatenate([np.roll(to_roll, i, axis=0) for i in range(window_size, 0, -1)], axis=1)
    return np.concatenate((prev, to_roll, nex), axis=1)[window_size: -window_size, :]


def create_features(partition, window_size):
    float_arr, kmer_arr, tx_id_arr, tx_pos_arr = partition[:4]
    return roll(float_arr, window_size), roll(kmer_arr, window_size), \
        tx_id_arr[window_size: -window_size], tx_pos_arr[window_size: -window_size], \
        partition[-1][window_size: -window_size]


def filter_events(events, window_size, kmers):
    events = partition_into_continuous_positions(events, window_size)
    events = filter_partitions(events, window_size, kmers)
    return events


def combine_sequence(kmers):
    kmer = kmers[0]
    for _kmer in kmers[1:]:
        kmer += _kmer[-1]
    return kmer


def index(eventalign_result,pos_start,out_paths,locks):
   eventalign_result = eventalign_result.set_index(['contig','read_index'])
   pos_end=pos_start
   with locks['index'], open(out_paths['index'],'a', encoding='utf-8') as f_index:
       for _index in list(dict.fromkeys(eventalign_result.index)):
           transcript_id,read_index = _index
           pos_end += eventalign_result.loc[_index]['line_length'].sum()
           f_index.write('%s,%d,%d,%d\n' %(transcript_id,read_index,pos_start,pos_end))
           pos_start = pos_end


def parallel_index(eventalign_filepath,chunk_size,out_dir,n_processes):
    # Create output paths and locks.
    out_paths,locks = dict(),dict()
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
        task_queue.put((chunk_concat[index_features],pos_start,out_paths))
        pos_start += sum(lines)
        chunk_split = chunk[chunk['read_index'] == chunk.iloc[-1]['read_index']].copy()

    ## the loop above leaves off w/o adding the last read_index to eventalign.index
    chunk_split_size = len(chunk_split.index)
    lines = [len(eventalign_file.readline()) for i in range(chunk_split_size)]
    chunk_split.loc[:, 'line_length'] = np.array(lines)
    task_queue.put((chunk_split[index_features],pos_start,out_paths))

    # Put the stop task into task_queue.
    task_queue = helper.end_queue(task_queue,n_processes)

    # Wait for all of the tasks to finish.
    task_queue.join()


def combine(events_str):
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


def parallel_preprocess_tx(eventalign_filepath,out_dir,n_processes,readcount_min,readcount_max, n_neighbors, min_segment_count):

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
    consumers = [helper.Consumer(task_queue=task_queue,task_function=preprocess_tx,locks=locks) for i in range(n_processes)]

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
                task_queue.put((tx_id,data_dict, n_neighbors, min_segment_count, out_paths))

    # Put the stop task into task_queue.
    task_queue = helper.end_queue(task_queue, n_processes)

    # Wait for all of the tasks to finish.
    task_queue.join()


def preprocess_tx(tx_id, data_dict, n_neighbors, min_segment_count, out_paths, locks):
    """
    Extract features from eventalign.txt on a single transcript id

    Parameters
    ----------
        tx_id: str
            Transcript ID.
        data_dict: {read_id:events_array}
            Dictionary containing events for each read.
        n_neighbors: [int]
            A list of features to collect from the reads that are aligned to each genomic coordinate in the output.
        min_segment_count: int
            Minimum number of reads per segment
        out_paths: {out_file_type: fpath}
            Dictionary containing output filepath for each output file type

    Returns
    -------
    None
    """

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


def main():
    args = get_args()
    #
    n_processes = args.n_processes
    eventalign_filepath = args.eventalign
    chunk_size = args.chunk_size
    out_dir = args.out_dir
    readcount_min = args.readcount_min
    readcount_max = args.readcount_max
    skip_index = args.skip_index
    n_neighbors = args.n_neighbors
    min_segment_count = args.min_segment_count
    misc.makedirs(out_dir)

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    if not skip_index:
        parallel_index(eventalign_filepath,chunk_size,out_dir,n_processes)

    # For each read, combine multiple events aligned to the same positions,
    # the results from nanopolish eventalign, into a single event per position.

    parallel_preprocess_tx(eventalign_filepath, out_dir, n_processes,
                           readcount_min, readcount_max, n_neighbors, min_segment_count)

if __name__ == '__main__':
    main()
