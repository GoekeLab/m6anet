import argparse
import numpy as np
import pandas as pd
import os
import multiprocessing 
import h5py
import csv
from itertools import product
from operator import itemgetter
from collections import defaultdict

from . import helper
from ..utils import misc

def get_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--eventalign', dest='eventalign', help='Eventalign filepath from nanopolish.',required=True)
    parser.add_argument('--summary', dest='summary', help='Summary filepath from nanopolish.',required=True)
    parser.add_argument('--out_dir', dest='out_dir', help='Output directory.',required=True)

    # Optional
    # parser.add_argument('--features', dest='features', help='Signal features to extract.',type=list,default=['norm_mean'])
    parser.add_argument('--n_processes', dest='n_processes', help='Number of processes.',type=int, default=1)

    return parser.parse_args()

def combine(read_name, eventalign_per_read, out_paths, locks):
    eventalign_result = pd.DataFrame.from_records(eventalign_per_read)

    cond_successfully_eventaligned = eventalign_result['reference_kmer'] == eventalign_result['model_kmer']
    eventalign_result = eventalign_result[cond_successfully_eventaligned]

    keys = ['read_index','contig','position','reference_kmer'] # for groupby
    eventalign_result['length'] = pd.to_numeric(eventalign_result['end_idx']) - pd.to_numeric(eventalign_result['start_idx'])
    eventalign_result['sum_norm_mean'] = pd.to_numeric(eventalign_result['event_level_mean']) * eventalign_result['length']
    eventalign_result['sum_norm_std'] = pd.to_numeric(eventalign_result['event_stdv']) * eventalign_result['length']
    eventalign_result['sum_dwell_time'] = pd.to_numeric(eventalign_result['event_length']) * eventalign_result['length']

    eventalign_result = eventalign_result.groupby(keys)  
    sum_norm_mean = eventalign_result['sum_norm_mean'].sum() 
    sum_norm_std = eventalign_result["sum_norm_std"].sum()
    sum_dwell_time = eventalign_result["sum_dwell_time"].sum()

    start_idx = eventalign_result['start_idx'].min().astype('i8')
    end_idx = eventalign_result['end_idx'].max().astype('i8')
    total_length = eventalign_result['length'].sum()

    eventalign_result = pd.concat([start_idx,end_idx],axis=1)
    eventalign_result['norm_mean'] = sum_norm_mean / total_length
    eventalign_result["norm_std"] = sum_norm_std / total_length
    eventalign_result["dwell_time"] = sum_dwell_time / total_length
    eventalign_result.reset_index(inplace=True)

    eventalign_result['transcript_id'] = [contig for contig in eventalign_result['contig']]
    eventalign_result['transcriptomic_position'] = pd.to_numeric(eventalign_result['position']) + 2 # the middle position of 5-mers.
    eventalign_result['read_id'] = [read_name] * len(eventalign_result)

    features = ['read_id','transcript_id','transcriptomic_position','reference_kmer','norm_mean','norm_std','dwell_time','start_idx','end_idx']

    df_events_per_read = eventalign_result[features]
    
    # write to file.
    df_events_per_read = df_events_per_read.set_index(['transcript_id','read_id'])

    with locks['hdf5'], h5py.File(out_paths['hdf5'],'a') as hf:
        for tx_id,read_id in df_events_per_read.index.unique():
            df2write = df_events_per_read.loc[[tx_id,read_id],:].reset_index() 
            events = np.rec.fromrecords(misc.str_encode(df2write[features]),names=features) #,dtype=features_dtype
            
            hf_tx = hf.require_group('%s/%s' %(tx_id,read_id))
            if 'events' in hf_tx:
                continue
            else:
                hf_tx['events'] = events
    
    with locks['log'], open(out_paths['log'],'a') as f:
        f.write('%s\n' %(read_name))    

def prepare_for_inference(tx, read_task, all_kmers, out_dir, log_path, locks):    
    reads = np.concatenate(read_task)
    reads = reads[np.isin(reads["reference_kmer"], all_kmers)] # Filter for motifs
    reads = reads[np.argsort(reads["transcriptomic_position"])] # sort by reference kmer
    positions, indices = np.unique(reads["transcriptomic_position"],return_index=True) # retrieve group indexing

    for i in range(len(positions)):
        pos = positions[i]
        if len(positions) > 1:
            start_idx = indices[i]
            end_idx = indices[i + 1] if i < len(positions) - 1 else None
            read = reads[start_idx:end_idx]
            kmer = read["reference_kmer"][0].decode()
            # Converting to np array
            
            X = read[["norm_mean", "norm_std", "dwell_time"]]\
                        .astype([('norm_mean', '<f8'), ('norm_std', '<f8'), ('dwell_time', '<f8')]).view('<f8')
            read_ids = read["read_id"].view('<S36')
            start_event_indices = read["start_idx"].view('<i8')
            end_event_indices = read["end_idx"].view('<i8')
        else:
            read = reads[0]
            kmer = read["reference_kmer"].item().decode()
            # Converting to np array when there is only one entry
            
            X = np.array(read[["norm_mean", "norm_std", "dwell_time"]].tolist())
            read_ids = np.array(read["read_id"].tolist())
            start_event_indices = np.array(read["start_idx"].tolist())
            end_event_indices = np.array(read["end_idx"].tolist())
        
        # Reshaping columns
        X = X.reshape(-1, 3)
        read_ids = read_ids.reshape(-1, 1)
        start_event_indices = start_event_indices.reshape(-1, 1)
        end_event_indices = end_event_indices.reshape(-1, 1)

        # Saving output in hdf5 file format

        n_reads = len(X)
        fname = os.path.join(out_dir, '{}_{}_{}_{}.hdf5'.format(tx, pos, kmer, n_reads))
        with h5py.File(fname, 'w') as f:
            assert(n_reads == len(read_ids))
            f['X'] = X
            f['read_ids'] = read_ids
            f['start_idx'] = start_event_indices
            f['end_idx'] = end_event_indices
        f.close()

    with locks['log'], open(log_path,'a') as f:
        f.write('%s\n' %(tx))   
        
        
def parallel_prepare_for_inference(eventalign_filepath, inference_prep_dir, n_processes):
    # Create output path and locks.
    out_dir = os.path.join(inference_prep_dir, "inference")
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    locks = {'log': multiprocessing.Lock()}
    log_path = os.path.join(inference_prep_dir, "prepare_for_inference.log")
    # Create empty files for logs.
    open(log_path,'w').close()

    # Create communication queues.
    task_queue = multiprocessing.JoinableQueue(maxsize=n_processes * 2)

    # Create and start consumers.
    consumers = [helper.Consumer(task_queue=task_queue,task_function=prepare_for_inference,locks=locks) for i in range(n_processes)]
    for p in consumers:
        p.start()
        
    ## Load tasks into task_queue. A task is a read information from a specific site.

    # Only include reads that conform to DRACH motifs

    all_kmers = np.array(["".join(x) for x in product(['A', 'G', 'T'], ['G', 'A'], ['A'], ['C'], ['A', 'C', 'T'])], dtype='S5')
    with h5py.File(eventalign_filepath, 'r') as f:
        for tx in f:
            read_task = []
            for read in f[tx]:
                read_task.append(f[tx][read]['events'][:])
            task_queue.put((tx, read_task, all_kmers, out_dir, log_path))

    # Put the stop task into task_queue.
    task_queue = helper.end_queue(task_queue, n_processes)

    # Wait for all of the tasks to finish.
    task_queue.join()
    
    with open(log_path,'a+') as f:
        f.write(helper.decor_message('successfully finished'))


def parallel_combine(eventalign_filepath, summary_filepath, out_dir, n_processes):
    # Create output paths and locks.
    out_paths,locks = dict(),dict()
    for out_filetype in ['hdf5','log']:
        out_paths[out_filetype] = os.path.join(out_dir,'eventalign.%s' %out_filetype)
        locks[out_filetype] = multiprocessing.Lock()
        
        
    # Create empty files.
    open(out_paths['hdf5'],'w').close()
    open(out_paths['log'],'w').close()

    # Create communication queues.
    task_queue = multiprocessing.JoinableQueue(maxsize=n_processes * 2)

    # Create and start consumers.
    consumers = [helper.Consumer(task_queue=task_queue,task_function=combine,locks=locks) for i in range(n_processes)]
    for p in consumers:
        p.start()
        
    ## Load tasks into task_queue. A task is eventalign information of one read.            
    with open(eventalign_filepath,'r') as eventalign_file, open(summary_filepath,'r') as summary_file:

        reader_summary = csv.DictReader(summary_file, delimiter="\t")
        reader_eventalign = csv.DictReader(eventalign_file, delimiter="\t")

        row_summary = next(reader_summary)
        read_name = row_summary['read_name']
        read_index = row_summary['read_index']
        eventalign_per_read = []
        for row_eventalign in reader_eventalign:
            if (row_eventalign['read_index'] == read_index):
                eventalign_per_read += [row_eventalign]
            else: 
                # Load a read info to the task queue.
                task_queue.put((read_name,eventalign_per_read,out_paths))
                # Next read.
                try:
                    row_summary = next(reader_summary)
                except StopIteration: # no more read.
                    break
                else:
                    read_index = row_summary['read_index']
                    read_name = row_summary['read_name']
                    assert row_eventalign['read_index'] == read_index 
                    eventalign_per_read = [row_eventalign]
    
    # Put the stop task into task_queue.
    task_queue = helper.end_queue(task_queue,n_processes)

    # Wait for all of the tasks to finish.
    task_queue.join()
    
    with open(out_paths['log'],'a+') as f:
        f.write(helper.decor_message('successfully finished'))


def main():
    args = get_args()
    #
    n_processes = args.n_processes        
    eventalign_filepath = args.eventalign
    summary_filepath = args.summary
    out_dir = args.out_dir

    misc.makedirs(out_dir)
    
    # (1) For each read, combine multiple events aligned to the same positions, the results from nanopolish eventalign, into a single event per position.
    eventalign_log_filepath = os.path.join(out_dir,'eventalign.log')

    if not helper.is_successful(eventalign_log_filepath):
        parallel_combine(eventalign_filepath,summary_filepath,out_dir,n_processes)
    
    # (2) Split the hdf files into multiple files, each containing all reads from a single DRACH site

    parallel_prepare_for_inference(os.path.join(out_dir, 'eventalign.hdf5'), out_dir, n_processes)

if __name__ == '__main__':
    """
    Usage:
        m6anet-dataprep --eventalign --summary --out_dir --n_processes
    """
    main()


