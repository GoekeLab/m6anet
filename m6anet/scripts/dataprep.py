import argparse
import numpy as np
import pandas as pd
import os,re
import multiprocessing 
import h5py
import csv
import ujson
from pyensembl import EnsemblRelease
from pyensembl import Genome
from operator import itemgetter
from collections import defaultdict
from itertools import groupby
from io import StringIO

from . import helper
from .constants import M6A_KMERS, KMER_TO_INT, NUM_NEIGHBORING_FEATURES
from ..utils import misc


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')

    # Required arguments
    required.add_argument('--eventalign', dest='eventalign', help='eventalign filepath, the output from nanopolish.',required=True)
    required.add_argument('--summary', dest='summary', help='eventalign summary filepath, the output from nanopolish.',required=True)
    required.add_argument('--out_dir', dest='out_dir', help='output directory.',required=True)
    
    


    # Optional
    # Use ensembl db
    optional.add_argument('--ensembl', dest='ensembl', help='ensembl version for gene-transcript mapping.',type=int, default=91)
    optional.add_argument('--species', dest='species', help='species for ensembl gene-transcript mapping.', default='homo_sapiens')

    # Use customised db
    # These arguments will be passed to Genome from pyensembl
    optional.add_argument('--customised_genome', dest='customised_genome', help='customised_genome.',default=False,action='store_true')
    optional.add_argument('--reference_name', dest='reference_name', help='reference_name.',type=str)
    optional.add_argument('--annotation_name', dest='annotation_name', help='annotation_name.',type=str)
    optional.add_argument('--gtf_path_or_url', dest='gtf_path_or_url', help='gtf_path_or_url.',type=str)
    optional.add_argument('--transcript_fasta_paths_or_urls', dest='transcript_fasta_paths_or_urls', help='transcript_fasta_paths_or_urls.',type=str)


    # parser.add_argument('--features', dest='features', help='Signal features to extract.',type=list,default=['norm_mean'])
    optional.add_argument('--genome', dest='genome', help='to run on Genomic coordinates. Without this argument, the program will run on transcriptomic coordinates',default=False,action='store_true') 
    optional.add_argument('--n_processes', dest='n_processes', help='number of processes to run.',type=int, default=1)
    optional.add_argument('--chunk_size', dest='chunk_size', help='number of lines from nanopolish eventalign.txt for processing.',type=int, default=1000000)
    optional.add_argument('--readcount_min', dest='readcount_min', help='minimum read counts per gene.',type=int, default=1)
    optional.add_argument('--readcount_max', dest='readcount_max', help='maximum read counts per gene.',type=int, default=1000)
    optional.add_argument('--resume', dest='resume', help='with this argument, the program will resume from the previous run.',default=False,action='store_true') #todo

    parser._action_groups.append(optional)
    return parser.parse_args()

def partition_into_continuous_positions(arr, window_size=1):
    arr = arr[np.argsort(arr["transcriptomic_position"])]
    float_features = ['dwell_time', 'norm_std', 'norm_mean']
    float_dtypes = [('norm_mean', '<f8'), ('norm_std', '<f8'), ('dwell_time', '<f8')]
    
    float_arr = arr[float_features].astype(float_dtypes).view('<f8').reshape(-1, 3)
    kmer_arr = arr["reference_kmer"].reshape(-1, 1)
    tx_pos_arr = arr["transcriptomic_position"]
    tx_id_arr = arr["transcript_id"]

    partitions = [list(map(itemgetter(0), g)) for k, g in groupby(enumerate(tx_pos_arr), 
                                                                  lambda x: x[0] - x[1])]
    return [(float_arr[partition],
             kmer_arr[partition], tx_id_arr[partition], tx_pos_arr[partition]) 
            for partition in partitions if len(partition) > 2 * window_size + 1]

def filter_by_kmer(partition, kmers, window_size):
    feature_arr, kmer_arr, tx_id_arr, tx_pos_arr = partition
    kmers_5 = kmer_arr[:, (2 * window_size + 1) // 2]
    mask = np.isin(kmers_5, kmers)
    filtered_feature_arr = feature_arr[mask, :]
    filtered_kmer_arr = kmer_arr[mask, :]
    filtered_tx_pos_arr = tx_pos_arr[mask]
    filtered_tx_id_arr = tx_id_arr[mask]

    if len(filtered_kmer_arr) == 0:
        return []
    else:
        return filtered_feature_arr, filtered_kmer_arr, filtered_tx_id_arr, filtered_tx_pos_arr

def filter_partitions(partitions, window_size, kmers):
    windowed_partition = [create_features(partition, window_size) for partition in partitions]
    filtered_by_kmers = [filter_by_kmer(partition, kmers, window_size) for partition in windowed_partition]
    final_partitions = [x for x in filtered_by_kmers if len(x) > 0]
    return final_partitions

def roll(to_roll, window_size=1):
    nex = np.concatenate([np.roll(to_roll, i, axis=0) for i in range(-1, - window_size - 1, -1)],
                          axis=1)
    prev = np.concatenate([np.roll(to_roll, i, axis=0) for i in range(window_size, 0, -1)], axis=1)
    return np.concatenate((prev, to_roll, nex), axis=1)[window_size: -window_size, :]

def create_features(partition, window_size=1):
    float_arr, kmer_arr, tx_id_arr, tx_pos_arr = partition
    return roll(float_arr, window_size), roll(kmer_arr, window_size), \
        tx_id_arr[window_size: -window_size], tx_pos_arr[window_size: -window_size]

def filter_events(events, window_size, kmers):
    events = partition_into_continuous_positions(events)
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
   with locks['index'], open(out_paths['index'],'a') as f_index:
       for index in list(dict.fromkeys(eventalign_result.index)):
           transcript_id,read_index = index
           pos_end += eventalign_result.loc[index]['line_length'].sum()
           f_index.write('%s,%d,%d,%d\n' %(transcript_id,read_index,pos_start,pos_end))
           pos_start = pos_end

def parallel_index(eventalign_filepath,summary_filepath,chunk_size,out_dir,n_processes,resume):
    # Create output paths and locks.
    out_paths,locks = dict(),dict()
    for out_filetype in ['index']:
        out_paths[out_filetype] = os.path.join(out_dir,'eventalign.%s' %out_filetype)
        locks[out_filetype] = multiprocessing.Lock()
    # TO DO: resume functionality for index creation
        
    read_names_done = []
    if resume and os.path.exists(out_paths['log']):
        read_names_done = [line.rstrip('\n') for line in open(out_paths['log'],'r')]
    else:
        # Create empty files.
        with open(out_paths['index'],'w') as f:
            f.write('transcript_id,read_index,pos_start,pos_end\n') # header


    # Create communication queues.
    task_queue = multiprocessing.JoinableQueue(maxsize=n_processes * 2)

    # Create and start consumers.
    consumers = [helper.Consumer(task_queue=task_queue,task_function=index,locks=locks) for i in range(n_processes)]
    for p in consumers:
        p.start()
        
    ## Load tasks into task_queue. A task is eventalign information of one read.
    eventalign_file = open(eventalign_filepath,'r')
    pos_start = len(eventalign_file.readline()) #remove header
    chunk_split = None
    index_features = ['contig','read_index','line_length']
    for chunk in pd.read_csv(eventalign_filepath, chunksize=chunk_size,sep='\t'):
        chunk_complete = chunk[chunk['read_index'] != chunk.iloc[-1]['read_index']]
        chunk_concat = pd.concat([chunk_split,chunk_complete])
        chunk_concat_size = len(chunk_concat.index)
        ## read the file at where it left off because the file is opened once ##
        lines = [len(eventalign_file.readline()) for i in range(chunk_concat_size)]
        chunk_concat['line_length'] = np.array(lines)
        task_queue.put((chunk_concat[index_features],pos_start,out_paths))
        pos_start += sum(lines)
        chunk_split = chunk[chunk['read_index'] == chunk.iloc[-1]['read_index']]
    ## the loop above leaves off w/o adding the last read_index to eventalign.index
    chunk_split_size = len(chunk_split.index)
    lines = [len(eventalign_file.readline()) for i in range(chunk_split_size)]
    chunk_split['line_length'] = np.array(lines)
    task_queue.put((chunk_split[index_features],pos_start,out_paths))

    # Put the stop task into task_queue.
    task_queue = helper.end_queue(task_queue,n_processes)

    # Wait for all of the tasks to finish.
    task_queue.join()

def combine(events_str):
    f_string = StringIO(events_str)
    eventalign_result = pd.read_csv(f_string,delimiter='\t',names=['contig','position','reference_kmer','read_index','strand','event_index','event_level_mean','event_stdv','event_length','model_kmer','model_mean','model_stdv','standardized_level','start_idx','end_idx'])
    f_string.close()
    cond_successfully_eventaligned = eventalign_result['reference_kmer'] == eventalign_result['model_kmer']
    if cond_successfully_eventaligned.sum() != 0:

        eventalign_result = eventalign_result[cond_successfully_eventaligned]

        keys = ['read_index','contig','position','reference_kmer'] # for groupby
        eventalign_result['length'] = pd.to_numeric(eventalign_result['end_idx'])-pd.to_numeric(eventalign_result['start_idx'])
        eventalign_result['sum_norm_mean'] = pd.to_numeric(eventalign_result['event_level_mean']) * eventalign_result['length']
        eventalign_result['sum_norm_std'] = pd.to_numeric(eventalign_result['event_stdv']) * eventalign_result['length']
        eventalign_result['sum_dwell_time'] = pd.to_numeric(eventalign_result['event_length']) * eventalign_result['length']
            
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
        eventalign_result.reset_index(inplace=True)


        eventalign_result['transcript_id'] = [contig.split('.')[0] for contig in eventalign_result['contig']]    #### CHANGE MADE ####
        #eventalign_result['transcript_id'] = eventalign_result['contig']

        eventalign_result['transcriptomic_position'] = pd.to_numeric(eventalign_result['position']) + 2 # the middle position of 5-mers.
        # eventalign_result = misc.str_encode(eventalign_result)
#         eventalign_result['read_id'] = [read_name]*len(eventalign_result)

        # features = ['read_id','transcript_id','transcriptomic_position','reference_kmer','norm_mean','start_idx','end_idx']
        # features_dtype = np.dtype([('read_id', 'S36'), ('transcript_id', 'S15'), ('transcriptomic_position', '<i8'), ('reference_kmer', 'S5'), ('norm_mean', '<f8'), ('start_idx', '<i8'), ('end_idx', '<i8')])
        
#         features = ['transcript_id','transcriptomic_position','reference_kmer','norm_mean']

#         df_events = eventalign_result[['read_index']+features]
#         # print(df_events.head())

        features = ['transcript_id','read_index','transcriptomic_position','reference_kmer','norm_mean','norm_std','dwell_time']
#        np_events = eventalign_result[features].reset_index().values.ravel().view(dtype=[('transcript_id', 'S15'), ('transcriptomic_position', '<i8'), ('reference_kmer', 'S5'), ('norm_mean', '<f8')])
        df_events = eventalign_result[features]
        np_events = np.rec.fromrecords(df_events, names=[*df_events])
        return np_events
    else:
        return np.array([])


def parallel_preprocess_tx(eventalign_filepath,out_dir,n_processes,readcount_min,readcount_max,resume):
    
    # Create output paths and locks.
    out_paths,locks = dict(),dict()
    for out_filetype in ['json','index','log','readcount']:
        out_paths[out_filetype] = os.path.join(out_dir,'data.%s' %out_filetype)
        locks[out_filetype] = multiprocessing.Lock()
                
    # Writing the starting of the files.
    gene_ids_done = []
    if resume and os.path.exists(out_paths['index']):
        df_index = pd.read_csv(out_paths['index'],sep=',')
        gene_ids_done = list(df_index['idx'].unique())
    else:
        # with open(out_paths['json'],'w') as f:
        #     f.write('{\n')
        #     f.write('"genes":{')
        open(out_paths['json'],'w').close()
        with open(out_paths['index'],'w') as f:
            f.write('transcript_id,transcript_position,start,end\n') # header
        with open(out_paths['readcount'],'w') as f:
            f.write('transcript_id,transcript_position,n_reads\n') # header
        open(out_paths['log'],'w').close()

    # Create communication queues.
    task_queue = multiprocessing.JoinableQueue(maxsize=n_processes * 2)

    # Create and start consumers.
    consumers = [helper.Consumer(task_queue=task_queue,task_function=preprocess_tx,locks=locks) for i in range(n_processes)]
    for p in consumers:
        p.start()
    
    df_eventalign_index = pd.read_csv(os.path.join(out_dir,'eventalign.index'))
    df_eventalign_index['transcript_id'] = [tx_id.split('.')[0] for tx_id in  df_eventalign_index['transcript_id']]
    tx_ids = df_eventalign_index['transcript_id'].values.tolist()
    tx_ids = list(dict.fromkeys(tx_ids))
    df_eventalign_index.set_index('transcript_id',inplace=True)
    with open(eventalign_filepath,'r') as eventalign_result:
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
                task_queue.put((tx_id,data_dict,out_paths)) # Blocked if necessary until a free slot is available. 


    # Put the stop task into task_queue.
    task_queue = helper.end_queue(task_queue,n_processes)

    # Wait for all of the tasks to finish.
    task_queue.join()
    
##    with open(out_paths['log'],'a+') as f:
##        f.write('Total %d genes.\n' %len(gene_ids_processed))
##        f.write(helper.decor_message('successfully finished'))

def preprocess_tx(tx_id,data_dict,out_paths,locks):  # todo
    """
    Convert transcriptomic to genomic coordinates for a gene.
    
    Parameters
    ----------
        tx_id: str
            Transcript ID.
        data_dict: {read_id:events_array}
            Events for each read.
        features: [str] # todo
            A list of features to collect from the reads that are aligned to each genomic coordinate in the output.
    Returns
    -------
    dict
        A dict of all specified features collected for each genomic coordinate.
    """
    
    # features = ['read_id','transcript_id','transcriptomic_position','reference_kmer','norm_mean','start_idx','end_idx'] # columns in the eventalign file per read.

    events = []
    condition_labels = []
    run_labels = []
    read_ids = []
    transcriptomic_coordinates = []
    
    # Concatenate
    if len(data_dict) == 0:
        return
    
    features_arrays = []
    reference_kmer_arrays = []
    transcript_id_arrays = []
    transcriptomic_positions_arrays = []

    for read_id,events_per_read in data_dict.items(): 
        # print(read_id)
        events_per_read = filter_events(events_per_read, NUM_NEIGHBORING_FEATURES, M6A_KMERS)
        for event_per_read in events_per_read:
            features_arrays.append(event_per_read[0])
            reference_kmer_arrays.append([combine_sequence(kmer) for kmer in event_per_read[1]])
            transcript_id_arrays.append(event_per_read[2])
            transcriptomic_positions_arrays.append(event_per_read[3])


    if len(features_arrays) == 0:
        return
    else:
        features_arrays = np.concatenate(features_arrays)
        reference_kmer_arrays = np.concatenate(reference_kmer_arrays)
        transcript_id_arrays = np.concatenate(transcript_id_arrays)
        transcriptomic_positions_arrays = np.concatenate(transcriptomic_positions_arrays)
        assert(len(features_arrays) == len(reference_kmer_arrays) == len(transcript_id_arrays) == len(transcriptomic_positions_arrays))
    # Sort and split

    idx_sorted = np.lexsort((reference_kmer_arrays,transcriptomic_positions_arrays,transcript_id_arrays))
    key_tuples, index = np.unique(list(zip(transcript_id_arrays[idx_sorted],transcriptomic_positions_arrays[idx_sorted],
                                           reference_kmer_arrays[idx_sorted])),return_index = True,axis=0) #'chr',
    features_arrays = np.split(features_arrays[idx_sorted], index[1:])
    reference_kmer_arrays = np.split(reference_kmer_arrays[idx_sorted], index[1:])

    # read_id_arrays = np.split(events['read_index'][idx_sorted], index[1:]) ####
    # idx_sorted = np.lexsort((events['reference_kmer'],events['transcriptomic_position'],events['transcript_id']))
    # key_tuples, index = np.unique(list(zip(events['transcript_id'][idx_sorted],events['transcriptomic_position'][idx_sorted],events['reference_kmer'][idx_sorted])),return_index = True,axis=0) #'chr',
    # features = ["dwell_time", "norm_std", "norm_mean"]

    # features_arrays = np.split(events[features][idx_sorted], index[1:])

    # x_arrays = np.split(events['norm_std'][idx_sorted], index[1:])
    # y_arrays = np.split(events['norm_mean'][idx_sorted], index[1:])
    # z_arrays = np.split(events['dwell_time'][idx_sorted], index[1:])


    # Prepare
    # print('Reformating the data for each genomic position ...')
    data = defaultdict(dict)


    # for each position, make it ready for json dump
    for key_tuple, features_array, reference_kmer_array in zip(key_tuples, features_arrays, reference_kmer_arrays):
        _,position,kmer = key_tuple
        position = int(position)
        kmer = kmer
        if (len(set(reference_kmer_array)) == 1) and ('XXXXX' in set(reference_kmer_array)) or (len(features_array) == 0):
            continue
        ####Hi Chris,
        #####Can you figure out how to output a tuple with mean (y_array),sd(x_array),and dwell time(z_array), please?
        data[position] = {kmer: features_array.tolist()}

    # write to file.
    log_str = '%s: Data preparation ... Done.' %(tx_id)
    with locks['json'], open(out_paths['json'],'a') as f, \
            locks['index'], open(out_paths['index'],'a') as g, \
            locks['readcount'], open(out_paths['readcount'],'a') as h:
        
        for pos, dat in data.items():
            pos_start = f.tell()
            f.write('{')
            f.write('"%s":{"%d":' %(tx_id,pos))
            ujson.dump(dat, f)
            f.write('}}\n')
            pos_end = f.tell()
        
            # with locks['index'], open(out_paths['index'],'a') as f:
            g.write('%s,%d,%d,%d\n' %(tx_id,pos,pos_start,pos_end))
        
            # with locks['readcount'], open(out_paths['readcount'],'a') as f: #todo: repeats no. of tx >> don't want it.
            n_reads = 0
            for kmer, features in dat.items():
                n_reads += len(features)
            h.write('%s,%d,%d\n' %(tx_id,pos,n_reads))
        
    with locks['log'], open(out_paths['log'],'a') as f:
        f.write(log_str + '\n')
        

def main():
    args = get_args()
    #
    n_processes = args.n_processes        
    eventalign_filepath = args.eventalign
    summary_filepath = args.summary
    chunk_size = args.chunk_size
    out_dir = args.out_dir
    ensembl_version = args.ensembl
    ensembl_species = args.species
    readcount_min = args.readcount_min    
    readcount_max = args.readcount_max
    resume = args.resume
    genome = args.genome

    customised_genome = args.customised_genome
    if customised_genome and (None in [args.reference_name,args.annotation_name,args.gtf_path_or_url,args.transcript_fasta_paths_or_urls]):
        print('If you have your own customised genome not in Ensembl, please provide the following')
        print('- reference_name')
        print('- annotation_name')
        print('- gtf_path_or_url')
        print('- transcript_fasta_paths_or_urls')
    else:
        reference_name = args.reference_name
        annotation_name = args.annotation_name
        gtf_path_or_url = args.gtf_path_or_url
        transcript_fasta_paths_or_urls = args.transcript_fasta_paths_or_urls
        
    misc.makedirs(out_dir) #todo: check every level.
    
    # (1) For each read, combine multiple events aligned to the same positions, the results from nanopolish eventalign, into a single event per position.
    eventalign_log_filepath = os.path.join(out_dir,'eventalign.log')
    # if not helper.is_successful(eventalign_log_filepath) and not resume: #some slight hack to skip index creation again after it is successful
    parallel_index(eventalign_filepath,summary_filepath,chunk_size,out_dir,n_processes,resume)
    parallel_preprocess_tx(eventalign_filepath,out_dir,n_processes,readcount_min,readcount_max,False) #TO DO: RESUME FUNCTION

if __name__ == '__main__':
    main()
