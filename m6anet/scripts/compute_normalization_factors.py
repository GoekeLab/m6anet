import argparse
import os 
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import joblib
import json


def get_mean_std(task):
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


def read_kmer(task):
    data_fpath, tx_id, tx_pos, start_pos, end_pos = task
    with open(data_fpath) as f:
        f.seek(start_pos, 0)
        json_str = f.read(end_pos - start_pos)
        pos_info = json.loads(json_str)[tx_id][str(tx_pos)]

        assert(len(pos_info.keys()) == 1)

        kmer, _ = list(pos_info.items())[0]
        return kmer


def read_features(data_fpath, tx_id, tx_pos, start_pos, end_pos):
    with open(data_fpath) as f:
        f.seek(start_pos, 0)
        json_str = f.read(end_pos - start_pos)
        pos_info = json.loads(json_str)[tx_id][str(tx_pos)]

        assert(len(pos_info.keys()) == 1)

        _, features = list(pos_info.items())[0]
        return np.array(features)


def annotate_kmer_information(data_fpath, index_df, n_processes):
    tasks = ((data_fpath, tx_id, tx_pos, start, end) for tx_id, tx_pos, start, end in zip(index_df["transcript_id"], index_df["transcript_position"],
                                                                                          index_df["start"], index_df["end"]))
    with Pool(n_processes) as p:
        kmer_info = [x for x in tqdm(p.imap(read_kmer, tasks), total=len(index_df))]
    index_df["kmer"] = kmer_info
    return index_df


def create_kmer_mapping_df(merged_df):
    kmer_mapping_df = []
    for _, row in merged_df.iterrows():
        tx, tx_position, sequence, start, end, n_reads = row["transcript_id"], row["transcript_position"], row["kmer"], row["start"], row["end"], row["n_reads"]
        kmers = [sequence[i:i+5] for i in range(len(sequence) - 4)]
        for i in range(len(kmers)):
            kmer_mapping_df += [(tx, tx_position, start, end, n_reads, kmers[i], i)]

    kmer_mapping_df = pd.DataFrame(kmer_mapping_df, 
                                   columns=["transcript_id", "transcript_position", "start", "end", "n_reads", "kmer", "segment_number"])
    return kmer_mapping_df


def create_norm_dict(kmer_mapping_df, data_fpath, n_processes):
    tasks = ((kmer, df, data_fpath) for kmer, df in kmer_mapping_df.groupby("kmer"))
    with Pool(n_processes) as p:
        norm_dict = [x for x in tqdm(p.imap_unordered(get_mean_std, tasks), total=len(kmer_mapping_df["kmer"].unique()))]
    return {tup[0]: (tup[1], tup[2]) for tup in norm_dict}


def main():
    parser = argparse.ArgumentParser(description="a script to compute normalization factors from training set")
    parser.add_argument('-i', '--input_dir', dest='input_dir', default=None,
                        help="Input directory containing the data.readcount file")
    parser.add_argument('-o', '--output', dest='save_dir', default=None,
                        help="Output directory for the annotated data.readcount file")                   
    parser.add_argument('-n', '--n_jobs', dest='n_jobs', default=1, type=int,
                        help="Number of processors to run the dataloader")
    args = parser.parse_args()
    
    input_dir = args.input_dir
    save_dir = args.save_dir
    n_processes = args.n_jobs  

    data_fpath = os.path.join(input_dir, "data.json")
    info_df = pd.read_csv(os.path.join(input_dir, "data.readcount.labelled"))
    info_df = info_df[info_df["set_type"] == 'Train']
    index_df = pd.read_csv(os.path.join(input_dir, "data.index"))
    
    merged_df = info_df.merge(index_df, on=["transcript_id", "transcript_position"])
    merged_df = annotate_kmer_information(data_fpath, merged_df, n_processes)


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    kmer_mapping_df = create_kmer_mapping_df(merged_df)
    norm_dict = create_norm_dict(kmer_mapping_df, data_fpath, n_processes)

    joblib.dump(norm_dict, os.path.join(save_dir, "norm_dict_nanopolish.joblib"))

if __name__ == '__main__':
    main()

