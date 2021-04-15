import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from pyensembl import Genome
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

def add_chromosome_and_gene_info(info_df):
    info_df["chr"] = info_df["transcript_id"].apply(lambda x: chr_map[genome.transcript_by_id(x).contig])
    info_df["gene_id"] = info_df["transcript_id"].apply(lambda x: genome.transcript_by_id(x).gene_id)
    return info_df


def _add_genomic_position(task):
    tx, tx_df = task
    gt_map = pd.read_csv(os.path.join(gt_dir, tx, "gt_mapping.csv.gz")).set_index("tx_pos")
    tx_df["genomic_position"] = gt_map["g_pos"].loc[tx_df["transcript_position"]].values
    tx_df["kmer"] = gt_map["kmer"].loc[tx_df["transcript_position"]].values
    return tx_df


def add_genomic_position(info_df, n_jobs=1):
    with Pool(n_jobs) as p:
        tasks = ((tx, df) for tx, df in info_df.groupby("transcript_id"))
        n_transcripts = len(info_df.transcript_id.unique())
        res_df = [x for x in tqdm(p.imap_unordered(_add_genomic_position, tasks), total=n_transcripts)]
    return pd.concat(res_df).reset_index(drop=True)


def get_y(info_df, table, col_name):
    g, chrsm = info_df["genomic_position"].values, info_df["chr"].values
    info_df[col_name] = np.array([1 if (g_pos, chr_id) in table.index else 0
                                               for g_pos, chr_id in tqdm(zip(g, chrsm), total=len(g),
                                               desc="Getting label information")]) 
    return info_df


def train_test_val_split(info_df):
    info_df["set_type"] = np.repeat("NA", len(info_df))
    np.random.seed(0)
    all_sites = np.arange(len(info_df))
    
    # Create validation set
    train_test_index, val_index = next(GroupShuffleSplit(n_splits=10).split(all_sites, groups=info_df["gene_id"]))
    train_test_sites, val_sites = all_sites[train_test_index], all_sites[val_index]
    
    # Further split train set into train and test
    train_test_group = info_df.iloc[train_test_sites]["gene_id"]
    train_index, test_index = next(GroupShuffleSplit(n_splits=10).split(train_test_sites, groups=train_test_group))
    train_sites, test_sites = train_test_sites[train_index], train_test_sites[test_index]
    
    info_df.loc[train_sites, "set_type"] = np.repeat("Train", len(train_sites))
    info_df.loc[test_sites, "set_type"] = np.repeat("Test", len(test_sites))
    info_df.loc[val_sites, "set_type"] = np.repeat("Val", len(val_sites))
    return info_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="a script to extract raw signals and event align features from tx hdf5 files")
    parser.add_argument('-i', '--input_dir', dest='input_dir', default=None,
                        help="Input directory containing the data.readcount file")
    parser.add_argument('-o', '--output', dest='save_dir', default=None,
                        help="Output directory for the annotated data.readcount file")                   
    parser.add_argument('-n', '--n_jobs', dest='n_jobs', default=1, type=int,
                        help="Number of processors to run the dataloader")
    args = parser.parse_args()

    chrsm_annot_dir = "/home/christopher/annotations/chrsm_annot.txt"
    genome = Genome(reference_name='GRCh38',
                annotation_name='my_genome_features',
                gtf_path_or_url='/home/christopher/annotations/Homo_sapiens.GRCh38.91.chr_patch_hapl_scaff.gtf',
                transcript_fasta_paths_or_urls='/home/christopher/annotations/Homo_sapiens.GRCh38.cdna.ncrna.fa') 
    chr_map = {}
    with open(chrsm_annot_dir) as f:
        for line in f:
            ensembl, ucsc = line.strip("\n").split("\t")
            chr_map[ensembl] = ucsc

    m6ace = pd.read_csv("/home/christopher/annotations/m6ACE_HEK293T.csv.gz").set_index(["End", "Chr"])
    miclip = pd.read_csv("/home/christopher/annotations/miclip_HEK293T.csv.gz").set_index(["End", "Chr"])

    input_dir = args.input_dir
    save_dir = args.save_dir
    n_processes = args.n_jobs   

    gt_dir = "/data03/christopher/gt_mapping_final/"
    info_df = pd.read_csv(os.path.join(input_dir, "data.readcount"))

    all_transcripts = set(genome.transcript_ids())

    info_df = info_df[info_df.transcript_id.apply(lambda x: x in all_transcripts)].reset_index(drop=True)
    info_df = add_chromosome_and_gene_info(info_df)
    info_df = add_genomic_position(info_df, n_processes)

    info_df = get_y(info_df, m6ace, "m6ACE")
    info_df = get_y(info_df, miclip, "miCLIP")
    info_df["modification_status"] = np.any(info_df[["m6ACE", "miCLIP"]], axis=1) * 1

    # Filtering sites with less than 20 reads
    info_df = info_df[info_df["n_reads"] >= 20].reset_index(drop=True)
    
    info_df = train_test_val_split(info_df)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    info_df.to_csv(os.path.join(save_dir, "data.readcount.labelled"), index=False)
