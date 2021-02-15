import argparse
import os 
import numpy as np
import pandas as pd
import joblib
from .compute_normalization_factors import annotate_kmer_information, create_kmer_mapping_df, create_norm_dict
from tqdm import tqdm
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def main():
    parser = argparse.ArgumentParser(description="a script to compute normalization factors from training set")
    parser.add_argument('-i', '--input_dir', dest='input_dir', default=None,
                        help="Input directory containing the data.readcount file")
    parser.add_argument('-o', '--output_dir', dest='cv_dir', default=None,
                        help="Output directory for the annotated data.readcount file")     
    parser.add_argument('-cv', '--cv', dest='cv', default=5, type=int,
                        help="Number of cross validation folds")              
    parser.add_argument('-n', '--n_jobs', dest='n_jobs', default=1, type=int,
                        help="Number of processors to run the dataloader")
    args = parser.parse_args()
    
    input_dir = args.input_dir
    save_dir = args.cv_dir
    cv = args.cv
    n_processes = args.n_jobs

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    data_fpath = os.path.join(input_dir, "data.json")
    info_df = pd.read_csv(os.path.join(input_dir, "data.readcount.labelled"))
    index_df = pd.read_csv(os.path.join(input_dir, "data.index"))

    info_df = info_df[info_df["n_reads"] >= 20].reset_index(drop=True) # Filtering for positions with low number of reads
    info_df = info_df.merge(index_df, on=["transcript_id", "transcript_position"])
    info_df = annotate_kmer_information(data_fpath, info_df, n_processes) # Adding k-mer information for later use
    
    # KFold cross validation on gene level
    all_sites = np.arange(len(info_df))
    modification_status = info_df["modification_status"]
    all_genes = info_df["gene_id"]
    fold_num = 1

    for train_test_index, val_index in tqdm(GroupKFold(n_splits=cv).split(all_sites, modification_status, groups=all_genes), total=cv,
                                            desc="Creating cross validation split"):
        
        train_test_sites, val_sites = all_sites[train_test_index], all_sites[val_index]
        # Further split train set into train and test
        train_test_group = info_df.iloc[train_test_sites]["gene_id"]
        train_index, test_index = next(GroupShuffleSplit(n_splits=10).split(train_test_sites, groups=train_test_group))
        train_sites, test_sites = train_test_sites[train_index], train_test_sites[test_index]
        
        info_df.loc[train_sites, "set_type"] = np.repeat("Train", len(train_sites))
        info_df.loc[test_sites, "set_type"] = np.repeat("Test", len(test_sites))
        info_df.loc[val_sites, "set_type"] = np.repeat("Val", len(val_sites))
        
        fold_dir = os.path.join(save_dir, str(fold_num))

        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)
        columns = ["transcript_id", "transcript_position", "n_reads", "chr", "gene_id", "genomic_position", "kmer", "modification_status", "set_type"]
        info_df[columns].to_csv(os.path.join(fold_dir, "data.readcount.labelled"), index=False)
        
        info_df_train = info_df[info_df["set_type"] == 'Train'].reset_index(drop=True)
        kmer_mapping_df = create_kmer_mapping_df(info_df_train)
        norm_dict = create_norm_dict(kmer_mapping_df, data_fpath, n_processes)
        joblib.dump(norm_dict, os.path.join(fold_dir, "norm_dict.joblib"))
        fold_num += 1


if __name__ == '__main__':
    main()

