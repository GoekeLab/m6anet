import os
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
import pandas as pd
import joblib
from ..utils.norm_utils import annotate_kmer_information, create_kmer_mapping_df, \
        create_norm_dict


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument('--input_dir',
                        help="Input directory containing the data.info.labelled file and data.json file",
                        default=None)
    parser.add_argument('--out_dir',
                        help="Output directory for the normalization file",
                        default=None)
    parser.add_argument('--n_processes',
                        help="Number of processors to run the dataloader",
                        default=1, type=int)
    return parser


def main(args):
    input_dir = args.input_dir
    out_dir = args.out_dir
    n_processes = args.n_processes

    data_fpath = os.path.join(input_dir, "data.json")
    info_df = pd.read_csv(os.path.join(input_dir, "data.info.labelled"))
    info_df = info_df[info_df["set_type"] == 'Train']
    info_df["transcript_position"] = info_df["transcript_position"].astype('int')
    info_df = annotate_kmer_information(data_fpath, info_df, n_processes)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    kmer_mapping_df = create_kmer_mapping_df(info_df)
    norm_dict = create_norm_dict(kmer_mapping_df, data_fpath, n_processes)
    joblib.dump(norm_dict, os.path.join(out_dir, "norm_dict_nanopolish.joblib"))
