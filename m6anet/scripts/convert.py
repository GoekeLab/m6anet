import os
import pandas as pd
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    # Required arguments
    parser.add_argument("--input_dir",
                        help='directories containing data.readcount and data.index.',
                        required=True)
    parser.add_argument("--out_dir",
                        help='directory to output inference results.',
                        required=True)
    return parser


def main(args):

    input_dir = args.input_dir
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_index = pd.read_csv(os.path.join(input_dir, "data.index"))
    data_readcount = pd.read_csv(os.path.join(input_dir, "data.readcount"))

    data_info = data_readcount.merge(data_index, on=["transcript_id", "transcript_position"])
    data_info.to_csv(os.path.join(out_dir, "data.info"), index=False)
