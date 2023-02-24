import warnings
import os
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
import pandas as pd
from ..utils.constants import NUM_NEIGHBORING_FEATURES
from ..utils.dataprep_utils import parallel_index, parallel_preprocess_tx


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )

    # Required arguments
    parser.add_argument('--eventalign',
                        help='eventalign filepath, the output from nanopolish.',
                        required=True)

    parser.add_argument('--out_dir',
                        help='output directory.',
                        required=True)

    # Optional arguments

    parser.add_argument('--n_processes',
                        help='number of processes to run.',
                        default=1, type=int)
    parser.add_argument('--chunk_size',
                        help='number of lines from nanopolish eventalign.txt for processing.',
                        default=1000000, type=int)
    parser.add_argument('--readcount_min',
                        help='minimum read counts per gene',
                        default=1, type=int)
    parser.add_argument('--readcount_max',
                        help='maximum read counts per gene',
                        default=1000, type=int)
    parser.add_argument('--min_segment_count',
                        help='minimum read counts per candidate segment.',
                        default=20, type=int)
    parser.add_argument('--skip_index',
                        help='with this argument the program will skip indexing eventalign.txt first.',
                        default=False, action='store_true')
    parser.add_argument('--n_neighbors',
                        help='number of neighboring features to extract.',
                        default=NUM_NEIGHBORING_FEATURES, type=int)
    parser.add_argument('--compress',
                        help='number of neighboring features to extract.',
                        default=False, action='store_true')
    return parser


def main(args):

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    if not args.skip_index:
        parallel_index(args.eventalign, args.chunk_size,
                       args.out_dir, args.n_processes)

    # For each read, combine multiple events aligned to the same positions,
    # the results from nanopolish eventalign, into a single event per position.

    parallel_preprocess_tx(args.eventalign, args.out_dir, args.n_processes,
                           args.readcount_min, args.readcount_max, args.n_neighbors,
                           args.min_segment_count, args.compress)
