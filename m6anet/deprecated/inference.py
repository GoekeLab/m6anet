import os
import torch
import toml
import warnings
import numpy as np
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from ..model.model import MILModel
from ..utils.constants import DEFAULT_MODEL_CONFIG, DEFAULT_MODEL_WEIGHTS,\
    DEFAULT_NORM_PATH, DEFAULT_MIN_READS, DEFAULT_READ_THRESHOLD
from ..utils.data_utils import NanopolishDS, NanopolishReplicateDS, inference_collate
from ..utils.inference_utils import run_inference
from torch.utils.data import DataLoader


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    # Required arguments
    parser.add_argument("--input_dir", nargs="*",
                        help='directories containing data.info and data.json.',
                        required=True)
    parser.add_argument("--out_dir",
                        help='directory to output inference results.',
                        required=True)

    # Optional arguments

    parser.add_argument("--model_config",
                        help='path to model config file.',
                        default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--model_state_dict",
                        help='path to model weights.',
                        default=DEFAULT_MODEL_WEIGHTS)
    parser.add_argument("--norm_path",
                        help='path to normalization factors file',
                        default=DEFAULT_NORM_PATH)
    parser.add_argument("--batch_size",
                        help='batch size for inference.',
                        default=16, type=int)
    parser.add_argument("--save_per_batch",
                        help='saving inference results every save_per_batch multiples.',
                        default=2, type=int)
    parser.add_argument("--n_processes",
                        help='number of processes to run.',
                        default=25, type=int)
    parser.add_argument("--num_iterations",
                        help='number of sampling run.',
                        default=1000, type=int)
    parser.add_argument("--device",
                        help='device to perform inference with.',
                        default='cpu', type=str)
    parser.add_argument("--seed",
                        help='random seed for sampling.',
                        default=0, type=int)
    parser.add_argument("--infer_mod_rate", default=False, action='store_true',
                        help="m6Anet will always infer modification rate and this flag is going to be deprecated")
    parser.add_argument("--read_proba_threshold",
                        help='default probability threshold for a read to be considered modified',
                        default=DEFAULT_READ_THRESHOLD, type=float)
    return parser


def main():
    args = argparser().parse_args()
    warnings.warn('m6anet-run_inference is deprecated and going to be removed in the next release. Please use m6anet inference instead', DeprecationWarning, stacklevel=2)

    input_dir = args.input_dir
    out_dir = args.out_dir

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    model = MILModel(toml.load(args.model_config)).to(args.device)
    model.load_state_dict(torch.load(args.model_state_dict,
                                     map_location=torch.device(args.device)))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(args.out_dir, "data.site_proba.csv"),'w', encoding='utf-8') as f:
        f.write('transcript_id,transcript_position,n_reads,probability_modified,kmer,mod_ratio\n')
    with open(os.path.join(args.out_dir, "data.indiv_proba.csv"), 'w', encoding='utf-8') as g:
        g.write('transcript_id,transcript_position,read_index,probability_modified\n')

    if len(input_dir) == 1:
        ds = NanopolishDS(input_dir[0], DEFAULT_MIN_READS, args.norm_path, mode='Inference')
    else:
        ds = NanopolishReplicateDS(input_dir, DEFAULT_MIN_READS, args.norm_path, mode='Inference')

    dl = DataLoader(ds, num_workers=args.n_processes, collate_fn=inference_collate, batch_size=args.batch_size,
                    shuffle=False)
    run_inference(model, dl, args)

if __name__ == '__main__':
    main()