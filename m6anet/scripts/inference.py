import os
import torch
import toml
import pkg_resources
import pathlib
import numpy as np
import warnings
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from ..model.model import MILModel
from ..utils.constants import DEFAULT_MODEL_CONFIG,\
    DEFAULT_MIN_READS, DEFAULT_READ_THRESHOLD,\
    DEFAULT_NORM_PATH, PRETRAINED_CONFIGS,\
    DEFAULT_PRETRAINED_MODEL, DEFAULT_PRETRAINED_MODELS
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
    parser.add_argument("--pretrained_model",
                        help="pre-trained model available at m6anet. Options include {}.".format(DEFAULT_PRETRAINED_MODELS),
                        default=DEFAULT_PRETRAINED_MODEL, type=str)
    parser.add_argument("--model_config",
                        help='path to model config file.',
                        default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--model_state_dict",
                        help='path to model weights.',
                        default=None)
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
    parser.add_argument("--read_proba_threshold",
                        help='default probability threshold for a read to be considered modified.',
                        default=DEFAULT_READ_THRESHOLD, type=float)
    return parser


def main(args):

    input_dir = args.input_dir

    if args.model_state_dict is not None:
        warnings.warn("--model_state_dict is specified, overwriting default model weights")
    else:
        if args.pretrained_model not in DEFAULT_PRETRAINED_MODELS:
            raise ValueError("Invalid pretrained model {}, must be one of {}".format(args.pretrained_model, DEFAULT_PRETRAINED_MODELS))

        args.model_state_dict = PRETRAINED_CONFIGS[args.pretrained_model][0]
        args.read_proba_threshold = PRETRAINED_CONFIGS[args.pretrained_model][1]
        args.norm_path = PRETRAINED_CONFIGS[args.pretrained_model][2]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    model = MILModel(toml.load(args.model_config)).to(args.device)
    model.load_state_dict(torch.load(args.model_state_dict,
                                     map_location=torch.device(args.device)))

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

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
