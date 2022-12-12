import os
import torch
import toml
import pkg_resources
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from ..model.model import MILModel
from ..utils.data_utils import NanopolishDS, NanopolishReplicateDS, inference_collate
from ..utils.inference_utils import inference
from torch.utils.data import DataLoader


DEFAULT_MODEL_CONFIG = pkg_resources.resource_filename('m6anet.model', 'configs/model_configs/prod_pooling.toml')
DEFAULT_MODEL_WEIGHTS = pkg_resources.resource_filename('m6anet.model', 'model_states/prod_pooling_pr_auc.pt')
NORM_PATH = pkg_resources.resource_filename('m6anet.model', 'norm_factors/norm_dict.joblib')
MIN_READS = 20
DEFAULT_READ_THRESHOLD = 0.033379376

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("--input_dir", default=None, nargs="*", required=True)
    parser.add_argument("--out_dir", default=None, required=True)
    parser.add_argument("--model_config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--model_state_dict", default=DEFAULT_MODEL_WEIGHTS)
    parser.add_argument("--norm_path", default=NORM_PATH)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--save_per_batch", default=2, type=int)
    parser.add_argument("--n_processes", default=25, type=int)
    parser.add_argument("--num_iterations", default=5, type=int)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--read_proba_threshold", default=DEFAULT_READ_THRESHOLD, type=float)

    return parser


def run_inference(args):

    input_dir = args.input_dir
    out_dir = args.out_dir

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    model = MILModel(toml.load(args.model_config)).to(args.device)
    model.load_state_dict(torch.load(args.model_state_dict, map_location=torch.device(args.device)))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(args.out_dir, "data.site_proba.csv"),'w', encoding='utf-8') as f:
        f.write('transcript_id,transcript_position,n_reads,probability_modified,kmer,mod_ratio\n')
    with open(os.path.join(args.out_dir, "data.indiv_proba.csv"), 'w', encoding='utf-8') as g:
        g.write('transcript_id,transcript_position,read_index,probability_modified\n')

    if len(input_dir) == 1:
        ds = NanopolishDS(input_dir[0], MIN_READS, args.norm_path, mode='Inference')
    else:
        ds = NanopolishReplicateDS(input_dir, MIN_READS, args.norm_path, mode='Inference')

    dl = DataLoader(ds, num_workers=args.n_processes, collate_fn=inference_collate, batch_size=args.batch_size,
                    shuffle=False)
    inference(model, dl, args)


def main():
    args = argparser().parse_args()
    run_inference(args)


if __name__ == '__main__':
    main()
