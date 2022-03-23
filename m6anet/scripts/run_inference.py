import os
import torch
import toml
import pkg_resources
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy
from ..model.model import MILModel
from ..utils.builder import random_fn
from ..utils.data_utils import NanopolishDS, NanopolishReplicateDS, inference_collate
from ..utils.training_utils import inference
from torch.utils.data import DataLoader


DEFAULT_MODEL_CONFIG = pkg_resources.resource_filename('m6anet.model', 'configs/model_configs/prod_pooling.toml')
DEFAULT_MODEL_WEIGHTS = pkg_resources.resource_filename('m6anet.model', 'model_states/prod_pooling_pr_auc.pt')
NORM_PATH = pkg_resources.resource_filename('m6anet.model', 'norm_factors/norm_dict.joblib')
MIN_READS = 20

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
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--n_processes", default=25, type=int)
    parser.add_argument("--num_iterations", default=5, type=int)
    parser.add_argument("--device", default='cpu', type=str)
    return parser


def run_inference(args):
    
    input_dir = args.input_dir
    out_dir = args.out_dir

    model = MILModel(toml.load(args.model_config)).to(args.device)
    model.load_state_dict(torch.load(args.model_state_dict, map_location=torch.device(args.device)))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if len(input_dir) == 1:
        input_dir = input_dir[0]
        ds = NanopolishDS(input_dir, MIN_READS, args.norm_path, mode='Inference')
    else:
        ds = NanopolishReplicateDS(input_dir, MIN_READS, args.norm_path, mode='Inference')

    dl = DataLoader(ds, num_workers=args.n_processes, collate_fn=inference_collate, batch_size=args.batch_size, worker_init_fn=random_fn, shuffle=False)
    result_df = ds.data_info[["transcript_id", "transcript_position", "n_reads"]].copy(deep=True)   
    results, kmers = inference(model, dl, args.device, args.num_iterations)
    result_df["probability_modified"] = results 
    result_df["kmer"] = kmers
    result_df.to_csv(os.path.join(out_dir, "data.result.csv.gz"), index=False)
    
def main():
    args = argparser().parse_args()
    run_inference(args)


if __name__ == '__main__':
    main()
