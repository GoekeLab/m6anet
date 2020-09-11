import argparse
import torch
import pandas as pd
import numpy as np
import os
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from itertools import product
from ..utils.data_utils import ValDS, custom_collate_val
from ..utils.model import MultiInstanceNNEmbedding


def extract_positions(tx_dir):
    fnames = os.listdir(tx_dir)
    fpaths = [os.path.join(tx_dir, fname) for fname in fnames]
    transcripts = [x.split("_")[0] for x in fnames]
    positions = [np.array(x.split("_"))[1] for x in fnames]
    n_samples = [int(x.split("_")[-1].split(".hdf5")[0]) for x in fnames]
    kmers = [x.split("_")[2] for x in fnames]
    return pd.DataFrame({'filepath': fpaths, 'position': positions, 'transcript_id': transcripts,
                         'n_samples': n_samples,
                         'kmer': kmers})


def get_args():
    parser = argparse.ArgumentParser(description="a script to preprocess all files in a nanopolish event align directory")
    parser.add_argument('-i', '--input', dest='input_dir', default=None,
                        help="Directory containing the inference folder to predict on")
    parser.add_argument('-o', '--output', dest='out_dir', default=None,
                        help="Save directory for the prediction results")
    parser.add_argument('-m', '--model', dest='model_path', default=None,
                        help="Path to directory containing norm constant and state dictionary for torch model")
    parser.add_argument('-d', '--device', dest='device', default='cpu',
                        help="cpu or cuda to run the inference on")
    parser.add_argument('-n', '--n_processors', dest='n_processors', default=None,
                        help="number of workers for dataloader")

    return parser.parse_args()


def main():

    args = get_args()
    device = args.device

    norm_constant = pd.read_csv(os.path.join(args.model_path, "norm_constant.csv"))    
    model = MultiInstanceNNEmbedding(dim_cov=3, p=8, embedding_dim=2).to(device)
    state_dict = torch.load(os.path.join(args.model_path, "best_model.pt"), map_location=device)
    model.load_state_dict(state_dict)
    
    data_dir = os.path.join(args.input_dir, "inference")
    df = extract_positions(data_dir)
    ds = ValDS(norm_constant, data_dir)
    dl = DataLoader(ds, num_workers=int(args.n_processors), batch_size=150,
                    shuffle=False, collate_fn=custom_collate_val)
    y_preds = []
    
    for _, inp in tqdm(enumerate(dl), total=len(dl)):
        out = model(inp[0].to(device), inp[1].to(device), inp[2])
        
        y_preds.extend(out.detach().cpu().numpy())

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    df["proba"] = np.array(y_preds)
    df[["transcript_id", "kmer", "n_samples", "proba", "filepath"]]\
        .to_csv(os.path.join(args.out_dir, "m6Anet_predictions.csv.gz"), index=False)


if __name__ == '__main__':
    main()

