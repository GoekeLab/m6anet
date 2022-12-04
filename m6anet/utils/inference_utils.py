import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from multiprocessing import Pool


def inference(model, dl, device, read_proba_threshold):
    """
    Run inference on unlabelled dataset
    """
    model.eval()
    kmer_maps = dl.dataset.int_to_kmer

    with torch.no_grad():
        read_probs, read_lengths, read_ids, all_kmers = [], [], [], []
        for data in iter(dl):
            features, kmers, n_reads, read_id = data
            features = model.get_read_representation({'X': features.to(device), 'kmer': kmers.to(device)})
            probs = model.pooling_filter.probability_layer(features).flatten()
            all_kmers.append([kmer_maps[_kmer.item()] for _kmer in kmers[:, kmers.shape[-1] // 2]])

            read_probs.append(probs.detach().cpu().numpy())
            read_lengths.append(n_reads.numpy())
            read_ids.append(read_id)

        read_probs, read_ids, all_kmers = group_results(np.concatenate(read_probs),
                                                        np.concatenate(read_ids),
                                                        np.concatenate(all_kmers),
                                                        np.concatenate(read_lengths))
        return np.array([np.mean(x >= read_proba_threshold) for x in read_probs]), read_probs, read_ids, all_kmers


def _calculate_site_proba(task):
    tx_id, tx_pos, proba, n_iters, n_samples = task
    proba = np.random.choice(proba, n_iters * n_samples, replace=True).reshape(n_iters, n_samples)
    proba = (1 - np.prod(1 - proba, axis=1)).mean()
    return tx_id, tx_pos, proba


def calculate_site_proba(indiv_proba_results, n_iters, n_samples, n_processes):
    tasks = ((tx_id, tx_pos, sub_df['probability_modified'].values, n_iters, n_samples) for (tx_id, tx_pos), sub_df in indiv_proba_results.groupby(["transcript_id", "transcript_position"]))
    with Pool(n_processes) as p:
        results = [x for x in p.imap(_calculate_site_proba, tasks)]
    return pd.DataFrame(results, columns=["transcript_id", "transcript_position", "probability_modified"])



def group_results(read_probs, read_ids, kmers, n_reads):
    i = 0
    grouped_probs = []
    grouped_ids = []
    grouped_kmers = []

    for n_read in n_reads:
        grouped_probs.append(read_probs[i: i + n_read])
        grouped_ids.append(read_ids[i: i + n_read])
        grouped_kmers.append(kmers[i])
        i += n_read
    return grouped_probs, grouped_ids, grouped_kmers
