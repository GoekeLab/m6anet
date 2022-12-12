import torch
import numpy as np
import pandas as pd
import os
from collections import OrderedDict
from multiprocessing import Pool


def inference(model, dl, args):
    """
    Run inference on unlabelled dataset
    """
    model.eval()
    kmer_maps = dl.dataset.int_to_kmer

    with open(os.path.join(args.out_dir, "data.site_proba.csv"), 'a', encoding='utf-8') as f:
        with open(os.path.join(args.out_dir, "data.indiv_proba.csv"), 'a', encoding='utf-8') as g:
            with torch.no_grad():
                read_probs, read_lengths, tx_ids, tx_positions, read_ids, all_kmers = \
                    [], [], [], [], [], []
                for it, data in enumerate(dl):
                    features, kmers, n_reads, tx_id, tx_position, read_id = data
                    features = model.get_read_representation({'X': features.to(args.device),
                                                              'kmer': kmers.to(args.device)})
                    probs = model.pooling_filter.probability_layer(features).flatten()

                    all_kmers.append([kmer_maps[_kmer.item()] for _kmer in kmers[:, kmers.shape[-1] // 2]])

                    read_probs.append(probs.detach().cpu().numpy())
                    read_lengths.append(n_reads.numpy())
                    tx_ids.append(tx_id)
                    tx_positions.append(tx_position)
                    read_ids.append(read_id)

                    if (it + 1) % args.save_per_batch:
                        read_probs, tx_ids, tx_positions, read_ids, all_kmers, n_reads = \
                            group_results(np.concatenate(read_probs), np.concatenate(tx_ids),
                                          np.concatenate(tx_positions), np.concatenate(read_ids),
                                          np.concatenate(all_kmers), np.concatenate(read_lengths))

                        mod_ratios = np.array([np.mean(x >= args.read_proba_threshold) for x in read_probs])
                        site_probs = calculate_site_proba(read_probs, args.num_iterations, 20, args.n_processes)

                        assert(len(read_probs) == len(tx_ids) == len(tx_positions) == len(read_ids) \
                                == len(all_kmers) == len(mod_ratios) == len(site_probs) == len(n_reads))

                        for tx_id, tx_pos, kmer, mod_ratio, site_prob, read_prob, read_id, n_read in \
                                zip(tx_ids, tx_positions, all_kmers, mod_ratios, site_probs,
                                    read_probs, read_ids, n_reads):
                                    f.write('%s,%d,%s,%.16f,%s,%.16f\n' % (tx_id, tx_pos, n_read,
                                                                           site_prob, kmer, mod_ratio))
                                    assert(len(read_prob) == len(read_id))
                                    for _read_prob, _read_id in zip(read_prob, read_id):
                                        g.write('%s,%d,%s,%.16f\n' % (tx_id, tx_pos, _read_id,
                                                                   _read_prob))


                        read_probs, read_lengths, tx_ids, tx_positions, read_ids, all_kmers = \
                            [], [], [], [], [], []


def _calculate_site_proba(task):
    proba, n_iters, n_samples = task
    proba = np.random.choice(proba, n_iters * n_samples, replace=True).reshape(n_iters, n_samples)
    proba = (1 - np.prod(1 - proba, axis=1)).mean()
    return proba


def calculate_site_proba(read_probs, n_iters, n_samples, n_processes):
    tasks = ((read_prob, n_iters, n_samples) for read_prob in read_probs)
    with Pool(n_processes) as p:
        return [x for x in p.imap(_calculate_site_proba, tasks)]


def group_results(read_probs, tx_ids, tx_positions, read_ids, kmers, n_reads):
    i = 0

    grouped_probs = []
    grouped_tx_ids = []
    groped_tx_positions = []
    grouped_ids = []
    grouped_kmers = []
    grouped_n_reads = []

    for n_read in n_reads:
        grouped_probs.append(read_probs[i: i + n_read])
        grouped_ids.append(read_ids[i: i + n_read])
        grouped_tx_ids.append(tx_ids[i])
        groped_tx_positions.append(tx_positions[i])
        grouped_kmers.append(kmers[i])
        grouped_n_reads.append(n_read)
        i += n_read
    return grouped_probs, grouped_tx_ids, groped_tx_positions, grouped_ids, grouped_kmers, grouped_n_reads
