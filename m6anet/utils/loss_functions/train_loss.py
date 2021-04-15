from multiprocessing import pool
import torch
from torch import matmul, diagonal
from torch.nn import CrossEntropyLoss
from torch.nn import BCELoss
from sklearn.cluster import KMeans


def create_train_results(y_pred, loss):
    return {'y_pred': y_pred, 'loss': loss}


def cross_entropy_loss(model, X, y_true):
    y_pred = model(X)
    loss = CrossEntropyLoss()(y_pred, y_true)
    return create_train_results(y_pred, loss)


def weighted_cross_entropy_loss(model, X, y_true):
    y_pred = model(X)
    _, counts = torch.unique(y_true, return_counts=True)
    weights = 1 / counts.float()
    loss = CrossEntropyLoss(weight=weights)(y_pred, y_true)
    return create_train_results(y_pred, loss)


def binary_cross_entropy_loss(model, X, y_true):
    y_pred = model(X)
    loss = BCELoss()(y_pred.flatten(), y_true.float())
    return create_train_results(y_pred, loss)


def weighted_binary_cross_entropy_loss(model, X, y_true):
    y_pred = model(X)
    _, counts = torch.unique(y_true, return_counts=True)
    pos_weight, neg_weight = counts
    sample_weights = torch.where(y_true == 0, neg_weight, pos_weight)
    loss = BCELoss(reduce=False)(y_pred, y_true.float()) * sample_weights
    loss = loss.mean()
    return create_train_results(y_pred, loss)


def read_site_level_entropy(model, X, y_true, 
                            lambda_1=1, lambda_2=1,
                            mode='prod_pooling'):
    read_level_probability, site_level_probability, _ = model.get_read_site_probability(X)

    if mode == 'prod_pooling':
        pooled_read_probability = 1 - torch.prod(1 - read_level_probability, axis=1)
    elif mode == 'mean_pooling':
        pooled_read_probability = torch.mean(read_level_probability, axis=1)
    elif mode == 'max_pooling':
        pooled_read_probability = torch.max(read_level_probability, axis=1).values

    site_level_loss = CrossEntropyLoss()(site_level_probability, y_true)
    read_level_loss = BCELoss()(pooled_read_probability, y_true.float())
    loss = lambda_1 * site_level_loss + lambda_2 * read_level_loss
    return create_train_results(site_level_probability, loss)


def read_site_level_entropy_cluster(model, X, y_true, 
                                    lambda_1=1, lambda_2=1, lambda_3=1,
                                    mode='prod_pooling'):
    read_level_probability, site_level_probability, read_representation = model.get_read_site_probability(X)
    
    # Calculating site probability through pooling of read level probability and classifier

    if mode == 'prod_pooling':
        pooled_read_probability = 1 - torch.prod(1 - read_level_probability, axis=1)
    elif mode == 'mean_pooling':
        pooled_read_probability = torch.mean(read_level_probability, axis=1)
    elif mode == 'max_pooling':
        pooled_read_probability = torch.max(read_level_probability, axis=1).values

    site_level_loss = CrossEntropyLoss()(site_level_probability, y_true)
    read_level_loss = BCELoss()(pooled_read_probability, y_true.float())

    # Calculate loss based on the compactness of reads with similar probability score
    
    with torch.no_grad():
        prob = read_level_probability.detach().cpu().numpy().reshape(-1, 1)
        kmeans = KMeans(2).fit(prob)
        labels = kmeans.labels_
    
    read_representation = read_representation.reshape(-1, read_representation.shape[-1])
    probability_mask = (labels == 1)   
    pos_reps = read_representation[probability_mask]
    neg_reps = read_representation[~probability_mask]
    device = read_representation.device
    eps = torch.Tensor([1e-08]).to(device)
    rep_loss = 0
    for reps in [pos_reps, neg_reps]:
        if len(reps) > 1:
            reps = reps.div(torch.max(reps.pow(2).sum(1, keepdim=True).pow(0.5), eps))
            rep_similarities = reps @ reps.t()
            norm_constants = rep_similarities.numel() - rep_similarities.shape[0]
            avg_similarities = (torch.sum(rep_similarities) - torch.sum(torch.diagonal(rep_similarities))) / norm_constants # zeroing the diagonal
            sim_loss = -torch.log(avg_similarities)
            rep_loss += sim_loss

    loss =  lambda_1 * site_level_loss + lambda_2 * read_level_loss + lambda_3 * rep_loss
    return {'y_pred': site_level_probability, 'loss': loss, 'y_pred_read': pooled_read_probability,
            'site_loss': site_level_loss, 'read_loss': read_level_loss, 'sim_loss': sim_loss}


def read_site_level_correlation(model, X, y_true, 
                                lambda_1=1, lambda_2=1, lambda_3=1,
                                mode='prod_pooling'):
    read_level_probability, site_level_probability, read_representation = model.get_read_site_probability(X)
    
    # Calculating site probability through pooling of read level probability and classifier

    if mode == 'prod_pooling':
        pooled_read_probability = 1 - torch.prod(1 - read_level_probability, axis=1)
    elif mode == 'mean_pooling':
        pooled_read_probability = torch.mean(read_level_probability, axis=1)
    elif mode == 'max_pooling':
        pooled_read_probability = torch.max(read_level_probability, axis=1).values

    site_level_loss = BCELoss()(site_level_probability.flatten(), y_true.float())
    read_level_loss = BCELoss()(pooled_read_probability, y_true.float())

    # Calculate correlation between site level probabiltiy and read level
    
    device = pooled_read_probability.device

    vx = site_level_probability - torch.mean(site_level_probability.flatten())
    vy = pooled_read_probability - torch.mean(pooled_read_probability)
    eps = torch.Tensor([1e-6]).to(device)
    pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + eps) * torch.sqrt(torch.sum(vy ** 2) + eps))

    pcc_loss = 1 - pcc

    loss =  lambda_1 * site_level_loss + lambda_2 * read_level_loss + lambda_3 * pcc_loss
    return {'y_pred': site_level_probability, 'loss': loss, 'y_pred_read': pooled_read_probability,
            'site_loss': site_level_loss, 'read_loss': read_level_loss, 'pcc_loss': pcc_loss}


def site_level_entropy_cluster(model, X, y_true, 
                               lambda_1=1, lambda_2=1,
                                mode='prod_pooling'):
    read_level_probability, site_level_probability, read_representation = model.get_read_site_probability(X)
    
    # Calculating site probability through pooling of read level probability and classifier

    site_level_loss = BCELoss()(site_level_probability, y_true.float())

    # Calculate loss based on the compactness of reads with similar probability score
    
    with torch.no_grad():
        prob = read_level_probability.detach().cpu().numpy().reshape(-1, 1)
        kmeans = KMeans(2).fit(prob)
        labels = kmeans.labels_
    
    read_representation = read_representation.reshape(-1, read_representation.shape[-1])
    probability_mask = (labels == 1)   
    pos_reps = read_representation[probability_mask]
    neg_reps = read_representation[~probability_mask]
    device = read_representation.device
    eps = torch.Tensor([1e-08]).to(device)
    rep_loss = 0
    for reps in [pos_reps, neg_reps]:
        if len(reps) > 1:
            reps = reps.div(torch.max(reps.pow(2).sum(1, keepdim=True).pow(0.5), eps))
            rep_similarities = reps @ reps.t()
            norm_constants = rep_similarities.numel() - rep_similarities.shape[0]
            avg_similarities = (torch.sum(rep_similarities) - torch.sum(torch.diagonal(rep_similarities))) / norm_constants # zeroing the diagonal
            sim_loss = -torch.log(avg_similarities)
            rep_loss += sim_loss

    loss =  lambda_1 * site_level_loss + lambda_2 * rep_loss
    return {'y_pred': site_level_probability, 'loss': loss, 
            'site_loss': site_level_loss, 'sim_loss': sim_loss}
