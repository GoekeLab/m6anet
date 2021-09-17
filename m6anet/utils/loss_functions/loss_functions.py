from multiprocessing import pool
import torch
from torch import matmul, diagonal
from torch.nn import CrossEntropyLoss
from torch.nn import BCELoss
from sklearn.cluster import KMeans


def binary_cross_entropy_loss(y_pred, y_true):
    loss = BCELoss()(y_pred.flatten(), y_true.float())
    return loss


def weighted_binary_cross_entropy_loss(y_pred, y_true):
    _, counts = torch.unique(y_true, return_counts=True)
    pos_weight, neg_weight = counts
    sample_weights = torch.where(y_true == 0, neg_weight, pos_weight)
    loss = BCELoss(reduce=False)(y_pred, y_true.float()) * sample_weights
    loss = loss.mean()
    return loss
