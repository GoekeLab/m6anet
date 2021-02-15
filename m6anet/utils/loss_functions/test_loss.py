import torch
from torch import matmul, diagonal
from torch.nn import CrossEntropyLoss
from torch.nn import BCELoss
from sklearn.cluster import KMeans

def cross_entropy_loss(y_pred, y_true):
    return CrossEntropyLoss()(y_pred, y_true)


def binary_cross_entropy_loss(y_pred, y_true):
    return BCELoss()(y_pred, y_true.float())
