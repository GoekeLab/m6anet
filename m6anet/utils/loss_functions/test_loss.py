import torch
import numpy as np
from torch import matmul, diagonal
from torch.nn import CrossEntropyLoss
from torch.nn import BCELoss
from sklearn.cluster import KMeans

def cross_entropy_loss(y_pred, y_true):
    return CrossEntropyLoss()(y_pred, y_true)


def binary_cross_entropy_loss(y_pred, y_true):
    return BCELoss()(y_pred.flatten(), y_true.float())


def weighted_binary_cross_entropy_loss(y_pred, y_true):
    classes, counts = np.unique(y_true, return_counts=True)
    minority_idx, majority_idx = np.argmin(counts), np.argmax(counts)
    minority_class, majority_class = classes[minority_idx], classes[majority_idx]
    minority_weight = counts[majority_idx] / counts[minority_idx]
    weights = torch.where(y_true == minority_class, minority_weight, 1.0)
    return BCELoss(weights, reduction='sum')(y_pred, y_true) / len(y_true)
