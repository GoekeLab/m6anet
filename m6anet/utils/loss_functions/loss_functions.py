r"""
This module is a collection of wrapper functions for loss functions used in m6Anet training
"""
import torch
from torch.nn import BCELoss


def binary_cross_entropy_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    r'''
    Wrapper for binary cross entropy loss in PyTorch

            Args:
                    y_pred (torch.Tensor): A PyTorch Tensor object containing model prediction
                    y_true (torch.Tensor): A PyTorch Tensor object containing prediction groundtruth

            Returns:
                    loss (torch.Tensor): loss value between y_pred and y_true
    '''
    loss = BCELoss()(y_pred.flatten(), y_true.float())
    return loss


def weighted_binary_cross_entropy_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    r'''
    Wrapper for weighted binary cross entropy loss in PyTorch

            Args:
                    y_pred (torch.Tensor): A PyTorch Tensor object containing model prediction
                    y_true (torch.Tensor): A PyTorch Tensor object containing prediction groundtruth

            Returns:
                    loss (torch.Tensor): weighted loss value between y_pred and y_true
    '''
    _, counts = torch.unique(y_true, return_counts=True)
    pos_weight, neg_weight = counts
    sample_weights = torch.where(y_true == 0, neg_weight, pos_weight)
    loss = BCELoss(reduce=False)(y_pred, y_true.float()) * sample_weights
    loss = loss.mean()
    return loss
