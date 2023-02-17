r"""
This module is a collection functions used during training of m6Anet
"""
import os
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, auc
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Callable
from m6anet.model.model import MILModel


def get_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r'''
    Function to compute Receiving Operating Characteristic (ROC) AUC of m6Anet prediction against ground truth

            Args:
                    y_true (np.ndarray): A NumPy array containing th ground truth
                    y_true (np.ndarray): A NumPy array containing th ground truth

            Returns:
                    roc_auc (float): ROC AUC of the prediction
    '''
    fpr, tpr, _  = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def get_pr_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r'''
    Function to compute Precision-Recall (PR) AUC of m6Anet prediction against ground truth

            Args:
                    y_true (np.ndarray): A NumPy array containing th ground truth
                    y_true (np.ndarray): A NumPy array containing th ground truth

            Returns:
                    pr_auc (float): PR AUC of the prediction
    '''
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
    pr_auc = auc(recall, precision)
    return pr_auc


def get_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r'''
    Function to compute accuracy of m6Anet prediction against ground truth

            Args:
                    y_true (np.ndarray): A NumPy array containing th ground truth
                    y_true (np.ndarray): A NumPy array containing th ground truth

            Returns:
                    accuracy_score (float): Accuracy of the prediction
    '''
    return accuracy_score(y_true, y_pred)


def train(model: MILModel, train_dl: DataLoader, val_dl: DataLoader,
          optimizer: torch.optim.Optimizer, n_epoch:int, device:str,
          criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
          save_dir: Optional[str] = None,
          clip_grad: Optional[float] = None,
          save_per_epoch: Optional[int] = 10,
          epoch_increment: Optional[int] = 0,
          n_iterations: Optional[int] = 1) -> Tuple[Dict, Dict]:
    r'''
    The main function to train m6Anet model

            Args:
                    model (MILModel): An instance of MILModel class to perform Multiple Instance Learning based inference
                    train_dl (DataLoader): A PyTorch DataLoader object to load the preprocessed data.json file and train the model on
                    val_dl (DataLoader): A PyTorch DataLoader object to load the preprocessed data.json file and validate the model on
                    optimizer (torch.optim.Optimizer): A PyTorch compatible optimizer
                    n_epoch (int): Number of epochs to train m6Anet
                    device (str): Device id to perform training with
                    criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): A loss function that takes in  PyTorch Tensor prediction and
                                                                                      PyTorch Tensor groundtruth and output another PyTorch Tensor loss value
                    save_dir (str): Directory to save the training results
                    clip_grad (float): Maximum gradient value when using gradient clipping
                    save_per_epoch (int): Number of epoch multiple to save training checkpoint
                    epoch_increment (int): Increment to the number of current epoch, only used when resuming training
                    n_iterations (int): Number of sampling passes on each site of the validation dataset

            Returns:
                    (train_results, val_results) (tuple[dict, dict]): A tuple containing training results and validation results
    '''
    assert(save_per_epoch <= n_epoch)

    total_train_time = 0
    train_results = {}
    val_results = {}

    for epoch in range(1, n_epoch + 1):
        train_results_epoch = train_one_epoch(model, train_dl, device, optimizer, criterion,
                                              clip_grad=clip_grad)
        val_results_epoch = validate(model, val_dl, device, criterion, n_iterations)

        total_train_time += train_results_epoch["compute_time"] + val_results_epoch["compute_time"]
        print("Epoch:[{epoch}/{n_epoch}] \t "
              "train time:{train_time:.0f}s \t "
              "val time:{val_time:.0f}s \t "
              "({total:.0f}s)".format(epoch=epoch + epoch_increment,
                                      n_epoch = n_epoch + epoch_increment,
                                      train_time=train_results_epoch["compute_time"],
                                      val_time=val_results_epoch["compute_time"],
                                      total=total_train_time))
        print("Train Loss:{loss:.2f}\t "
              "Train ROC AUC: {roc_auc:.3f}\t "
              "Train PR AUC: {pr_auc:.3f}".format(loss=train_results_epoch["avg_loss"],
                                                  roc_auc=train_results_epoch["roc_auc"],
                                                  pr_auc=train_results_epoch["pr_auc"]))

        print("Val Loss:{loss:.2f} \t "
              "Val ROC AUC: {roc_auc:.3f}\t "
              "Val PR AUC: {pr_auc:.3f}".format(loss=val_results_epoch["avg_loss"],
                                                roc_auc=val_results_epoch["roc_auc"],
                                                pr_auc=val_results_epoch["pr_auc"]))

        print("=====================================")

        #save statistics for later plotting

        for key in train_results_epoch.keys():
            result = train_results_epoch[key]
            if key not in train_results:
                train_results[key] = [result]
            else:
                train_results[key].append(result)


        for key in val_results_epoch.keys():
            result = val_results_epoch[key]
            if key not in val_results:
                val_results[key] = [result]
            else:
                val_results[key].append(result)

        if (save_dir is not None) and ((epoch + epoch_increment) % save_per_epoch == 0):
            save_path = os.path.join(save_dir, "model_states", str(epoch + epoch_increment))
            os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, "model_states.pt"))
    return train_results, val_results


def train_one_epoch(model: MILModel, pair_dataloader: DataLoader, device:str,
                    optimizer: torch.optim.Optimizer,
                    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                    clip_grad: Optional[float] = None) -> Dict:
    r'''
    Function to train m6Anet model for one epoch

            Args:
                    model (MILModel): An instance of MILModel class to perform Multiple Instance Learning based inference
                    pair_dataloader (DataLoader): A PyTorch DataLoader object to load the preprocessed data.json file and train the model on
                    device (str): Device id to perform training with
                    optimizer (torch.optim.Optimizer): A PyTorch compatible optimizer
                    criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): A loss function that takes in  PyTorch Tensor prediction and
                                                                                      PyTorch Tensor groundtruth and output another PyTorch Tensor loss value
                    clip_grad (float): Maximum gradient value when using gradient clipping

            Returns:
                    loss_results (dict): A dictionary containing training metrics such as loss objective, accuracy, roc auc, and pr auc of the this training iteration
    '''
    model.train()
    train_loss_list = []

    start = time.time()
    all_y_true = []
    all_y_pred = []

    loss_results = {}
    for batch in pair_dataloader:
        y_true = batch.pop('y').to(device).flatten()
        y_pred = model({key: val.to(device) for key, val in batch.items()})
        loss = criterion(y_pred, y_true)
        loss.backward()

        if clip_grad is not None:
            clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        optimizer.zero_grad()

        train_loss_list.append(loss.item())

        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()


        all_y_true.extend(y_true)

        if (len(y_pred.shape) == 1) or (y_pred.shape[1] == 1):
            all_y_pred.extend(y_pred.flatten())

        else:
            all_y_pred.extend(y_pred[:, 1])

    compute_time = time.time() - start
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    loss_results['compute_time'] = compute_time
    loss_results['avg_loss'] = np.mean(np.array(train_loss_list))
    loss_results['roc_auc'] = get_roc_auc(all_y_true, all_y_pred)
    loss_results['pr_auc'] = get_pr_auc(all_y_true, all_y_pred)

    return loss_results


def validate(model: MILModel, val_dl: DataLoader,
             device: str, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             n_iterations: Optional[int] = 1) -> Dict:
    r'''
    Function to validate m6Anet model on the validation dataset

            Args:
                    model (MILModel): An instance of MILModel class to perform Multiple Instance Learning based inference
                    val_dl (DataLoader): A PyTorch DataLoader object to load the preprocessed data.json file and validate the model on
                    device (str): Device id to perform training with
                    criterion (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): A loss function that takes in  PyTorch Tensor prediction and
                                                                                      PyTorch Tensor groundtruth and output another PyTorch Tensor loss value
                    n_iterations (int): Number of sampling passes on each site of the validation dataset

            Returns:
                    val_results (dict): A dictionary containing validation metrics such as loss objective, accuracy, roc auc, and pr auc
    '''
    model.eval()
    all_y_true = None
    all_y_pred = []
    start = time.time()

    with torch.no_grad():
        for _ in range(n_iterations):
            y_true_tmp = []
            y_pred_tmp = []
            for batch in val_dl:
                y_true = batch.pop('y').to(device).flatten()
                y_pred = model({key: val.to(device) for key, val in batch.items()}).detach().cpu().numpy()
                if all_y_true is None:
                    y_true_tmp.extend(y_true.detach().cpu().numpy())

                if (len(y_pred.shape) == 1) or (y_pred.shape[1] == 1):
                    y_pred_tmp.extend(y_pred.flatten())
                else:
                    y_pred_tmp.extend(y_pred[:, 1])

            if all_y_true is None:
                all_y_true = y_true_tmp

            all_y_pred.append(y_pred_tmp)

    compute_time = time.time() - start
    y_pred_avg = np.mean(all_y_pred, axis=0)
    all_y_true = np.array(all_y_true).flatten()

    val_results = {}

    val_results['y_pred'] = all_y_pred
    val_results['y_true'] = all_y_true
    val_results['compute_time'] = compute_time
    val_results['roc_auc'] = get_roc_auc(all_y_true, y_pred_avg)
    val_results['pr_auc'] = get_pr_auc(all_y_true, y_pred_avg)
    val_results["avg_loss"] = criterion(torch.Tensor(y_pred_avg), torch.Tensor(all_y_true)).item()

    return val_results
