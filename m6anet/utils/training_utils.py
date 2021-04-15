import numpy as np
import os
import torch
import time
from collections import Iterable
from torch import nn
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, auc
from torch.nn.utils import clip_grad_norm_


def get_roc_auc(y_true, y_pred):
    fpr, tpr, _  = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def get_pr_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
    pr_auc = auc(recall, precision)
    return pr_auc


def get_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def train(model, train_dl, test_dl, optimizer, n_epoch, device, train_criterion,
          test_criterion, scheduler=None, save_dir=None, clip_grad=None,
          save_per_epoch=10, epoch_increment=0, n_iterations=1, read_level_info=False):
    
    assert(save_per_epoch <= n_epoch)
    
    total_train_time = 0
    train_results = {}
    test_results = {}
    
    for epoch in range(1, n_epoch + 1):
        train_results_epoch = train_one_epoch(model, train_dl, device, optimizer, train_criterion, 
                                              scheduler=scheduler, clip_grad=clip_grad, read_level_info=read_level_info)
        test_results_epoch = test(model, test_dl, device, test_criterion, n_iterations, read_level_info=read_level_info)

        total_train_time += train_results_epoch["compute_time"] + test_results_epoch["compute_time"]
        print("Epoch:[{epoch}/{n_epoch}] \t "
              "train time:{train_time:.0f}s \t "
              "test time:{test_time:.0f}s \t "
              "({total:.0f}s)".format(epoch=epoch + epoch_increment, 
                                      n_epoch = n_epoch + epoch_increment,
                                      train_time=train_results_epoch["compute_time"],
                                      test_time=test_results_epoch["compute_time"],
                                      total=total_train_time))
        print("Train Loss:{loss:.2f}\t "
              "Train accuracy: {acc:.3f}\t "
              "ROC AUC: {roc_auc:.3f}\t "
              "PR AUC: {pr_auc:.3f}".format(loss=train_results_epoch["avg_loss"],
                                               acc=train_results_epoch["accuracy"],
                                               roc_auc=train_results_epoch["roc_auc"],
                                               pr_auc=train_results_epoch["pr_auc"]))
        if read_level_info:
            print("Train accuracy read: {acc:.3f}\t "
                  "ROC AUC read: {roc_auc:.3f}\t "
                  "PR AUC read: {pr_auc:.3f}".format(acc=train_results_epoch["accuracy_read"],
                                                     roc_auc=train_results_epoch["roc_auc_read"],
                                                     pr_auc=train_results_epoch["pr_auc_read"]))        
        print("Test Loss:{loss:.2f} \t "
              "Test accuracy: {acc:.3f}\t "
              "ROC AUC: {roc_auc:.3f}\t "
              "PR AUC: {pr_auc:.3f}".format(loss=test_results_epoch["avg_loss"],
                                               acc=test_results_epoch["accuracy"],
                                               roc_auc=test_results_epoch["roc_auc"],
                                               pr_auc=test_results_epoch["pr_auc"]))
        if read_level_info:
            print("Test loss read: {loss:.3f}\t "
                  "Test accuracy read: {acc:.3f}\t "
                  "ROC AUC read: {roc_auc:.3f}\t "
                  "PR AUC read: {pr_auc:.3f}".format(loss=test_results_epoch["avg_loss_read"],
                                                     acc=test_results_epoch["accuracy_read"],
                                                     roc_auc=test_results_epoch["roc_auc_read"],
                                                     pr_auc=test_results_epoch["pr_auc_read"]))
        print("=====================================")
        #save statistics for later plotting

        for key in train_results_epoch.keys():
            result = train_results_epoch[key]
            if key not in train_results:
                if not isinstance(result, Iterable):
                    train_results[key] = [result]
                else:
                    train_results[key] = result
            else:
                train_results[key].append(result)


        for key in test_results_epoch.keys():
            result = test_results_epoch[key]
            if key not in test_results:
                if not isinstance(result, Iterable):
                    test_results[key] = [result]
                else:
                    test_results[key] = result
            else:
                test_results[key].append(result)

        if (save_dir is not None) and ((epoch + epoch_increment) % save_per_epoch == 0):
            save_path = os.path.join(save_dir, "model_states", str(epoch + epoch_increment))
            os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, "model_states.pt"))
    return train_results, test_results


def train_one_epoch(model, pair_dataloader, device, optimizer, criterion, scheduler=None, clip_grad=None, read_level_info=False):
    model.train()
    train_loss_list = []

    start = time.time()
    all_y_true = []
    all_y_pred = []
    all_y_pred_read = []

    loss_results = {}
    for batch in pair_dataloader:
        y_true = batch.pop('y').to(device).flatten()
        X = {key: val.to(device) for key, val in batch.items()}
        train_results = criterion(model, X, y_true)
        y_pred, loss = train_results.pop("y_pred"), train_results.pop("loss")

        loss.backward()

        if clip_grad is not None:
            clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        optimizer.zero_grad()
                
        train_loss_list.append(loss.item())
        
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()


        all_y_true.extend(y_true)
        
        if read_level_info:
            y_pred_read = train_results.pop("y_pred_read")
            y_pred_read = y_pred_read.detach().cpu().numpy()
            all_y_pred_read.extend(y_pred_read)

        if len(loss_results) == 0:
            loss_results = {key: [item.detach().cpu().item()] for key, item in train_results.items()}
        else:
            for key in loss_results.keys():
                loss_results[key].append(train_results[key].detach().cpu().item())  

        if (len(y_pred.shape) == 1) or (y_pred.shape[1] == 1):
            all_y_pred.extend(y_pred.flatten())

        else:
            all_y_pred.extend(y_pred[:, 1])

        if scheduler is not None:
            #update learning rate when scheduler is given
            scheduler.step()
            
    compute_time = time.time() - start
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)


    if len(all_y_pred_read) > 0:
        all_y_pred_read = np.array(all_y_pred_read)
        loss_results['accuracy_read'] = get_accuracy(all_y_true.flatten(), (all_y_pred_read.flatten() > 0.5) * 1)
        loss_results['roc_auc_read'] = get_roc_auc(all_y_true, all_y_pred_read)
        loss_results['pr_auc_read'] = get_pr_auc(all_y_true, all_y_pred_read)
        
    loss_results['compute_time'] = compute_time
    loss_results['avg_loss'] = np.mean(np.array(train_loss_list))
    loss_results['accuracy'] = get_accuracy(all_y_true.flatten(), (all_y_pred.flatten() > 0.5) * 1)
    loss_results['roc_auc'] = get_roc_auc(all_y_true, all_y_pred)
    loss_results['pr_auc'] = get_pr_auc(all_y_true, all_y_pred)

    return loss_results



def test(model, pair_dataloader, device, criterion, n_iterations=1, read_level_info=False) :
    """
    Test time function
    """
    model.eval()
    all_y_true = None
    all_y_pred = []
    all_y_pred_read = []
    
    start = time.time()

    with torch.no_grad():
        for n in range(n_iterations):
            y_true_tmp = []
            y_pred_tmp = []
            y_pred_read_tmp = []
            for batch in pair_dataloader:
                y_true = batch.pop('y').to(device).flatten()
                X = {key: val.to(device) for key, val in batch.items()}
                y_pred = model(X).detach().cpu().numpy()
                
                if all_y_true is None:
                    y_true_tmp.extend(y_true.detach().cpu().numpy())

                if (len(y_pred.shape) == 1) or (y_pred.shape[1] == 1):
                    y_pred_tmp.extend(y_pred.flatten())
                else:
                    y_pred_tmp.extend(y_pred[:, 1])
                
                if read_level_info:
                    y_pred_read = model.get_read_probability(X)
                    y_pred_read = 1 - torch.prod(1 - y_pred_read, axis=1)
                    y_pred_read = y_pred_read.detach().cpu().numpy()
                    y_pred_read_tmp.extend(y_pred_read)

            if all_y_true is None:
                all_y_true = y_true_tmp

            all_y_pred.append(y_pred_tmp)

            if read_level_info:
                all_y_pred_read.append(y_pred_read_tmp)

    compute_time = time.time() - start
    y_pred_avg = np.mean(all_y_pred, axis=0)
    all_y_true = np.array(all_y_true).flatten()

    test_results = {}

    test_results['compute_time'] = compute_time
    test_results['accuracy'] = get_accuracy(all_y_true, (y_pred_avg.flatten() > 0.5) * 1)
    test_results['roc_auc'] = get_roc_auc(all_y_true, y_pred_avg)
    test_results['pr_auc'] = get_pr_auc(all_y_true, y_pred_avg)
    test_results["avg_loss"] = criterion(torch.Tensor(y_pred_avg), torch.Tensor(all_y_true)).item()

    if len(all_y_pred_read) > 0:
        all_y_pred_read = np.mean(all_y_pred_read, axis=0)
        all_y_pred_read = np.array(all_y_pred_read)

        test_results['accuracy_read'] = get_accuracy(all_y_true.flatten(), (all_y_pred_read.flatten() > 0.5) * 1)
        test_results['roc_auc_read'] = get_roc_auc(all_y_true, all_y_pred_read)
        test_results['pr_auc_read'] = get_pr_auc(all_y_true, all_y_pred_read)
        test_results['avg_loss_read'] = criterion(torch.Tensor(all_y_pred_read), torch.Tensor(all_y_true)).item()
    return test_results


def validate(model, pair_dataloader, device, criterion, n_iterations=1, read_level_info=False):
    """
    extract all the features from a given dataset
    """
    model.eval()
    all_y_true = None
    all_y_pred = []
    all_y_pred_read = []
    start = time.time()

    with torch.no_grad():
        for n in range(n_iterations):
            y_true_tmp = []
            y_pred_tmp = []
            y_pred_read_tmp = []
            for batch in pair_dataloader:
                y_true = batch.pop('y').to(device).flatten()
                X = {key: val.to(device) for key, val in batch.items()}
                y_pred = model(X).detach().cpu().numpy()
                
                if all_y_true is None:
                    y_true_tmp.extend(y_true.detach().cpu().numpy())

                if (len(y_pred.shape) == 1) or (y_pred.shape[1] == 1):
                    y_pred_tmp.extend(y_pred.flatten())
                else:
                    y_pred_tmp.extend(y_pred[:, 1])

                if read_level_info:
                    y_pred_read = model.get_read_probability(X)
                    y_pred_read = 1 - torch.prod(1 - y_pred_read, axis=1)
                    y_pred_read = y_pred_read.detach().cpu().numpy()
                    y_pred_read_tmp.extend(y_pred_read)

            if all_y_true is None:
                all_y_true = y_true_tmp

            all_y_pred.append(y_pred_tmp)
            
            if read_level_info:
                all_y_pred_read.append(y_pred_read_tmp)

    compute_time = time.time() - start
    y_pred_avg = np.mean(all_y_pred, axis=0)
    all_y_true = np.array(all_y_true).flatten()

    val_results = {}

    val_results['y_pred'] = all_y_pred
    val_results['y_true'] = all_y_true
    val_results['compute_time'] = compute_time
    val_results['accuracy'] = get_accuracy(all_y_true, (y_pred_avg.flatten() > 0.5) * 1)
    val_results['roc_auc'] = get_roc_auc(all_y_true, y_pred_avg)
    val_results['pr_auc'] = get_pr_auc(all_y_true, y_pred_avg)
    val_results["avg_loss"] = criterion(torch.Tensor(y_pred_avg), torch.Tensor(all_y_true)).item()

    if len(all_y_pred_read) > 0:
        all_y_pred_read = np.mean(all_y_pred_read, axis=0)
        all_y_pred_read = np.array(all_y_pred_read)

        val_results['accuracy_read'] = get_accuracy(all_y_true.flatten(), (all_y_pred_read.flatten() > 0.5) * 1)
        val_results['roc_auc_read'] = get_roc_auc(all_y_true, all_y_pred_read)
        val_results['pr_auc_read'] = get_pr_auc(all_y_true, all_y_pred_read)
        val_results["avg_loss_read"] = criterion(torch.Tensor(all_y_pred_read), torch.Tensor(all_y_true)).item()

    return val_results


def inference(model, dl, device, n_iterations=1):
    """
    Run inference on unlabelled dataset
    """
    model.eval()
    all_y_pred = []
    with torch.no_grad():
        for n in range(n_iterations):
            y_pred_tmp = []
            for batch in dl:
                X = {key: val.to(device) for key, val in batch.items()}
                y_pred = model(X)
                y_pred = y_pred.detach().cpu().numpy()
                if (len(y_pred.shape) == 1) or (y_pred.shape[1] == 1):
                    y_pred_tmp.extend(y_pred.flatten())
                else:
                    y_pred_tmp.extend(y_pred[:, 1])

            all_y_pred.append(y_pred_tmp)
    return np.mean(all_y_pred, axis=0)
