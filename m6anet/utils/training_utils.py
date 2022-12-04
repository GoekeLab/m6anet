import numpy as np
import os
import torch
import time
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


def train(model, train_dl, val_dl, optimizer, n_epoch, device, criterion,
          save_dir=None, clip_grad=None,
          save_per_epoch=10, epoch_increment=0, n_iterations=1):

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


def train_one_epoch(model, pair_dataloader, device, optimizer, criterion, clip_grad=None):
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


def validate(model, val_dl, device, criterion, n_iterations=1):
    """
    Function to run validation
    """
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
