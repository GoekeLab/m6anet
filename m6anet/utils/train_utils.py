import numpy as np
import math
import os
import torch
from functools import partial
from multiprocessing import Pool
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, auc
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_roc_auc(y_true, y_pred):
    fpr, tpr, _  = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def get_pr_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1)
    pr_auc = auc(recall, precision)
    return pr_auc


def get_accuracy_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred >= 0.5)


def get_pr_auc_str(pr_auc):
    if np.isnan(pr_auc):
        return 'ROC AUC: N.A   '
    else:
        return 'PR AUC: %.4f' % pr_auc


def get_roc_auc_str(roc_auc):
    if np.isnan(roc_auc):
        return 'ROC AUC: N.A    , '
    else:
        return 'ROC AUC: %.4f, ' % roc_auc
    

def print_progress(loss, y_true, y_pred, 
                   epoch_num, idx_num, steps_per_epoch,
                   mode):
    accuracy = get_accuracy_score(y_true, y_pred)
    roc_auc = get_roc_auc(y_true, y_pred)
    pr_auc = get_pr_auc(y_true, y_pred)  
    
    progress_bar = '\r[{}: {}/{}]: '.format(epoch_num, idx_num + 1, steps_per_epoch)
    loss_str = '%s loss: %.4f, ' % (mode, loss)
    accuracy_str = '%s accuracy: %.4f, ' % (mode, accuracy)
    skip_line = (idx_num + 1) in (np.array([0.25, 0.50, 0.75, 1.0]) * steps_per_epoch)
    end_token = '\n' if skip_line else '\n'
    print(progress_bar, loss_str, accuracy_str,
          get_roc_auc_str(roc_auc),
          get_pr_auc_str(pr_auc), 
          end=end_token)
    if idx_num + 1 == steps_per_epoch:
        print("=================================")


def print_callback(y_true_running, y_pred_running, y_true, y_pred, 
                   i, n_print_iters, mode, 
                   n_samples_per_epoch, loss, loss_divisor,
                   epoch_num=None):
    y_true_running = np.append(y_true_running, y_true.detach().cpu().numpy())
    y_pred_running = np.append(y_pred_running, y_pred.detach().cpu().numpy())
    if (i + 1) % n_print_iters == 0:
        print_loss = loss / loss_divisor
        print_progress(print_loss, y_true_running, y_pred_running, epoch_num, i, n_samples_per_epoch, mode)
        y_true_running, y_pred_running = [], []
    return y_true_running, y_pred_running

def loss_callback(running_loss, losses, i, n_print_iters):
    if (i + 1) % n_print_iters == 0:
        losses.append(running_loss / n_print_iters)
        running_loss = 0.0
    return running_loss


def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)


def fit_one_epoch(model, dl, opt, device, criterion, clip_gradient, 
                  epoch_num, n_print_iters, scheduler, print_func):

    train_loss = [] 
    running_loss = 0.0
    y_true_running = []
    y_pred_running = []
    for i, inp in enumerate(dl):

        opt.zero_grad()

        y_true = inp[-1].to(device).view(-1, 1)
        y_pred = model(inp[0].to(device), inp[1].to(device), inp[2])
        loss = criterion()(y_pred, y_true)

        # Gradient update
        
        loss.backward()
        running_loss += loss.item()

        if clip_gradient is not None:
            # gradient clipping
            clip_grad_norm_(model.parameters(), clip_gradient)
            
        opt.step()

        if scheduler is not None:
            # learning rate scheduling
            scheduler.step()

        # Appending and Printing results
        y_true_running, y_pred_running = print_func(y_true_running, y_pred_running, 
                                                    y_true, y_pred, i, n_print_iters, 
                                                    'Train', len(dl), running_loss, 
                                                    n_print_iters, epoch_num)
        running_loss = loss_callback(running_loss, train_loss, i, n_print_iters)

    return model, train_loss


def predict(model, dl, device, criterion, 
            n_print_iters, mode, epoch_num=None, save_dir=None,
            val_prefix="", pred_prefix="",
            print_callback_func=None):
    if mode not in ('Test', 'Val'):
        raise ValueError("Mode must be one of (Test, Val)")
    else:
        model.eval()
        loss = 0.0
        n_samples = 0
        
        if mode == 'Val':
            y_pred_all = []
            y_true_all = []
            kmer_all = []

        y_true_running = []
        y_pred_running = []
        losses = []
        running_loss = 0.0
        print_func = print_callback if print_callback_func is None else print_callback_func
        for i, inp in enumerate(dl):
            # Inference
            y_true = inp[-1].to(device).view(-1, 1)
            y_pred = model(inp[0].to(device), inp[1].to(device), inp[2])

            if mode == 'Val':
                y_pred_all = np.append(y_pred_all, y_pred.detach().cpu().numpy())
                y_true_all = np.append(y_true_all, y_true.detach().cpu().numpy())
                kmer_all = np.append(kmer_all, inp[1].detach().cpu().numpy()[0])

            loss += criterion(reduction='sum')(y_pred, y_true).item()
            running_loss += loss / len(y_true)
            n_samples += len(y_true)

            y_true_running, y_pred_running = print_func(y_true_running, y_pred_running, y_true, y_pred, 
                                                        i, n_print_iters, mode, len(dl), loss, n_samples, 
                                                        epoch_num if epoch_num is not None else mode
                                                        )
            running_loss = loss_callback(running_loss, losses, i, n_print_iters)

        loss = loss / n_samples

        if (save_dir is not None) and (mode == 'Val'):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            
            np.save(os.path.join(save_dir, "y_pred{}.npy".format(pred_prefix)), y_pred_all)
            np.save(os.path.join(save_dir, "y_val{}.npy".format(val_prefix)), y_true_all)
            np.save(os.path.join(save_dir, "kmer_val{}.npy".format(pred_prefix)), kmer_all)

    return loss, losses

    
def fit(model, train_dl, device, opt, criterion, n_epochs, 
        test_dl=None, clip_gradient=False, scheduler=None,
        n_print_iters=20, save_dir=None, train_prefix="",
        test_prefix="",
        print_callback_func=None):
    running_train_losses = []
    test_loss_per_epoch = []
    running_test_losses = []
    model_states = []
    print_func = print_callback if print_callback_func is None else print_callback_func
    for epoch_num in range(1, n_epochs + 1):
        model.train()
        model, train_loss = fit_one_epoch(model, train_dl, opt, device, criterion, clip_gradient, 
                                          epoch_num, n_print_iters, scheduler, print_func)
        running_train_losses.extend(train_loss)

        if test_dl is not None:
            test_loss, running_test_loss = predict(model, test_dl, device, criterion, n_print_iters, 'Test', epoch_num,
                                                   print_callback_func=print_func)
            test_loss_per_epoch.append(test_loss)
            running_test_losses.extend(running_test_loss)
            model_states.append(model.state_dict())
    
    if save_dir is not None:
        np.save(os.path.join(save_dir, "train_loss{}.npy".format(train_prefix)), running_train_losses)
        np.save(os.path.join(save_dir, "test_loss{}.npy".format(test_prefix)), running_test_losses)

    return running_train_losses, running_test_losses, test_loss_per_epoch, model_states


def cross_validate(model, optimizer, data_dir, ds,  
                   device, criterion, save_dir,
                   num_epochs, mini_batch_size,
                   save_model=False,
                   print_callback_func=None,
                   opt_kwargs={},
                   train_ds_kwargs={}, test_ds_kwargs={}, val_ds_kwargs={},
                   train_dl_kwargs={}, test_dl_kwargs={}, val_dl_kwargs={},
                   training_kwargs={}, validation_kwargs={}):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    model = model.to(device)

    opt_func = partial(optimizer, **opt_kwargs)
    for fold_num in np.sort(os.listdir(data_dir)):
        print("Running cross validation for fold number: {}".format(fold_num))
        # Preparing save directory
        fold_dir = os.path.join(data_dir, fold_num)
        fold_save_dir = os.path.join(save_dir, fold_num)

        if not os.path.exists(fold_save_dir):
            os.mkdir(fold_save_dir)
    
        # Initializing model and optimizer
        model.apply(weight_reset)
        opt = opt_func(model.parameters())
    
        # Initializing datasets and dataloaders
        train_ds = ds(fold_dir, mode="train", **train_ds_kwargs)
        test_ds = ds(fold_dir, mode="test", **test_ds_kwargs)
        val_ds = ds(fold_dir, mode="val", **val_ds_kwargs)

        train_dl = DataLoader(train_ds, **train_dl_kwargs)
        test_dl = DataLoader(test_ds, **test_dl_kwargs)
        val_dl = DataLoader(val_ds, **val_dl_kwargs)

        # Fitting our model and choosing the best model out of all the epochs
        _, _, test_loss_per_epoch, model_states = fit(model, train_dl, device, opt, criterion, num_epochs, 
                                                      test_dl=test_dl, save_dir=fold_save_dir, 
                                                      print_callback_func=print_callback_func,
                                                      **training_kwargs)
        model.load_state_dict(model_states[np.argmin(test_loss_per_epoch)])

        # Running prediction on validation dataset
        predict(model, val_dl, device, criterion, 
                mode='Val', save_dir=fold_save_dir, **validation_kwargs,
                print_callback_func=print_callback_func)
        print("=======================================================")


def retrieve_and_save_norm_constants(input_dir, fpaths, out_dir, n_processes=1):
    all_kmers = np.array([x.split("_")[2] for x in fpaths])
    sort_idx = np.argsort(all_kmers)
    sorted_fpaths, sorted_kmers = fpaths[sort_idx], all_kmers[sort_idx]
    kmers, indices = np.unique(sorted_kmers, return_index=True)
    norm_const = []
    for i in range(len(kmers)):
        kmer = kmers[i]
        start_idx = indices[i]
        kmer_fpaths = sorted_fpaths[start_idx: indices[i + 1] if i < len(kmers) -1 else None]
        tasks = [(os.path.join(input_dir, fpath)) for fpath in kmer_fpaths]
        with Pool(n_processes) as p:
            sum_arrs = [x for x in tqdm(p.imap_unordered(compute_sum_and_sum_square, tasks), 
                                        total=len(kmer_fpaths), 
                                        desc="Computing mean and std for kmer {}".format(kmer))]
            N = np.sum([x[0] for x in sum_arrs])

            means = np.sum([x[1] for x in sum_arrs], axis=0) / N
            sum_of_squares = np.sum([x[2] for x in sum_arrs], axis=0)
            stds = np.sqrt((sum_of_squares / N) - means ** 2)

            norm_const.append([kmer] + means.tolist() + stds.tolist())
    pd.DataFrame(norm_const).to_csv(os.path.join(out_dir, "norm_constant.csv"), index=False)

    
def prepare_data_for_training(inference_dir, save_dir, labels,
                              train_idx, test_idx, val_idx=None,
                              n_processes=1):

    for mode in ('train', 'test'):
        mode_dir = os.path.join(save_dir, mode)
        os.makedirs(mode_dir)
    
    train_labels_df, test_labels_df = labels.iloc[train_idx], labels.iloc[test_idx]
    retrieve_and_save_norm_constants(inference_dir, train_labels_df["fnames"].values,
                                     save_dir, n_processes=n_processes)
    train_labels_df.to_csv(os.path.join(save_dir, "train", "data.csv.gz"))
    test_labels_df.to_csv(os.path.join(save_dir, "test", "data.csv.gz"))

    if val_idx is not None:
        os.makedirs(os.path.join(save_dir, "val"))
        val_labels_df = labels.iloc[val_idx]
        val_labels_df.to_csv(os.path.join(save_dir, "val", "data.csv.gz"))
