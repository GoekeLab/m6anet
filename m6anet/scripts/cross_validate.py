
  
import pandas as pd
import numpy as np
import os
import torch
import datetime
import joblib
import toml
import json
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from ..utils.training_utils import train, validate
from ..utils.builder import build_dataloader, build_loss_function
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from tqdm import tqdm
from ..model.model import MILModel
from copy import deepcopy


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("--cv_dir", default=None, required=True)
    parser.add_argument("--model_config", default=None, required=True)
    parser.add_argument("--train_config", default=None, required=True)
    parser.add_argument("--cv", dest='cv', default=5, type=int)
    parser.add_argument("--save_dir", default=None, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--lr_scheduler", dest='lr_scheduler', default=None, action='store_true')
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--num_workers", default=25, type=int)
    parser.add_argument("--save_per_epoch", default=10, type=int)
    parser.add_argument("--weight_decay", dest="weight_decay", default=0, type=float)
    parser.add_argument("--num_iterations", default=1, type=int)
    return parser


def cross_validate(args):
    
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    cv_dir = args.cv_dir

    device = args.device
    num_workers = args.num_workers
    n_epoch = args.epochs
    lr = args.lr
    lr_scheduler = args.lr_scheduler
    save_per_epoch = args.save_per_epoch
    save_dir = args.save_dir
    cv = args.cv
    weight_decay = args.weight_decay
    n_iterations = args.num_iterations

    model_config = toml.load(args.model_config)
    train_config = toml.load(args.train_config)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Saving training information to {}".format(save_dir))

    cv_info = dict()
    cv_info["model_config"] = model_config
    cv_info["train_config"] = train_config
    cv_info["train_config"]["learning_rate"] = lr
    cv_info["train_config"]["epochs"] = n_epoch
    cv_info["train_config"]["save_per_epoch"] = save_per_epoch
    cv_info["train_config"]["weight_decay"] = weight_decay
    cv_info["train_config"]["number_of_validation_iterations"] = n_iterations
    cv_info["train_config"]["lr_scheduler"] = lr_scheduler
    cv_info["train_config"]["cv_folds"] = cv
    cv_info["train_config"]["seed"] = seed

    with open(os.path.join(save_dir, "cv_info.toml"), 'w') as f:
        toml.dump(cv_info, f)

    final_test_df = {}
    selection_criterions = ['avg_loss', 'roc_auc', 'pr_auc']
    columns = ["transcript_id", "transcript_position", "n_reads", "chr", "gene_id", "genomic_position", "kmer", "modification_status", "probability_modified"]

    for fold_num in range(1, cv + 1):
        fold_dir_save = os.path.join(save_dir, str(fold_num))
        
        if not os.path.exists(fold_dir_save):
            os.mkdir(fold_dir_save)

        print("Begin running cross validation for fold number {} for a total of {} folds".format( fold_num, cv))

        model_config_copy, train_config_copy = deepcopy(model_config), deepcopy(train_config)
        fold_dir = os.path.join(cv_dir, str(fold_num))
        train_config_copy["dataset"]["site_info"] = fold_dir
        train_config_copy["dataset"]["norm_path"] = os.path.join(fold_dir, "norm_dict.joblib")

        model = MILModel(model_config_copy).to(device)
        train_dl, val_dl, test_dl = build_dataloader(train_config, num_workers)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        criterion = build_loss_function(train_config_copy['loss_function'])

        train_results, val_results = train(model, train_dl, val_dl, optimizer, n_epoch, device, 
                                           criterion, save_dir=fold_dir_save,
                                           save_per_epoch=save_per_epoch, n_iterations=n_iterations)

        joblib.dump(train_results, os.path.join(fold_dir_save, "train_results.joblib"))      
        joblib.dump(val_results, os.path.join(fold_dir_save, "val_results.joblib"))   

        for selection_criterion in selection_criterions:
            val_loss = [val_results[selection_criterion][i] for i in range (0, len(val_results[selection_criterion]), save_per_epoch)]

            if selection_criterion == 'avg_loss':
                best_model = (np.argmin(val_loss) + 1) * save_per_epoch
            else:
                best_model = (np.argmax(val_loss) + 1) * save_per_epoch

            model.load_state_dict(torch.load(os.path.join(fold_dir_save, "model_states", str(best_model), "model_states.pt")))
            test_results = validate(model, test_dl, device, criterion, n_iterations)
            print("Criteria: {criteria} \t"
                "Compute time: {compute_time:.3f}".format(criteria=selection_criterion, compute_time=test_results["compute_time"]))
            print("Test Loss: {loss:.3f} \t"
                "Test ROC AUC: {roc_auc:.3f} \t "
                "Test PR AUC: {pr_auc:.3f}".format(loss=test_results["avg_loss"],
                                                    roc_auc=test_results["roc_auc"],
                                                    pr_auc=test_results["pr_auc"]))
            print("=====================================")

            joblib.dump(test_results, os.path.join(save_dir, "test_results_{}.joblib".format(selection_criterion)))   
            test_df = deepcopy(test_dl.dataset.data_info)
            test_df.loc[:, "probability_modified"] = np.mean(test_results["y_pred"], axis=0)
            test_df = test_df[columns]
            
            if selection_criterion not in final_test_df:
                final_test_df[selection_criterion] = [test_df]
            else:
                final_test_df[selection_criterion].append(test_df)
        
    for selection_criterion in selection_criterions:
        pd.concat(final_test_df[selection_criterion]).reset_index(drop=True).to_csv(os.path.join(save_dir, "test_results_{}.csv.gz".format(selection_criterion)), 
                                                                                    index=False)
        

def main():
    args = argparser().parse_args()
    cross_validate(args)


if __name__ == '__main__':
    main()
