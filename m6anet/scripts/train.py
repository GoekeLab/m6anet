import numpy as np
import os
import torch
import joblib
import toml
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from ..utils.training_utils import train, test, validate
from ..utils.builder import build_dataloader, build_train_loss_function, build_test_loss_function
from ..model.model import MILModel
  

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("--model_config", default=None, required=True)
    parser.add_argument("--train_config", default=None, required=True)
    parser.add_argument("--save_dir", default=None, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--num_workers", default=25, type=int)
    parser.add_argument("--save_per_epoch", default=10, type=int)
    parser.add_argument("--lr_scheduler", dest='lr_scheduler', default=None, action='store_true')
    parser.add_argument("--read_level_info", dest='read_level_info', default=None, action='store_true')
    parser.add_argument("--weight_decay", dest="weight_decay", default=0, type=float)
    parser.add_argument("--num_iterations", default=1, type=int)
    return parser


def train_and_save(args):
    
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = args.device
    num_workers = args.num_workers
    n_epoch = args.epochs
    lr = args.lr
    lr_scheduler = args.lr_scheduler
    save_per_epoch = args.save_per_epoch
    save_dir = args.save_dir
    weight_decay = args.weight_decay
    n_iterations = args.num_iterations
    read_level_info = args.read_level_info

    model_config = toml.load(args.model_config)
    train_config = toml.load(args.train_config)

    print("Saving training information to {}".format(save_dir))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_info = dict()
    train_info["model_config"] = model_config
    train_info["train_config"] = train_config
    train_info["train_config"]["learning_rate"] = lr
    train_info["train_config"]["epochs"] = n_epoch
    train_info["train_config"]["save_per_epoch"] = save_per_epoch
    train_info["train_config"]["weight_decay"] = weight_decay
    train_info["train_config"]["number_of_validation_iterations"] = n_iterations
    train_info["train_config"]["lr_scheduler"] = lr_scheduler
    train_info["train_config"]["seed"] = seed

    with open(os.path.join(save_dir, "train_info.toml"), 'w') as f:
        toml.dump(train_info, f)
    
    model = MILModel(model_config).to(device)
    train_dl, test_dl, val_dl = build_dataloader(train_config, num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_criterion = build_train_loss_function(train_config['train_loss_function'])
    test_criterion = build_test_loss_function(train_config['test_loss_function'])

    train_results, test_results = train(model, train_dl, test_dl, optimizer, n_epoch, device, 
                                        train_criterion, test_criterion,
                                        save_dir=save_dir, scheduler=None,
                                        save_per_epoch=save_per_epoch, n_iterations=n_iterations, read_level_info=read_level_info)

    joblib.dump(train_results, os.path.join(save_dir, "train_results.joblib"))      
    joblib.dump(test_results, os.path.join(save_dir, "test_results.joblib"))   
    test_results = joblib.load(os.path.join(save_dir, "test_results.joblib"))
    selection_criteria = ['avg_loss', 'roc_auc', 'pr_auc']
    
    if read_level_info:
        selection_criteria = ['avg_loss', 'avg_loss_read', 'roc_auc', 'pr_auc', 'roc_auc_read', 'pr_auc_read']

    for selection_criterion in selection_criteria:
        test_loss = [test_results[selection_criterion][i] for i in range (0, len(test_results[selection_criterion]), save_per_epoch)]

        if selection_criterion in ('avg_loss', 'avg_loss_read'):
            best_model = (np.argmin(test_loss) + 1) * save_per_epoch
        else:
            best_model = (np.argmax(test_loss) + 1) * save_per_epoch

        model.load_state_dict(torch.load(os.path.join(save_dir, "model_states", str(best_model), "model_states.pt")))
        val_results = validate(model, val_dl, device, test_criterion, n_iterations, read_level_info=read_level_info)
        print("Criteria: {criteria} \t"
              "Compute time: {compute_time:.3f}".format(criteria=selection_criterion, compute_time=val_results["compute_time"]))
        print("Val Loss: {loss:.3f} \t"
              "Val Accuracy: {accuracy:.3f} \t "
              "Val ROC AUC: {roc_auc:.3f} \t "
              "Val PR AUC: {pr_auc:.3f}".format(loss=val_results["avg_loss"],
                                                accuracy=val_results["accuracy"],
                                                roc_auc=val_results["roc_auc"],
                                                pr_auc=val_results["pr_auc"]))
        if read_level_info:
            print("Val Loss read: {loss:.3f} \t"
                  "Val Accuracy read: {accuracy:.3f} \t "
                  "Val ROC AUC read: {roc_auc:.3f} \t "
                  "Val PR AUC read: {pr_auc:.3f}".format(loss=val_results["avg_loss_read"],
                                                         accuracy=val_results["accuracy_read"],
                                                         roc_auc=val_results["roc_auc_read"],
                                                         pr_auc=val_results["pr_auc_read"]))
        print("=====================================")
        joblib.dump(val_results, os.path.join(save_dir, "val_results_{}.joblib".format(selection_criterion)))   


def main():
    args = argparser().parse_args()
    train_and_save(args)

if __name__ == '__main__':
    main()
