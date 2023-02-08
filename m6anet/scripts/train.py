import numpy as np
import os
import torch
import joblib
import toml
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from ..utils.builder import build_dataloader, build_loss_function
from..utils.constants import DEFAULT_MODEL_CONFIG
from ..utils.training_utils import train, validate
from ..model.model import MILModel


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("--model_config",
                        help='path to model config file.',
                        default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--train_config",
                        help='path to training config file.',
                        required=True)
    parser.add_argument("--save_dir",
                        help='directory to output training results.',
                        required=True)
    parser.add_argument("--device",
                        help='device to perform training with.',
                        default='cpu', type=str)
    parser.add_argument("--lr",
                        help='training learning rate.',
                        default=4e-4, type=float)
    parser.add_argument("--seed",
                        help='random seed for training.',
                        default=25, type=int)
    parser.add_argument("--epochs",
                        help='number of training epochs',
                        default=50, type=int)
    parser.add_argument("--n_processes",
                        help='number of processes to use for training.',
                        default=25, type=int)
    parser.add_argument("--save_per_epoch",
                        help='number of epoch multiple to save training checkpoint.',
                        default=10, type=int)
    parser.add_argument("--weight_decay",
                        help='weight decay argument to regularize training.',
                        default=0, type=float)
    parser.add_argument("--num_iterations",
                        help='number of pass during evaluation step',
                        default=5, type=int)
    return parser


def main(args):

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = args.device
    n_epoch = args.epochs
    lr = args.lr
    save_per_epoch = args.save_per_epoch
    save_dir = args.save_dir
    weight_decay = args.weight_decay
    n_iterations = args.num_iterations

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
    train_info["train_config"]["seed"] = seed

    with open(os.path.join(save_dir, "train_info.toml"), 'w', encoding='utf-8') \
            as f:
        toml.dump(train_info, f)

    model = MILModel(model_config).to(device)
    train_dl, val_dl, test_dl = build_dataloader(train_config, args.n_processes)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    criterion = build_loss_function(train_config['loss_function'])

    train_results, val_results = train(model, train_dl, val_dl, optimizer, n_epoch, device,
                                       criterion, save_dir=save_dir,
                                       save_per_epoch=save_per_epoch,
                                       n_iterations=n_iterations)

    joblib.dump(train_results, os.path.join(save_dir, "train_results.joblib"))
    joblib.dump(val_results, os.path.join(save_dir, "val_results.joblib"))

    selection_criteria = ['avg_loss', 'roc_auc', 'pr_auc']

    for selection_criterion in selection_criteria:
        val_loss = [val_results[selection_criterion][i]
                    for i in range(0, len(val_results[selection_criterion]), save_per_epoch)]

        if selection_criterion in ('avg_loss', 'avg_loss_read'):
            best_model = (np.argmin(val_loss) + 1) * save_per_epoch
        else:
            best_model = (np.argmax(val_loss) + 1) * save_per_epoch

        state_dict = torch.load(os.path.join(save_dir, "model_states", str(best_model), "model_states.pt"))
        torch.save(state_dict, os.path.join(save_dir, "{}.pt".format(selection_criterion)))

        model.load_state_dict(state_dict)
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
