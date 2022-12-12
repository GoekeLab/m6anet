import pandas as pd
import numpy as np
import datetime
from copy import deepcopy
from torch import nn
from torch.utils.data import DataLoader
from functools import partial
from .data_utils import NanopolishDS, NanopolishReplicateDS, train_collate


def random_fn(x):
    np.random.seed(datetime.datetime.now().second)


def build_dataset(config, mode=None):
    if 'root_dir' in config:
        root_dir = config['root_dir']
        if isinstance(root_dir, list):
            if len(root_dir) > 1:
                return NanopolishReplicateDS(**config, mode=mode)
            else:
                raise ValueError("root_dir is a list but of size 1, please pass root_dir as a string instead")
        elif isinstance(root_dir, str):
            return NanopolishDS(**config, mode=mode)
        else:
            raise ValueError("Invalid type for argument root_dir")
    else:
        if 'data_info' not in config:
            raise ValueError("Must pass either root_dir or data_info in toml files")

        if 'replicate' not in config:
            return NanopolishDS(**config, mode=mode)
        else:
            config = deepcopy(config)
            replicate = config.pop('replicate')
            if replicate:
                return NanopolishReplicateDS(**config, mode=mode)
            else:
                return NanopolishDS(**config, mode=mode)

def build_dataloader(train_config, num_workers, verbose=True):

    train_ds = build_dataset(train_config["dataset"], mode='Train')
    val_ds = build_dataset(train_config["dataset"], mode='Val')
    test_ds = build_dataset(train_config["dataset"], mode='Test')

    if verbose:
        print("There are {} train sites".format(len(train_ds)))
        print("There are {} val sites".format(len(val_ds)))
        print("There are {} test sites".format(len(test_ds)))

    if "sampler" in train_config["dataloader"]["train"]:
        from . import sampler_utils
        sampler = getattr(sampler_utils, train_config["dataloader"]["train"].pop("sampler"))(train_ds)
    else:
        sampler = None

    train_dl = DataLoader(train_ds, worker_init_fn=random_fn, num_workers=num_workers,
                          collate_fn=train_collate, **train_config["dataloader"]["train"],
                          sampler=sampler)
    val_dl = DataLoader(val_ds, worker_init_fn=random_fn, num_workers=num_workers,
                        collate_fn=train_collate, **train_config["dataloader"]["val"])
    test_dl = DataLoader(test_ds, worker_init_fn=random_fn, num_workers=num_workers,
                         collate_fn=train_collate, **train_config["dataloader"]["test"])

    return train_dl, val_dl, test_dl


def build_loss_function(config):
    from .loss_functions import loss_functions
    loss_function_name = config.pop("loss_function_type")
    return partial(getattr(loss_functions, loss_function_name), **config)
