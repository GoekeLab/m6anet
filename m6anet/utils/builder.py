import pandas as pd
import numpy as np
import datetime
from torch import nn
from torch.utils.data import DataLoader
from functools import partial
from .data_utils import NanopolishDS, train_collate


def random_fn(x):
    np.random.seed(datetime.datetime.now().second)


def build_dataloader(train_config, num_workers, verbose=True):
    
    train_ds = NanopolishDS(**train_config["dataset"], mode='Train')
    test_ds = NanopolishDS(**train_config["dataset"], mode='Test')
    val_ds = NanopolishDS(**train_config["dataset"], mode='Val')
    
    if verbose:
        print("There are {} train sites".format(len(train_ds)))
        print("There are {} test sites".format(len(test_ds)))
        print("There are {} val sites".format(len(val_ds)))

    if "sampler" in train_config["dataloader"]["train"]:
        from . import data_utils
        sampler = getattr(data_utils, train_config["dataloader"]["train"].pop("sampler"))(train_ds)
    else:
        sampler = None

    train_dl = DataLoader(train_ds, worker_init_fn=random_fn, num_workers=num_workers, 
                          collate_fn=train_collate, **train_config["dataloader"]["train"],
                          sampler=sampler)
    test_dl = DataLoader(test_ds, worker_init_fn=random_fn, num_workers=num_workers, 
                         collate_fn=train_collate, **train_config["dataloader"]["test"])
    val_dl = DataLoader(val_ds, worker_init_fn=random_fn, num_workers=num_workers, 
                        collate_fn=train_collate, **train_config["dataloader"]["val"])
    return train_dl, test_dl, val_dl
    

def build_train_loss_function(config):
    from .loss_functions import train_loss
    loss_function_name = config.pop("loss_function_type")
    return partial(getattr(train_loss, loss_function_name), **config)


def build_test_loss_function(config):
    from .loss_functions import test_loss
    loss_function_name = config.pop("loss_function_type")
    return partial(getattr(test_loss, loss_function_name), **config)
