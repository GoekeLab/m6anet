r"""
This module is a collection of builder functions used during training and inference for building Python object from configuration files
"""
import numpy as np
import datetime
from copy import deepcopy
from torch.utils.data import DataLoader
from functools import partial
from .data_utils import NanopolishDS, NanopolishReplicateDS, train_collate
from typing import Dict, Optional, Tuple, Union


def random_fn(x: int):
    r'''
    worker_init_fn for PyTorch Dataloader class to randomize the seed for different worker's id

            Args:
                    x (int): an integer describing the worker id

            Returns:
                    None
     '''
    np.random.seed(datetime.datetime.now().second)


def build_dataset(config: Dict, mode: Optional[str] = None) -> Union[NanopolishDS, NanopolishReplicateDS]:
    r'''
    Build PyTorch compatible dataset from configuration file

            Args:
                    config (dictionary): A dictionary constructed from configuration file containing dataset parameters
                    mode (str): A string describing running mode, must be one of ('Train', 'Test', 'Val', 'Inference') (default is None)

            Returns:
                    A NanopolishDS dataset or NanopolishReplicateDS dataset if config file contains more than one sample directory

            Raises:
                    ValueError: Raises an exception when some of the dictionary keys are not present / not in the correct format
     '''
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


def build_dataloader(train_config: Dict, num_workers: int, verbose: Optional[bool] = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    r'''
    Build PyTorch compatible datasets from training configuration file

            Args:
                    train_config (dictionary): A dictionary constructed from configuration file containing training parameters
                    num_workers (int): Number of processes passed to PyTorch DataLoader constructor
                    verbose (bool): Boolean value indicating whether to print the number of sites present in the trainining, validation, and test splits
                    mode (str): A string describing running mode, must be one of ('Train', 'Test', 'Val', 'Inference')

            Returns:
                    train_dl (DataLoader) : A PyTorch DataLoader object used for training m6Anet model
                    val_dl (DataLoader) : A PyTorch DataLoader object for validation of m6Anet model
                    test_dl (DataLoader) : A PyTorch DataLoader object for testing of m6Anet model
     '''
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


def build_loss_function(config: Dict):
    r'''
    Build PyTorch compatible loss function from configuration file

            Args:
                    config (dictionary): A dictionary constructed from configuration file containing loss function parameters

            Returns:
                    PyTorch loss function as specified in the config file. See m6anet.utils.loss_functions.loss_functions for all available loss functions

            Raises:
                    ValueError: Raises an exception when config does not contain the loss_function_type key
     '''
    if "loss_function_type" not in config:
        raise ValueError("Config must specify loss_function_type")
    from .loss_functions import loss_functions
    loss_function_name = config.pop("loss_function_type")
    return partial(getattr(loss_functions, loss_function_name), **config)
