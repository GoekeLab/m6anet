import pandas as pd
import numpy as np
import datetime
from torch import nn
from torch.utils.data import DataLoader
from functools import partial


def random_fn(x):
    np.random.seed(datetime.datetime.now().second)


def get_transform_func(transform_func_dict):
    from utils import transform_utils
    transform_func_name = transform_func_dict.pop("transform_name")
    transform_func = getattr(transform_utils, transform_func_name)
    return partial(transform_func, **transform_func_dict)


def get_dataset(site_df, transform_func, dataset_config):
    import utils.dataset as ds
    dataset_class = getattr(ds, dataset_config.pop("dataset_class"))
    return dataset_class(site_df, transform_func=transform_func, **dataset_config) 


def build_dataloader(site_df, train_config, num_workers):
    from utils.dataset import dataset_utils
    
    train_sites = site_df[site_df["set_type"] == 'Train']
    test_sites = site_df[site_df["set_type"] == 'Test']
    val_sites = site_df[site_df["set_type"] == 'Val']

    print("There are {} train sites".format(len(train_sites)))
    print("There are {} test sites".format(len(test_sites)))
    print("There are {} val sites".format(len(val_sites)))

    transform_func = get_transform_func(train_config["dataset"]["transform_function"])

    train_ds = get_dataset(train_sites, transform_func, train_config["dataset"]["train"])
    test_ds = get_dataset(test_sites, transform_func, train_config["dataset"]["test"])
    val_ds = get_dataset(val_sites, transform_func, train_config["dataset"]["val"])

    if "collate_fn" in train_config["dataloader"]:
        collate_fn = getattr(dataset_utils, train_config["dataloader"]["collate_fn"]["collate_fn"])
    else:
        collate_fn = getattr(dataset_utils, "default_collate_fn")
    
    if "sampler" in train_config["dataloader"]["train"]:
        sampler = getattr(dataset_utils, train_config["dataloader"]["train"].pop("sampler"))(train_ds)
    else:
        sampler = None

    train_dl = DataLoader(train_ds, worker_init_fn=random_fn, num_workers=num_workers, 
                          collate_fn=collate_fn, **train_config["dataloader"]["train"],
                          sampler=sampler)
    test_dl = DataLoader(test_ds, worker_init_fn=random_fn, num_workers=num_workers, 
                         collate_fn=collate_fn, **train_config["dataloader"]["test"])
    val_dl = DataLoader(val_ds, worker_init_fn=random_fn, num_workers=num_workers, 
                        collate_fn=collate_fn, **train_config["dataloader"]["val"])
    return train_dl, test_dl, val_dl
    

def build_inference_dataloader(data_config, num_workers):
    from utils.dataset import dataset_utils
    site_df = pd.read_csv(data_config["dataset"].pop("data_path"))
    
    transform_func = get_transform_func(data_config["dataset"].pop("transform_function"))

    ds = get_dataset(site_df, transform_func, data_config["dataset"])

    if "collate_fn" in data_config["dataloader"]:
        collate_fn = getattr(dataset_utils, data_config["dataloader"].pop("collate_fn"))
    else:
        collate_fn = getattr(dataset_utils, "default_collate_fn")
    
    return DataLoader(ds, worker_init_fn=random_fn, num_workers=num_workers, 
                      collate_fn=collate_fn, **data_config["dataloader"])

def build_model(config):
    blocks = config['block']
    seq_model = []
    for block in blocks:
        block_type = block.pop('block_type')
        seq_model.append(_build_block(block_type, **block))
    return nn.Sequential(*seq_model)


def build_train_loss_function(config):
    from utils.loss_functions import train_loss
    loss_function_name = config.pop("loss_function_type")
    return partial(getattr(train_loss, loss_function_name), **config)


def build_test_loss_function(config):
    from utils.loss_functions import test_loss
    loss_function_name = config.pop("loss_function_type")
    return partial(getattr(test_loss, loss_function_name), **config)


