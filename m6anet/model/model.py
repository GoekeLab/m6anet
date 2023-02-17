import torch
from torch.nn import Module, Identity, Sequential
from typing import Dict, Tuple
from .model_blocks.pooling_blocks import PoolingFilter


class MILModel(Module):
    r"""
    The main m6Anet model, used to construct m6Anet model from a configuration file

    ...

    Attributes
    -----------
    model_config (Dict): Dictionary containing configurations to build this model
    read_level_encoder (nn.Module): PyTorch module to encode read level features into high dimensional vectors / read level probability
    pooling_filter (nn.Module): PyTorch module to pool read level representation into site level representation
    decoder (nn.Module): PyTorch module to predict site level probability from the output of pooling_filter
    """
    def __init__(self, model_config: Dict):
        r'''
        Initialization function for the class

                Args:
                        model_config (Dict): A dictionary containing model configurations (see m6anet/model/configs/model_configs/m6anet.toml for an example of model config file)

                Returns:
                        None
        '''
        super(MILModel, self).__init__()

        self.model_config = model_config
        self.read_level_encoder = None
        self.pooling_filter = None
        self.decoder = None

        # Building model block by block
        self.build_model()

    def build_model(self):
        r'''
        Instance method to build the model components (read_level_encoder, pooling_filter, and decoder) from configuration file
        '''
        blocks = self.model_config['block']
        seq_model = []
        for block in blocks:
            block_type = block.pop('block_type')
            block_obj = self._build_block(block_type, **block)

            if isinstance(block_obj, PoolingFilter):
                if len(seq_model) > 0:
                    self.read_level_encoder = Sequential(*seq_model)
                else:
                    self.read_level_encoder = None

                self.pooling_filter = block_obj
                seq_model = []
            else:
                seq_model.append(block_obj)

        if (self.read_level_encoder is None) and (self.pooling_filter is None):
            self.read_level_encoder = Sequential(*seq_model)
            self.pooling_filter = Identity()
            self.decoder = Identity()
        else:
            if len(seq_model) == 0:
                self.decoder = Identity()
            else:
                self.decoder = Sequential(*seq_model)

    def _build_block(self, block_type: str, **kwargs: Dict) -> Module:
        r'''
        Instance method to parse the string block_type and return the corresponding module class

                Args:
                        block_type (str): A string corresponding to the class name in .model_blocks module
                        kwargs (dict): Dictionary containing additional keyword arguments for the model blocks
                Returns:
                        block_obj (Module): PyTorch compatible neural networks Module for m6Anet
        '''
        from . import model_blocks
        block_obj = getattr(model_blocks, block_type)
        return block_obj(**kwargs)

    def get_read_representation(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Instance method to get read level representation from read level features

                Args:
                        x (torch.Tensor): A tensor representation of the read level features
                Returns:
                        (torch.Tensor): A high dimensional tensor containing read-level representation
        '''
        if self.read_level_encoder is None:
            return x
        else:
            return self.read_level_encoder(x)

    def get_read_probability(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Instance method to get modification probability on the read level from read features

                Args:
                        x (torch.Tensor): A tensor representation of the read level features
                Returns:
                        (torch.Tensor): A tensor containing modification probability on the read level
        '''
        read_representation = self.get_read_representation(x)
        return self.pooling_filter.predict_read_level_prob(read_representation)

    def get_site_representation(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Instance method to get site level representation from read features

                Args:
                        x (torch.Tensor): A tensor representation of the read level features
                Returns:
                        (torch.Tensor): A high dimensional tensor containing site-level representation
        '''
        return self.pooling_filter(self.get_read_representation(x))

    def get_site_probability(self, x):
        r'''
        Instance method to get modification probability on the site level from read features

                Args:
                        x (torch.Tensor): A tensor representation of the read level features
                Returns:
                        (torch.Tensor): A tensor containing modification probability on the site level
        '''
        return self.decoder(self.get_site_representation(x))

    def get_read_site_probability(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r'''
        Instance method to get modification probability on both read and site level from read features and also
        the high dimensional representation for each read

                Args:
                        x (torch.Tensor): A tensor representation of the read level features
                Returns:
                        (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing three set of tensors which represent the modification probability at read level, site level,
                                                                           and the high dimensional representation of the read features
        '''
        read_representation = self.get_read_representation(x)
        read_level_probability = self.pooling_filter.predict_read_level_prob(read_representation)
        site_level_probability = self.decoder(self.pooling_filter(read_representation))
        return read_level_probability, site_level_probability, read_representation

    def get_attention_weights(self, x):
        if hasattr(self.pooling_filter, "get_attention_weights"):
            return self.pooling_filter.get_attention_weights(self.get_read_representation(x))
        else:
            raise ValueError("Pooling filter does not have attention weights")

    def forward(self, x):
        r'''
        Instance method to get modification probability on the site level from read features.

                Args:
                        x (torch.Tensor): A tensor representation of the read level features
                Returns:
                        (torch.Tensor): A tensor containing modification probability on the site level
        '''
        return self.get_site_probability(x)
