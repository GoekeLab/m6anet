r"""
This module is a collection of m6Anet building blocks
"""
import torch
from torch import nn
from typing import Dict, Optional


def get_activation(activation: str) -> torch.nn.Module:
    r'''
    Instance method to get modification probability on the site level from read features.

            Args:
                    activation (str): A string that corresponds to the desired activation function. Must be one of ('tanh', 'sigmoid', 'relu', 'softmax')
            Returns:
                    activation_func (torch.nn.Module): A PyTorch activation function
    '''
    allowed_activation = ('tanh', 'sigmoid', 'relu', 'softmax')
    activation_func = None
    if activation == 'tanh':
        activation_func = nn.Tanh()
    elif activation == 'sigmoid':
        activation_func = nn.Sigmoid()
    elif activation == 'relu':
        activation_func = nn.ReLU()
    elif activation == 'softmax':
        activation_func = nn.Softmax(dim=1)
    else:
        raise ValueError("Invalid activation, must be one of {}".format(allowed_activation))

    return activation_func


class Block(nn.Module):
    r"""
    The basic building block of m6Anet model
    """
    def __init__(self):
        super(Block, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        The forward method should be implemented in subclasses
        """
        raise NotImplementedError("Subclasses should implement this method")


class ConcatenateFeatures(Block):
    r"""
    Block object to concatenate several different features (i.e, k-mer embedding and signal features) into one high dimensional tensor
    """
    def __init__(self):
        super(ConcatenateFeatures, self).__init__()

    def forward(self, x: Dict) -> torch.Tensor:
        r'''
        Instance method to concatenate all tensor values in dictionary as one high dimensional tensor

                Args:
                        x (dict): Dictionary containing tensor values

                Returns:
                        x (torch.Tensor): PyTorch tensor from concatenating all values in the dictionary input
        '''
        x = torch.cat([val for _, val in x.items()], axis=1)
        return x


class ExtractSignal(Block):
    r"""
    Block object to extract only the signal features from input argument
    """
    def __init__(self):
        super(ExtractSignal, self).__init__()

    def forward(self, x):
        r'''
        Instance method to extract only the signal features from the input. The signal value must have the key 'X' in the dictionary input

                Args:
                        x (dict): Dictionary containing tensor values

                Returns:
                        x (torch.Tensor): PyTorch tensor containing the signal value corresponding to the key 'X' in the input dictionary
        '''
        return x['X']


class DeaggregateNanopolish(Block):
    r"""
    Block object to reshape both signal and sequence features from nanopolish preprocessed input

    ...

    Attributes
    -----------
    num_neighboring_features (int): Number of flanking positions included in the features
    n_features (int): Number of features in the signal data
    """
    def __init__(self, num_neighboring_features: int, n_features: Optional[int] = 3):
        r'''
        Initialization function for the class

                Args:
                        num_neighboring_features (int): Number of flanking positions included in the features
                        n_features (int): Number of features in the signal data

                Returns:
                        None
        '''
        super(DeaggregateNanopolish, self).__init__()
        self.num_neighboring_features = num_neighboring_features
        self.n_features = n_features * (2 * self.num_neighboring_features + 1)


    def forward(self, x: Dict) -> Dict:
        r'''
        Instance method to split features from nanopolish preprocessed input into signal and sequence features in a Python dictionary

                Args:
                        x (Dict): Python dictionary containing signal features and sequence features

                Returns:
                        (dict): Python dictionary containing signal features and sequence features
        '''
        return {'X': x['X'].view(-1, self.n_features), 'kmer':  x['kmer'].view(-1, 1)}


class Flatten(Block):
    r"""
    Block object that acts as a wrapper for torch.nn.Flatten
    ...

    Attributes
    -----------
    layers (nn.Module): PyTorch nn.Flatten
    """
    def __init__(self, start_dim, end_dim):
        r'''
        Initialization function for the class

                Args:
                        start_dim (int): Starting dimension to flatten
                        end_dim (int): Ending dimension to flatten

                Returns:
                        None
        '''
        super(Flatten, self).__init__()
        self.layers = nn.Flatten(start_dim, end_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Instance method to flatten the target tensor

                Args:
                        x (torch.Tensor): Tensor input

                Returns:
                        x (torch.Tensor): Flattened tensor output according to the specified start_dim and end_dim during initialization
        '''
        return self.layers(x)


class KmerMultipleEmbedding(Block):
    r"""
    Block object that applies PyTorch embedding layer to sequence information from nanopolish input
    ...

    Attributes
    -----------
    input_channel (int): Number of unique 5-mer motifs to be embedded
    output_channel (int): Output dimension of the transformed 5-mer motif
    embedding_layer (nn.Module): PyTorch nn.Embedding layer to transform categorical variable into vectors
    n_features (int): Number of features in the signal data
    """
    def __init__(self, input_channel: int, output_channel:int, num_neighboring_features: Optional[int] = 1):
        r'''
        Initialization function for the class

                Args:
                        input_channel (int): Number of unique 5-mer motifs to be embedded
                        output_channel (int): Output dimension of the transformed 5-mer motif
                        num_neighboring_features (int): Number of flanking positions around the target site

                Returns:
                        None
        '''
        super(KmerMultipleEmbedding, self).__init__()
        self.input_channel, self.output_channel = input_channel, output_channel
        self.embedding_layer = nn.Embedding(input_channel, output_channel)
        self.n_features = 2 * num_neighboring_features + 1

    def forward(self, x: Dict) -> Dict:
        r'''
        Instance method to apply embedding layer on sequence features, transforming them into high dimensional vector representation

                Args:
                        x (dict): Python dictionary containing signal features and sequence feature

                Returns:
                        (dict): Python dictionary containing signal features and transformed sequence features
        '''
        kmer =  x['kmer']
        return {'X': x['X'], 'kmer' :self.embedding_layer(kmer.long()).reshape(-1, self.n_features * self.output_channel)}


class Linear(Block):
    r"""
    Block object that applies PyTorch Linear layer, BatchNorm and Dropout
    ...

    Attributes
    -----------
    layers (nn.Module): A sequence of PyTorch Module classes
    """
    def __init__(self, input_channel, output_channel, activation='relu', batch_norm=True, dropout=0.0):
        r'''
        Initialization function for the class

                Args:
                        input_channel (int): Number of input dimension
                        output_channel (int): Number of output dimension
                        activation (str): Activation function
                        batch_norm (bool): Whether to use BatchNorm or not
                        dropout (float): Dropout value

                Returns:
                        None
        '''
        super(Linear, self).__init__()
        self.layers = self._make_layers(input_channel, output_channel, activation, batch_norm, dropout)

    def _make_layers(self, input_channel: int, output_channel: int, activation: str, batch_norm: bool, dropout: Optional[float] = 0.0):
        r'''
        Function to construct PyTorch Sequential object for this class, which comprised of a single
        Linear layer along with BatchNorm1d and possibly a dropout

                Args:
                        input_channel (int): Number of input dimension
                        output_channel (int): Number of output dimension
                        activation (str): Activation function
                        batch_norm (bool): Whether to use BatchNorm or not
                        dropout (float): Dropout value

                Returns:
                        None
        '''
        layers = [nn.Linear(input_channel, output_channel)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(num_features=output_channel))
        if activation is not None:
            layers.append(get_activation(activation))
        layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Instance method to apply linear layer on tensor features

                Args:
                        x (torch.Tensor): Tensor input
                Returns:
                        (torch.Tensor): Transformed tensor output
        '''
        return self.layers(x)
