r"""
This module is a collection of pooling layers for m6Anet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .blocks import get_activation
from typing import Optional


class PoolingFilter(nn.Module):
    r"""
    The abstract class of m6Anet pooling filter layer
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def predict_read_level_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def set_num_reads(self, n_reads_per_site: int):
        self.n_reads_per_site = n_reads_per_site


class InstanceBasedPooling(PoolingFilter):
    r"""
    An abstract class for instance based pooling approach
    ...

    Attributes
    -----------
    input_channel (int): The input dimension of the read features passed to this module
    n_reads_per_site (int): Number of reads expressed for each transcriptomic site
    probability_layer (nn.Module): A PyTorch linear layer that outputs read-level probability for each read
    """
    def __init__(self, input_channel: int, n_reads_per_site: Optional[int] = 20):
        r'''
        Initialization function for the class

                Args:
                    input_channel (int): The input dimension of the read features passed to this module
                    n_reads_per_site (int): Number of reads expressed for each transcriptomic site
                    probability_layer (nn.Module): A PyTorch linear layer that outputs read-level probability for each read

                Returns:
                        None
        '''
        super(PoolingFilter, self).__init__()
        self.input_channel = input_channel
        self.n_reads_per_site = n_reads_per_site
        self.probability_layer = nn.Sequential(*[nn.Linear(input_channel, 1), get_activation('sigmoid')])

    def predict_read_level_prob(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Instance based method that takes in transform high dimensional read features and output modification probability for each read

                Args:
                    x (torch.Tensor): The input read-level tensor representation

                Returns:
                        (torch.Tensor): The output read level modification probability
        '''
        return self.probability_layer(x).view(-1, self.n_reads_per_site)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class SigmoidMeanPooling(InstanceBasedPooling):
    r"""
    An average pooling layer that computes site probability by averaging the probability that each read is modified

    ...

    Attributes
    -----------
    input_channel (int): The input dimension of the read features passed to this module
    n_reads_per_site (int): Number of reads expressed for each transcriptomic site
    probability_layer (nn.Module): A PyTorch linear layer that outputs read-level probability for each read
    """
    def __init__(self, input_channel: int, n_reads_per_site: Optional[int] = 20):
        r'''
        Initialization function for the class

                Args:
                    input_channel (int): The input dimension of the read features passed to this module
                    n_reads_per_site (int): Number of reads expressed for each transcriptomic site
                    probability_layer (nn.Module): A PyTorch linear layer that outputs read-level probability for each read

                Returns:
                        None
        '''
        super(SigmoidMeanPooling, self).__init__(input_channel, n_reads_per_site)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        read_level_prob = self.predict_read_level_prob(x)
        return torch.mean(read_level_prob, axis=1)


class SigmoidProdPooling(InstanceBasedPooling):
    r"""
    A noisy-OR pooling layer that computes site probability by calculating the probability that at least one read is modified

    ...

    Attributes
    -----------
    input_channel (int): The input dimension of the read features passed to this module
    n_reads_per_site (int): Number of reads expressed for each transcriptomic site
    probability_layer (nn.Module): A PyTorch linear layer that outputs read-level probability for each read
    """
    def __init__(self, input_channel: int, n_reads_per_site: Optional[int] = 20):
        r'''
        Initialization function for the class

                Args:
                    input_channel (int): The input dimension of the read features passed to this module
                    n_reads_per_site (int): Number of reads expressed for each transcriptomic site
                    probability_layer (nn.Module): A PyTorch linear layer that outputs read-level probability for each read

                Returns:
                        None
        '''
        super(SigmoidProdPooling, self).__init__(input_channel, n_reads_per_site)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        read_level_prob = self.predict_read_level_prob(x)
        return 1 - torch.prod(1 - read_level_prob, axis=1)


class SigmoidMaxPooling(InstanceBasedPooling):
    r"""
    A max pooling layer that computes site probability by taking the maximum probability of a read being modified

    ...

    Attributes
    -----------
    input_channel (int): The input dimension of the read features passed to this module
    n_reads_per_site (int): Number of reads expressed for each transcriptomic site
    probability_layer (nn.Module): A PyTorch linear layer that outputs read-level probability for each read
    """
    def __init__(self, input_channel: int, n_reads_per_site: Optional[int] = 20):
        r'''
        Initialization function for the class

                Args:
                    input_channel (int): The input dimension of the read features passed to this module
                    n_reads_per_site (int): Number of reads expressed for each transcriptomic site
                    probability_layer (nn.Module): A PyTorch linear layer that outputs read-level probability for each read

                Returns:
                        None
        '''
        super(SigmoidMaxPooling, self).__init__(input_channel, n_reads_per_site)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        read_level_prob = self.predict_read_level_prob(x)
        return torch.max(read_level_prob, axis=1).values


class SummaryStatsAggregator(PoolingFilter):

    def __init__(self, input_channel, n_reads_per_site=20):
        super(SummaryStatsAggregator, self).__init__()
        self.input_channel = input_channel
        self.n_reads_per_site = n_reads_per_site

    def aggregate(self, x):
        x = x.view(-1, self.n_reads_per_site, self.input_channel)
        mean = torch.mean(x, axis=1)
        var = torch.var(x, axis=1)
        max_ = torch.max(x, axis=1).values
        min_ = torch.min(x, axis=1).values
        med_ = torch.median(x, axis=1).values
        x = torch.cat([mean, var, max_, min_, med_], axis=1)
        return x

    def forward(self, x):
        is_dict = isinstance(x, dict)

        if is_dict:
            x, kmer = x['X'], x['kmer']
        x = self.aggregate(x)

        if is_dict:
            return {'X': x, 'kmer': kmer}
        else:
            return x


class MeanAggregator(PoolingFilter):

    def __init__(self, input_channel, n_reads_per_site=20):
        super(MeanAggregator, self).__init__()
        self.input_channel = input_channel
        self.n_reads_per_site = n_reads_per_site

    def aggregate(self, x):
        x = x.view(-1, self.n_reads_per_site, self.input_channel)
        mean = torch.mean(x, axis=1)
        return mean

    def forward(self, x):
        is_dict = isinstance(x, dict)

        if is_dict:
            x, kmer = x['X'], x['kmer']
        x = self.aggregate(x)

        if is_dict:
            return {'X': x, 'kmer': kmer}
        else:
            return x


class Attention(PoolingFilter):

    def __init__(self, input_channel, hidden_layers, activation='relu', n_reads_per_site=20):
        super(PoolingFilter, self).__init__()

        self.hidden_layers = hidden_layers

        self.input_channel = input_channel
        self.output_channel = self.hidden_layers[-1]

        self.activation = activation
        self.attention = self._create_attention_layers(input_channel, hidden_layers, activation)

        self.n_reads_per_site = n_reads_per_site

    def _create_attention_layers(self, input_channel, hidden_layers, activation):
        prev_dim = input_channel
        layers = []

        for curr_dim in hidden_layers[:-1]:
            layers.append(nn.Linear(prev_dim, curr_dim))
            layers.append(get_activation(activation))
            prev_dim = curr_dim

        layers.append(nn.Linear(curr_dim, hidden_layers[-1]))
        return nn.Sequential(*layers)

    def forward(self, x):
        W = self.get_attention_weights(x)  # non-linear transformation

        # Linear combination of the original dimension
        M = torch.matmul(W, x.view(-1, self.n_reads_per_site, self.input_channel))  # KxL
        return nn.Flatten()(M)

    def predict_read_level_prob(self, x):
        return self.get_attention_weights(x)

    def get_attention_weights(self, x):
        W = self.attention(x)
        # Creating weight matrix
        W = W.view(-1, self.n_reads_per_site, self.output_channel)
        W = torch.transpose(W, 2, 1)  # KxN
        W = F.softmax(W, dim=2)  # softmax over N
        return W


class ProbabilityAttention(PoolingFilter):

    def __init__(self, input_channel, hidden_layers_1, hidden_layers_2,
                 n_bins, sigma, activation='relu', n_reads_per_site=20,
                 read_classifier='prod_pooling'):
        super(PoolingFilter, self).__init__()
        self.site_decoder = KDEGatedAttentionLayer(input_channel, hidden_layers_1, hidden_layers_2,
                                                   n_bins, sigma, activation, n_reads_per_site)
        if read_classifier == 'prod_pooling':
            self.read_classifier = SigmoidProdPooling(input_channel,  n_reads_per_site)
        elif read_classifier == 'mean_pooling':
            self.read_classifier = SigmoidMeanPooling(input_channel, n_reads_per_site)
        elif read_classifier == 'max_pooling':
            self.read_classifier = SigmoidMaxPooling(input_channel, n_reads_per_site)
        else:
            raise ValueError("Invalid read classifier name")

    def forward(self, x):
        return self.site_decoder(x)

    def predict_read_level_prob(self, x):
        return self.read_classifier.predict_read_level_prob(x)

    def get_attention_weights(self, x):
        return self.site_decoder.get_attention_weights(x)


class SummaryStatsProbability(PoolingFilter):

    def __init__(self, input_channel, n_reads_per_site=20,
                 read_classifier='prod_pooling'):
        super(PoolingFilter, self).__init__()
        self.site_decoder = SummaryStatsAggregator(input_channel, n_reads_per_site=20)
        if read_classifier == 'prod_pooling':
            self.read_classifier = SigmoidProdPooling(input_channel,  n_reads_per_site)
        elif read_classifier == 'mean_pooling':
            self.read_classifier = SigmoidMeanPooling(input_channel, n_reads_per_site)
        elif read_classifier == 'max_pooling':
            self.read_classifier = SigmoidMaxPooling(input_channel, n_reads_per_site)
        else:
            raise ValueError("Invalid read classifier name")

    def forward(self, x):
        return self.site_decoder(x)

    def predict_read_level_prob(self, x):
        return self.read_classifier.predict_read_level_prob(x)


class GatedAttention(Attention):

    def __init__(self, input_channel, hidden_layers_1,
                 hidden_layers_2, activation='relu',
                 n_reads_per_site=20):
        super(Attention, self).__init__()

        self.input_channel = input_channel
        self.hidden_layers_1 = hidden_layers_1
        self.hidden_layers_2 = hidden_layers_2
        self.n_reads_per_site = n_reads_per_site

        self.activation = activation
        self.attention_v = self._create_attention_layers(input_channel, hidden_layers_1, activation)
        self.attention_h = self._create_attention_layers(input_channel, hidden_layers_1, 'sigmoid')
        self.attention = Attention(hidden_layers_1[-1], hidden_layers_2, activation, self.n_reads_per_site)

    def forward(self, x):
        a_v = self.attention_v(x)
        a_h = self.attention_h(x)
        return self.attention(a_v * a_h)

    def predict_read_level_prob(self, x):
        a_v = self.attention_v(x)
        a_h = self.attention_h(x)
        return self.attention.predict_read_level_prob(a_v * a_h)

    def get_attention_weights(self, x):
        a_v = self.attention_v(x)
        a_h = self.attention_h(x)
        return self.attention.get_attention_weights(a_v * a_h)


class KDELayer(PoolingFilter):

    def __init__(self, input_channel, n_bins, sigma,
                 n_reads_per_site=20):
        super(KDELayer, self).__init__()

        self.input_channel = input_channel
        self.n_bins = n_bins
        self.var = sigma ** 2
        self.n_reads_per_site = n_reads_per_site

    def forward(self, x):
        x = x.view(-1, self.n_reads_per_site, self.input_channel)
        kde_vectors = []
        for v in torch.linspace(0, 1, self.n_bins):
             kde_vectors.append(torch.mean((1 / math.sqrt(2 * math.pi * self.var)) * torch.exp((-1 / (2 * self.var)) * torch.pow(x - v, 2)), axis=1))
        return torch.cat(kde_vectors, axis=1)


class KDEAttentionLayer(PoolingFilter):

    def __init__(self, input_channel, hidden_layers,
                 n_bins, sigma, activation='relu', n_reads_per_site=20):
        super(PoolingFilter, self).__init__()

        self.input_channel = input_channel
        self.n_bins = n_bins
        self.var = sigma ** 2
        self.n_reads_per_site = n_reads_per_site
        self.attention = Attention(input_channel,  hidden_layers, activation,  n_reads_per_site)

    def forward(self, x):
        x = x.view(-1, self.n_reads_per_site, self.input_channel)
        kde_vectors = []
        for v in torch.linspace(0, 1, self.n_bins):
            kde_vectors.append(self.attention((1 / math.sqrt(2 * math.pi * self.var)) * torch.exp((-1 / (2 * self.var)) * torch.pow(x - v, 2))))
        return torch.cat(kde_vectors, axis=1)

    def predict_read_level_prob(self, x):
        return self.attention.predict_read_level_prob(x)

class KDEGatedAttentionLayer(PoolingFilter):

    def __init__(self, input_channel, hidden_layers_1, hidden_layers_2,
                 n_bins, sigma, activation='relu', n_reads_per_site=20):
        super(PoolingFilter, self).__init__()

        self.input_channel = input_channel
        self.n_bins = n_bins
        self.var = sigma ** 2
        self.hidden_layers_1 = hidden_layers_1
        self.hidden_layers_2 = hidden_layers_2
        self.n_reads_per_site = n_reads_per_site
        self.gated_attention = GatedAttention(input_channel,  hidden_layers_1, hidden_layers_2, activation,  n_reads_per_site)

    def forward(self, x):
        x = x.view(-1, self.n_reads_per_site, self.input_channel)
        kde_vectors = []
        for v in torch.linspace(0, 1, self.n_bins):
            kde_vectors.append(self.gated_attention((1 / math.sqrt(2 * math.pi * self.var)) * torch.exp((-1 / (2 * self.var)) * torch.pow(x - v, 2))))
        return torch.cat(kde_vectors, axis=1)

    def predict_read_level_prob(self, x):
        return self.gated_attention.predict_read_level_prob(x)

    def get_attention_weights(self, x):
        return self.gated_attention.get_attention_weights(x)
