import torch
from torch import nn


def get_activation(activation):
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
        raise ValueError("Invalid activation")
    
    return activation_func


class Block(nn.Module):

    def __init__(self):
        super(Block, self).__init__()
    
    def forward(self, x):
        return self.layers(x)


class PoolingFilter(nn.Module):
    
    def forward(self, x):
        return x

    def predict_read_level_prob(self, x):
        return self.forward(x)


class ConcatenateFeatures(Block):

    def __init__(self):
        super(ConcatenateFeatures, self).__init__()

    def forward(self, x):
        x = torch.cat([val for _, val in x.items()], axis=1)
        return x


class ExtractSignal(Block):

    def __init__(self):
        super(ExtractSignal, self).__init__()

    def forward(self, x):
        return x['X']


class DeaggregateNanopolish(Block):

    def __init__(self, num_neighboring_features, n_features=3):
        super(DeaggregateNanopolish, self).__init__()
        self.num_neighboring_features = num_neighboring_features
        self.n_features = n_features * (2 * self.num_neighboring_features + 1)


    def forward(self, x):
        return {'X': x['X'].view(-1, self.n_features), 'kmer':  x['kmer'].view(-1, 1)}


class Flatten(Block):

    def __init__(self, start_dim, end_dim):
        super(Flatten, self).__init__()
        self.layers = nn.Flatten(start_dim, end_dim)


class KmerMultipleEmbedding(Block):

    def __init__(self, input_channel, output_channel, num_neighboring_features=0):
        super(KmerMultipleEmbedding, self).__init__()
        self.input_channel, self.output_channel = input_channel, output_channel
        self.embedding_layer = nn.Embedding(input_channel, output_channel)
        self.n_features = 2 * num_neighboring_features + 1

    def forward(self, x):
        kmer =  x['kmer']
        return {'X': x['X'], 'kmer' :self.embedding_layer(kmer.long()).reshape(-1, self.n_features * self.output_channel)}


class Linear(Block):

    def __init__(self, input_channel, output_channel, activation='relu', batch_norm=True, dropout=0.0):
        super(Linear, self).__init__()
        self.layers = self._make_layers(input_channel, output_channel, activation, batch_norm)
    
    def _make_layers(self, input_channel, output_channel, activation, batch_norm, dropout=0.0):
        layers = [nn.Linear(input_channel, output_channel)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(num_features=output_channel))
        if activation is not None:
            layers.append(get_activation(activation))
        layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)
