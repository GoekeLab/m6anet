r"""
This module is a collection sampler classes used during training of m6Anet
"""
import torch
import numpy as np
from .data_utils import NanopolishDS


class ImbalanceUnderSampler(torch.utils.data.Sampler):
    r"""
    A PyTorch Sampler class for maintaining balanced m6Anet training through undersampling the majority modification class in the dataset. This class is used
    for m6Anet training, producing undersampled sites corresponding to the number of minority classes in the dataset

    ...

    Attributes
    -----------
    data_source (NanopolishDS): An instance of NanopolishDS class
    class_counts (np.ndarray): A NumPy array containing the counts for all unique labels in the dataset
    minority_class (np.int64): Label for the minority class
    minority_class_idx (np.ndarray): Indices on data_source that corresponds to the minority class instances
    majority_class (np.int64): Label for the majority class
    majority_class_idx (np.ndarray): Indices on data_source that corresponds to the majority_class_idx class instances
    """
    def __init__(self, data_source: NanopolishDS):
        r'''
        Initialization function for the class

                Args:
                        data_source (NanopolishDS): An instance of NanopolishDS class

                Returns:
                        None
        '''
        self.data_source = data_source
        self.class_counts = np.unique(self.data_source.labels, return_counts=True)[1]
        self.minority_class, self.majority_class = np.argmin(self.class_counts), np.argmax(self.class_counts)
        self.minority_class_idx = np.argwhere(self.data_source.labels == self.minority_class).flatten()
        self.majority_class_idx = np.argwhere(self.data_source.labels == self.majority_class).flatten()

    def __iter__(self):
        r'''
        Instance method to return iterator object
        '''
        idx = np.append(self.minority_class_idx, np.random.choice(self.majority_class_idx,
                                                                  len(self.minority_class_idx), replace=False))
        np.random.shuffle(idx)
        return iter(idx)

    def __len__(self):
        r'''
        Number of sites processed by the iterator
        '''
        return 2 * len(self.minority_class_idx)


class ImbalanceOverSampler(torch.utils.data.Sampler):
    r"""
    A PyTorch Sampler class for maintaining balanced m6Anet training through oversampling the minority modification class in the dataset. This class is used
    for m6Anet training, producing oversampled sites corresponding to the number of majority classes in the dataset

    ...

    Attributes
    -----------
    data_source (NanopolishDS): An instance of NanopolishDS class
    class_counts (np.ndarray): A NumPy array containing the counts for all unique labels in the dataset
    minority_class (np.int64): Label for the minority class
    minority_class_idx (np.ndarray): Indices on data_source that corresponds to the minority class instances
    majority_class (np.int64): Label for the majority class
    majority_class_idx (np.ndarray): Indices on data_source that corresponds to the majority_class_idx class instances
    """
    def __init__(self, data_source):
        r'''
        Initialization function for the class

                Args:
                        data_source (NanopolishDS): An instance of NanopolishDS class

                Returns:
                        None
        '''
        self.data_source = data_source
        self.class_counts = np.unique(self.data_source.labels, return_counts=True)[1]
        self.minority_class, self.majority_class = np.argmin(self.class_counts), np.argmax(self.class_counts)
        self.minority_class_idx = np.argwhere(self.data_source.labels == self.minority_class).flatten()
        self.majority_class_idx = np.argwhere(self.data_source.labels == self.majority_class).flatten()

    def __iter__(self):
        r'''
        Instance method to return iterator object
        '''
        idx = np.append(self.majority_class_idx, np.random.choice(self.minority_class_idx,
                                                                  len(self.majority_class_idx), replace=True))
        np.random.shuffle(idx)
        return iter(idx)

    def __len__(self):
        r'''
        Number of sites processed by the iterator
        '''
        return 2 * len(self.majority_class_idx)


class ImbalanceKmerUnderSampler(torch.utils.data.Sampler):
    r"""
    A PyTorch Sampler class for maintaining balanced m6Anet training between different DRACH motif by undersampling each DRACH segment to the least expressed motif.
    This class is used for m6Anet training, producing balanced DRACH sites during each training epoch.

    ...

    Attributes
    -----------
    data_source (NanopolishDS): An instance of NanopolishDS class
    class_counts (np.ndarray): A NumPy array containing the counts for all unique labels in the dataset
    minority_class (np.int64): Label for the minority class
    minority_class_idx (Dict): Dictionary containing indices for each motif for the minority class
    majority_class (Dict): Label for the majority class
    majority_class_idx (Dict): Dictionary containing indices for each motif for the majority class
    length (int): Total number of sites being iterated
    """
    def __init__(self, data_source):
        r'''
        Initialization function for the class

                Args:
                        data_source (NanopolishDS): An instance of NanopolishDS class

                Returns:
                        None
        '''
        self.data_source = data_source
        self.data_kmers = self.data_source.data_info["kmer"].values
        self.all_motifs = np.unique(self.data_kmers)
        self.class_counts = np.unique(self.data_source.labels, return_counts=True)[1]
        self.minority_class, self.majority_class = np.argmin(self.class_counts), np.argmax(self.class_counts)

        self.minority_class_idx = {}
        self.majority_class_idx = {}

        for label, idx_dict in zip([self.minority_class, self.majority_class],
                                   [self.minority_class_idx, self.majority_class_idx]):
            for motif in self.all_motifs:
                label_mask = (self.data_source.labels == label)
                motif_mask = (self.data_kmers == motif)
                idx_dict[motif] = np.argwhere(label_mask & motif_mask).flatten()
        self.length = len(self.generate_indices())

    def generate_indices(self):
        r'''
        Instance method to generate indices for __iter__ call
        '''
        indices = []
        for motif, majority_idx in self.majority_class_idx.items():
            if motif in self.minority_class_idx:
                minority_idx = self.minority_class_idx[motif]
                minority_count = len(minority_idx)
                replace = len(majority_idx) < minority_count # Replace if there are more minority samples
                indices = np.append(indices, np.random.choice(majority_idx, minority_count, replace=replace))
                indices = np.append(indices, minority_idx)
            else:
                indices = np.append(indices, majority_idx)
        indices = indices.astype('int')
        np.random.shuffle(indices)
        return indices

    def __iter__(self):
        r'''
        Instance method to return iterator object
        '''
        indices = self.generate_indices()
        return iter(indices)

    def __len__(self):
        r'''
        Number of sites processed by the iterator
        '''
        return self.length


class ImbalanceKmerOverSampler(torch.utils.data.Sampler):
    r"""
    A PyTorch Sampler class for maintaining balanced m6Anet training between different DRACH motif by oversampling each DRACH segment to the most expressed motif.
    This class is used for m6Anet training, producing balanced DRACH sites during each training epoch.

    ...

    Attributes
    -----------
    data_source (NanopolishDS): An instance of NanopolishDS class
    class_counts (np.ndarray): A NumPy array containing the counts for all unique labels in the dataset
    minority_class (np.int64): Label for the minority class
    minority_class_idx (Dict): Dictionary containing indices for each motif for the minority class
    majority_class (Dict): Label for the majority class
    majority_class_idx (Dict): Dictionary containing indices for each motif for the majority class
    length (int): Total number of sites being iterated
    """
    def __init__(self, data_source):
        r'''
        Initialization function for the class

                Args:
                        data_source (NanopolishDS): An instance of NanopolishDS class

                Returns:
                        None
        '''
        self.data_source = data_source
        self.data_kmers = self.data_source.data_info["kmer"].values
        self.all_motifs = np.unique(self.data_kmers)
        self.class_counts = np.unique(self.data_source.labels, return_counts=True)[1]
        self.minority_class, self.majority_class = np.argmin(self.class_counts), np.argmax(self.class_counts)

        self.minority_class_idx = {}
        self.majority_class_idx = {}

        for label, idx_dict in zip([self.minority_class, self.majority_class],
                                   [self.minority_class_idx, self.majority_class_idx]):
            for motif in self.all_motifs:
                label_mask = (self.data_source.labels == label)
                motif_mask = (self.data_kmers == motif)
                idx_dict[motif] = np.argwhere(label_mask & motif_mask).flatten()

        self.length = len(self.generate_indices())

    def generate_indices(self):
        r'''
        Instance method to generate indices for __iter__ call
        '''
        indices = []
        for motif, minority_idx in self.minority_class_idx.items():
            if (motif in self.majority_class_idx) and (len(minority_idx) > 0):
                majority_idx = self.majority_class_idx[motif]
                majority_count = len(majority_idx)

                assert(majority_count >= len(minority_idx))

                n_samples = majority_count - len(minority_idx)
                replace = n_samples > len(minority_idx) # sample with replacement if more samples needed than the entire minority samples
                sampled_minority_idx = np.random.choice(minority_idx, n_samples, replace=replace)
                oversampled_minority_idx = np.append(minority_idx, sampled_minority_idx)
                indices = np.append(indices, oversampled_minority_idx)
                indices = np.append(indices, majority_idx)
            else:
                indices = np.append(indices, minority_idx)
        indices = indices.astype('int')
        np.random.shuffle(indices)
        return indices

    def __iter__(self):
        r'''
        Instance method to return iterator object
        '''
        indices = self.generate_indices()
        return iter(indices)

    def __len__(self):
        r'''
        Number of sites processed by the iterator
        '''
        return self.length
