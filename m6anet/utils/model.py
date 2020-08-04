import torch


class MultiInstanceNNEmbedding(torch.nn.Module):
    def __init__(self, dim_cov, p=1, embedding_dim=2):
        super(MultiInstanceNNEmbedding, self).__init__()
        self.p = p
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(18, self.embedding_dim)
        self.linear1 = torch.nn.Linear(dim_cov + self.embedding_dim, 150, bias=True)
        self.linear2 = torch.nn.Linear(150, p, bias=True)
        self.linear3 = torch.nn.Linear(5 * p, 150, bias=True)
        self.linear4 = torch.nn.Linear(150, 1, bias=True)

    def forward(self, x, kmer, indices):
        """ compute probability at site level"""
        kmer_embedding = self.embedding(kmer)
        x = torch.cat([x, kmer_embedding], axis=1)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.aggregate(x, indices)
        x = x.view(-1, 5 * self.p)
        x = torch.relu(self.linear3(x))
        out = torch.sigmoid(self.linear4(x))
        return out

    def aggregate(self, x, indices):
        grouped_tensors = [x[idx] for idx in indices]
        mean = torch.stack([torch.mean(tensor, axis=0) for tensor in grouped_tensors])
        std = torch.stack([torch.var(tensor, axis=0) for tensor in grouped_tensors])
        maximum = torch.stack([torch.max(tensor, axis=0).values for tensor in grouped_tensors])
        minimum = torch.stack([torch.min(tensor, axis=0).values for tensor in grouped_tensors])
        median = torch.stack([torch.median(tensor, axis=0).values for tensor in grouped_tensors])
        aggregated_tensors = torch.stack([mean, std, minimum, median, maximum], axis=1)
        return aggregated_tensors
