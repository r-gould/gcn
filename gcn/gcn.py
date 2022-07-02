import torch
import torch.nn as nn

from .layer import GCNLayer


class GCN(nn.Module):

    def __init__(self, hidden_sizes, num_features, num_classes, dropout=True):

        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.network = self.build_network(hidden_sizes, dropout)


    def forward(self, X, A=None):

        if A:
            self.set_adj(A)

        out = X
        for _, layer in self.network.named_children():
            if isinstance(layer, GCNLayer):
                out = layer(out, self.A_hat)
                continue
            out = layer(out)

        return out


    def encode(self, x, A=None):

        if A:
            self.set_adj(A)

        encoder = self.network[0]
        encoding = encoder(x, self.A_hat)
        return encoding
    

    def build_network(self, hidden_sizes, dropout):

        layers = []
        prev_size = self.num_features
        
        for i in range(len(hidden_sizes)):

            size = hidden_sizes[i]
            layers.append(GCNLayer(prev_size, size))
            layers.append(nn.ReLU())

            if dropout:
                layers.append(nn.Dropout(0.5))

            prev_size = size

        layers.append(GCNLayer(prev_size, self.num_classes))
        layers.append(nn.LogSoftmax(dim=1))
        return nn.Sequential(*layers)


    def set_adj(self, A):

        A_tilde = A + torch.eye(A.shape[0])
        D_diag = torch.sum(A_tilde, dim=1)
        D = torch.diag(D_diag)
        D_inv_root = torch.inverse(D**0.5)
        self.A_hat = D_inv_root @ A_tilde @ D_inv_root