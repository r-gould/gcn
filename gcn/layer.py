import torch
import torch.nn as nn


class GCNLayer(nn.Module):

    def __init__(self, input_size, output_size):

        super().__init__()
        self.init_weights(input_size, output_size)


    def forward(self, X, A_hat):
        
        return A_hat @ X @ self.W


    def init_weights(self, input_size, output_size):
        
        data = torch.randn(input_size, output_size)
        self.W = nn.Parameter(data, requires_grad=True)