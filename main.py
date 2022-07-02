import torch
import matplotlib.pyplot as plt

from gcn.gcn import GCN
from trainer import Trainer
from utils import plot_accs, visualize
from torch_geometric.datasets import Planetoid


def main(data_str, hidden_sizes, epochs, lr, weight_decay):

    dataset = Planetoid(root=f"data/{data_str}", name=data_str)
    data = dataset[0]

    num_features = dataset.num_features
    num_classes = dataset.num_classes

    gcn = GCN(hidden_sizes, num_features, num_classes)
    optimizer = torch.optim.Adam(gcn.parameters(), lr, weight_decay=weight_decay)
    trainer = Trainer()
    
    (train_accs, _), (test_accs, _) = trainer.train(gcn, data, optimizer, epochs)
    plot_accs(train_accs, test_accs)
    visualize(gcn, data, num_classes)


if __name__ == "__main__":

    data_str = "Cora"
    hidden_sizes = [16]
    epochs = 100
    lr = 1e-2
    weight_decay = 5e-4

    main(data_str, hidden_sizes, epochs, lr, weight_decay)