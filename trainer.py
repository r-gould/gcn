import torch
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj


class Trainer:

    def train(self, gcn, data, optimizer, epochs):

        x = data.x
        y_train = data.y[data.train_mask]
        train_accs = []
        train_losses = []
        test_accs = []
        test_losses = []

        A = torch.squeeze(to_dense_adj(data.edge_index))
        gcn.set_adj(A)

        for epoch in range(1, epochs+1):

            print("Epoch:", epoch)

            log_probs = gcn.forward(x)[data.train_mask]
            loss = F.nll_loss(log_probs, y_train)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = self.accuracy(log_probs, y_train)
            train_accs.append(acc)

            test_acc, test_loss = self.test(gcn, data)
            test_losses.append(test_loss.item())
            test_accs.append(test_acc)


        return (train_accs, train_losses), (test_accs, test_losses)

    
    def test(self, gcn, data):
        
        x = data.x
        y_test = data.y[data.test_mask]

        with torch.no_grad():

            log_probs = gcn(x)[data.test_mask]
            loss = F.nll_loss(log_probs, y_test)
            acc = self.accuracy(log_probs, y_test)

        return acc, loss

    
    def accuracy(self, log_probs, labels):

        preds = torch.argmax(log_probs, dim=1)
        correct = torch.sum(preds == labels)
        return correct / len(log_probs)