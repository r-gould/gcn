import torch
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def visualize(gcn, data, num_classes):

    x = data.x
    y = data.y
    encoding = gcn.encode(x).detach()
    tsne = TSNE(2)
    enc = tsne.fit_transform(encoding)

    plt.scatter(enc[:, 0], enc[:, 1], c=y, cmap="plasma")
    plt.colorbar()
    plt.show()


def plot_accs(train_accs, test_accs):

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.title("Train accuracy")
    plt.plot(train_accs)
    plt.show()

    plt.title("Test accuracy")
    plt.plot(test_accs)
    plt.show()