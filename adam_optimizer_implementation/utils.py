import torchvision
import torch
import numpy as np
import warnings
import torch.nn as nn

warnings.filterwarnings("ignore")


def get_label(x: int) -> torch.tensor:
    """
    The goal of this function
    is to generate fake labels
    for the try of the neural
    network on the MNIST dataset

    Arguments:
        -x: int: The number that's
        represented on a given image
    Returns:
        -labels: torch.tensor: The tensor
        containing binary position of the
        given number
    """
    labels = torch.zeros(size=(10,))
    labels[x] = 1
    return np.array(labels)


def load_mnist_data(batch_size: int = 64) -> torch._utils:
    """
    The goal of this function is to
    load the MNIST dataset to carry
    on experiments with Adam optimizer

    Arguments:
        -batch_size: int: The batch size
        associated with MNIST loading
    Returns:
        -data: torch.tensor: The MNIST
        data
    """

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data/",
            train=True,
            download=False,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    data_iterator = iter(train_loader)

    return data_iterator


class Feed_Forward_Neural_Network(nn.Module):
    """
    The goal of this class is creating
    a basic neural network that will
    be used with the coded optimizer to
    check its working well

    Arguments:
        -None
    Returns:
        -None
    """

    def __init__(self):
        super(Feed_Forward_Neural_Network, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        The goal of this function is
        to apply a forward pass operation
        to a given input data

        Arguments:
            -x: torch.tensor: The input
            data
        Returns:
            -output: torch.tensor: The
            transformed input data
        """

        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)

        return output
