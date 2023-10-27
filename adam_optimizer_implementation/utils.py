import torchvision
import torch
import numpy as np
import warnings

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
        -targets: torch.tensor: The targets
        of the loaded data
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
