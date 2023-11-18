import matplotlib.pyplot as plt
from typing import List
from optimizer_implementation.utils.utils import get_training_lossses
from optimizer_implementation.optim.optimizer import Adam, Adamax, SGD


def plot_optimizer_losses(
    chosen_otimizers: List = [Adam, Adamax, SGD], num_epochs: int = 5
) -> None:
    dict_loss = {}

    for optimizer in chosen_otimizers:
        optimizer_name = optimizer.__name__
        losses = get_training_lossses(optimizer, num_epochs=num_epochs)

        dict_loss[optimizer_name] = losses

    plt.figure(figsize=(10, 6))

    for optimizer in dict_loss:
        dict_loss[optimizer] = dict_loss[optimizer][::50]
        x = [i for i in range(0, len(dict_loss[optimizer]))]
        plt.plot(x, dict_loss[optimizer], label=optimizer)

    plt.legend()
    plt.title("Optimizer loss benchmarking")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()
