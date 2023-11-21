import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from typing import List
from optimizer_implementation.utils.utils import get_training_lossses
from optimizer_implementation.optim.optimizer import Adam, Adamax, SGD, RMSProp, Adagrad


def plot_optimizer_losses(
    chosen_otimizers: List = [Adam, Adamax, SGD, RMSProp, Adagrad], num_epochs: int = 5
) -> None:
    dict_loss = {}

    num_cores = multiprocessing.cpu_count()

    processed_list = Parallel(n_jobs=num_cores)(
        delayed(get_training_lossses)(optimizer) for optimizer in chosen_otimizers
    )

    for optimizer, loss in zip(chosen_otimizers, processed_list):
        optimizer_name = optimizer.__name__
        dict_loss[optimizer_name] = loss

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
