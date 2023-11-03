import copy
import torch
import torch.nn as nn
from typing import Dict, Tuple


class Optimizer:
    pass


class SGD:
    pass


class RMSProp:
    pass


class Adam:
    """
    The goal of this class is the implementation
    of the Adam optimizer algorithm

    Parameters:
        -params: Dict: The initial parameters
        to be optimized
        -lr: float: The learning_rate
        to be applied
        -betas: Tuple: The couple of betas to be
        applied respectively for first and second-order
        gradient moment
        -epsilon: float: The minimal denominator to be
        applied during optimization

    Returns:
        -None
    """

    def __init__(
        self,
        params: Dict,
        model: nn.Module,
        betas: Tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-08,
        lr: float = 1e-03,
        **kwargs
    ) -> None:
        self.params = params
        self.updated_params = []
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.m = [torch.zeros(size=p.size()) for p in self.params]
        self.v = copy.deepcopy(self.m)
        self.t = 0
        self.lr = lr
        self.epsilon = epsilon
        self.model = model

        assert 0 <= self.beta1 <= 1, "beta1 must be between 0 and 1"
        assert 0 <= self.beta2 <= 1, "beta2 must be between 0 and 1"
        assert epsilon > 0, "epsilon parameter must be strictly positive"

    def zero_grad(self) -> None:
        """
        The goal of this function is to
        reinitialize the gradients of
        the different parameters

        Arguments:
            -None
        Returns:
            -None
        """

        for param in self.updated_params:
            if param.grad is not None:
                param.grad.data.zero_()

    def step(self) -> None:
        """
        The goal of this function is to apply
        an optimisation step to the parameters
        according to ADAM rules

        Arguments:
            -None
        Returns:
            -None
        """

        with torch.no_grad():
            for i, model_param in enumerate(self.model.parameters()):
                self.t += 1
                self.g = model_param.grad
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * self.g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * self.g**2
                debiased_m = self.m[i] / (1 - self.beta1**self.t)
                debiased_v = self.v[i] / (1 - self.beta2**self.t)
                model_param.data -= (
                    self.lr * debiased_m / (torch.sqrt(debiased_v) + self.epsilon)
                )
