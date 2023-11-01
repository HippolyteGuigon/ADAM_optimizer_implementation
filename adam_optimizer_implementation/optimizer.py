from typing import Dict, Tuple


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
        betas: Tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-08,
        lr: float = 1e-03,
        **kwargs
    ) -> None:
        self.params = params
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.lr = lr

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

        for param in self.params:
            if param.grad is not None:
                param.grad = None

    def update_gradient(self) -> None:
        for param in self.params:
            param -= self.lr * param.grad
            if param.grad is not None:
                param.grad = None
            yield param

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

        self.params = self.update_gradient()
