from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit


class Layer(ABC):
    def __init__(self) -> None:
        self.params: Dict[str, NDArray[np.float32]] = {}
        self.grads: Dict[str, NDArray[np.float32]] = {}

    @abstractmethod
    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad: NDArray[np.float32]) -> NDArray[np.float32]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} layer".rstrip()


class Linear(Layer):
    def __init__(
        self, input_size: int, output_size: int, seed: Optional[int] = None
    ) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self._rng = np.random.default_rng(seed=seed)
        self.params["w"] = self._rng.random(
            size=(input_size, output_size), dtype=np.float32
        )
        self.params["b"] = self._rng.random(
            size=(output_size,), dtype=np.float32
        )

    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T

    def __repr__(self) -> str:
        return f"""{super().__repr__()}:
        Dimensions of W: {self.params['w'].shape}
        Dimension of b: {self.params['b'].shape}
        """.rstrip()


F = Callable[[NDArray[np.float32]], NDArray[np.float32]]


class Activation(Layer):
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.f_prime(self.inputs) * grad


def tanh(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.tanh(x)


def tanh_prime(x: NDArray[np.float32]) -> NDArray[np.float32]:
    y = tanh(x)
    return 1 - y ** 2


class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__(tanh, tanh_prime)


def sigmoid(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return expit(x)


def sigmoid_prime(x: NDArray[np.float32]) -> NDArray[np.float32]:
    y = sigmoid(x)
    return y * (1 - y)


class Sigmoid(Activation):
    def __init__(self) -> None:
        super().__init__(sigmoid, sigmoid_prime)
