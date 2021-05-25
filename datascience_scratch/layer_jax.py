from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Dict,
)

import jax.numpy as jnp
import jax.random as jrand


class Layer(ABC):
    def __init__(self) -> None:
        self.params: Dict[str, jnp.ndarray] = {}
        self.grads: Dict[str, jnp.ndarray] = {}

    @abstractmethod
    def forward(self, inputs: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    # @abstractmethod
    # def backward(self, grad: jnp.ndarray) -> jnp.ndarray:
    #     raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} layer".rstrip()


class Linear(Layer):
    def __init__(
        self, input_size: int, output_size: int, seed: int = 0
    ) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self._rng = jrand.PRNGKey(seed)
        self.params["w"] = jrand.normal(
            self._rng, shape=(input_size, output_size)
        )
        self.params["b"] = jrand.normal(self._rng, shape=(output_size,))

    def forward(self, inputs: jnp.ndarray) -> jnp.ndarray:
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    # def backward(self, grad: jnp.ndarray) -> jnp.ndarray:
    #     """
    #     if y = f(x) and x = a @ b + c
    #     then dy/da = f'(x) @ b.T
    #     and dy/db = a.T @ f'(x)
    #     and dy/dc = f'(x)
    #     """
    #     self.grads["b"] = jnp.sum(grad, axis=0)
    #     self.grads["w"] = self.inputs.T @ grad
    #     return grad @ self.params["w"].T

    def __repr__(self) -> str:
        return f"""{super().__repr__()}:
        Dimensions of W: {self.params['w'].shape}
        Dimension of b: {self.params['b'].shape}
        """.rstrip()
