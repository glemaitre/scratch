from abc import (
    ABC,
    abstractmethod,
)
from .neural_network import NeuralNetwork


class Optimizer(ABC):
    @abstractmethod
    def step(self, neural_network: NeuralNetwork) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def step(self, neural_network: NeuralNetwork) -> None:
        for param, grad in neural_network.params_and_grads():
            param -= self.learning_rate * grad


OPTIMIZERS = {
    "sgd": SGD,
}