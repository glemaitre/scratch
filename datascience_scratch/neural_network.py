from typing import Iterator, Tuple, Sequence

import numpy as np
from numpy.typing import NDArray

from .layer import Layer


class NeuralNetwork:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: NDArray[np.float32]) -> NDArray[np.float32]:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(
        self,
    ) -> Iterator[Tuple[NDArray[np.float32], NDArray[np.float32]]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def __repr__(self) -> str:
        output = [repr(layer) for layer in self.layers]
        return "Neural Network:\n" + "\n".join(output)
