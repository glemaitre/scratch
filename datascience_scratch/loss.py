from abc import (
    ABC,
    abstractmethod,
)

import numpy as np
from numpy.typing import NDArray


class Loss(ABC):
    @abstractmethod
    def loss(
        self, y_true: NDArray[np.float32], y_pred: NDArray[np.float32]
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def grad(
        self, t_true: NDArray[np.float32], y_pred: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        raise NotImplementedError


class MSE(Loss):
    def loss(
        self, y_true: NDArray[np.float32], y_pred: NDArray[np.float32]
    ) -> float:
        return np.average((y_pred - y_true) ** 2)

    def grad(
        self, y_true: NDArray[np.float32], y_pred: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        return 2 * (y_pred - y_true)


LOSSES = {
    "mse": MSE,
}
