from __future__ import annotations
from typing import (
    List,
    Optional,
    Tuple,
)

import numpy as np
from numpy.typing import NDArray
from sklearn.base import (  # type: ignore
    BaseEstimator,
    ClassifierMixin,
)

from .layer import ACTIVATIONS, LAYERS, Layer
from .loss import LOSSES
from .neural_network import NeuralNetwork
from .optimizer import OPTIMIZERS


class MLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_layers_size: Tuple[int, ...] = (100,),
        activation: str = "sigmoid",
        solver: str = "sgd",
        loss: str = "mse",
        learning_rate: float = 0.001,
        max_iter: int = 100,
        tol: float = 1e-5,
        seed: Optional[int] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        self.hiddent_layers_size = hidden_layers_size
        self.activation = activation
        self.solver = solver
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.verbose = verbose

    def fit(
        self, X: NDArray[np.float32], y: NDArray[np.float32]
    ) -> MLPClassifier:

        Activation = ACTIVATIONS[self.activation]
        self.solver_ = OPTIMIZERS[self.solver](
            learning_rate=self.learning_rate
        )
        self.loss_ = LOSSES[self.loss]()

        layers_size = (X.shape[1],) + self.hiddent_layers_size + (1,)
        layers: List[Layer] = []
        for input_size, output_size in zip(layers_size, layers_size[1:]):
            layers.append(
                LAYERS["linear"](
                    input_size=input_size,
                    output_size=output_size,
                    seed=self.seed,
                )
            )
            layers.append(Activation())

        self.model_ = NeuralNetwork(layers=layers)

        previous_loss, diff_loss, self.n_iter_ = np.inf, np.inf, 0
        while (diff_loss > self.tol) & (self.n_iter_ < self.max_iter):
            y_pred = self.model_.forward(X)
            epoch_loss = self.loss_.loss(y, y_pred)
            gradient_loss = self.loss_.grad(y, y_pred)
            print(gradient_loss)
            self.model_.backward(gradient_loss)
            self.solver_.step(self.model_)

            diff_loss = np.abs(previous_loss - epoch_loss)
            previous_loss = epoch_loss
            self.n_iter_ += 1

            if self.verbose:
                print(
                    f"Epoch {self.n_iter_}: "
                    f"{self.loss_.__class__.__name__}={epoch_loss:.3f}"
                )

        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.squeeze(self.model_.forward(X))
