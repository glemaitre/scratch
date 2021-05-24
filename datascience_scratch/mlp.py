from __future__ import annotations
from typing import (
    List,
    Optional,
    Tuple,
)

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit as sigmoid  # type: ignore

from sklearn.utils.validation import check_is_fitted  # type: ignore


class MultiLayerPerceptronClassifier:
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        seed: Optional[int] = None,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.seed = seed

    def fit(
        self, X: NDArray[np.float32], y: NDArray[np.int8]
    ) -> MultiLayerPerceptronClassifier:

        self._rng = np.random.default_rng(seed=self.seed)

        layer_size = (X.shape[1],) + self.hidden_layer_sizes + (1,)
        self.coefs_ = [
            self._rng.random(size=(n_input, n_output), dtype=np.float32)
            for n_input, n_output in zip(layer_size, layer_size[1:])
        ]
        self.intercepts_ = [
            self._rng.random(size=(n_neurons,), dtype=np.float32)
            for n_neurons in layer_size[1:]
        ]

        return self

    @staticmethod
    def _decision_function(
        input: NDArray[np.float32],
        coef: NDArray[np.float32],
        intercept: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        input_with_bias = np.concatenate(
            [input, np.ones(shape=(input.shape[0], 1), dtype=input.dtype)],
            axis=1,
        )
        weights = np.concatenate([coef, intercept[np.newaxis, :]], axis=0)
        return sigmoid(input_with_bias @ weights)

    def decision_function(
        self, X: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]:
        check_is_fitted(self, ["coefs_", "intercepts_"])
        input = X
        outputs: List[NDArray[np.float32]] = []
        for coef, intercept in zip(self.coefs_, self.intercepts_):
            outputs.append(self._decision_function(input, coef, intercept))
            input = outputs[-1]
        return outputs

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        return self.decision_function(X)[-1].squeeze()
