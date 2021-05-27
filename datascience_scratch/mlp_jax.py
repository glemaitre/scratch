from __future__ import annotations
from typing import (
    List,
    Optional,
    Tuple,
)

import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jrand
from jax import (
    jit,
    value_and_grad,
    vmap,
)

from sklearn.base import BaseEstimator, ClassifierMixin  # type: ignore


def _relu_layer(
    coef: jnp.ndarray, intercept: jnp.ndarray, x: jnp.ndarray
) -> jnp.ndarray:
    return jnp.maximum(0, jnp.dot(coef, x) + intercept)


def _sigmoid_layer(
    coef: jnp.ndarray, intercept: jnp.ndarray, x: jnp.ndarray
) -> jnp.ndarray:
    return jsp.special.expit(jnp.dot(coef, x) + intercept)


def _forward_sample(
    sample: jnp.ndarray, coefs: List[jnp.array], intercepts: List[jnp.ndarray]
) -> float:
    activations = sample
    for coef, intercept in zip(coefs, intercepts):
        # activations = _relu_layer(coef, intercept, activations)
        activations = _sigmoid_layer(coef, intercept, activations)
    # logits = jnp.dot(coefs[-1], activations) + intercepts[-1]
    # return logits - jsp.special.logsumexp(logits)
    return activations

# def _loss(self, X, y_encoded):
#     y_pred = self.forward(X)
#     return - jnp.sum(y_pred * y_encoded)

# @jit
# def update(self, X, y_encoded):
#     """ Compute the gradient for a batch and update the parameters """
#     return value_and_grad(self._loss)(X, y_encoded)


class MLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        activation: str = "ReLU",
        max_epoch: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

    @staticmethod
    def _initialize_parameter(
        input_size: int,
        output_size: int,
        random_key: jnp.ndarray,
        scale: Optional[float] = 1e-2,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        coef_key, intercept_key = jrand.split(random_key, num=2)
        return (
            scale * jrand.normal(coef_key, (input_size, output_size)),
            scale * jrand.normal(intercept_key, (output_size,)),
        )

    def forward(self, X):
        vmap_forward = vmap(
            _forward_sample, in_axes=(0, None, None), out_axes=0
        )
        return vmap_forward(X, self.coefs_, self.intercepts_)

    @staticmethod
    def _target_encoder(y, n_classes):
        return jnp.array(y[:, None] == jnp.arange(n_classes), jnp.float32)

    def _loss(self, X, y_encoded):
        y_pred = self.forward(X)
        return - jnp.sum(y_pred * y_encoded)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray) -> MLPClassifier:
        self.classes_ = jnp.unique(y)
        self.n_classes_ = len(self.classes_)
        layers_size = (
            (X.shape[1],) + self.hidden_layer_sizes + (self.n_classes_,)
        )
        self._random_key = jrand.PRNGKey(self.seed)
        self._random_key_layers = jrand.split(
            self._random_key, len(layers_size)
        )
        y_encoded = self._target_encoder(y, self.n_classes_)

        self.coefs_, self.intercepts_ = zip(
            *[
                self._initialize_parameter(input_size, output_size, key)
                for input_size, output_size, key in zip(
                    layers_size[:-1], layers_size[1:], self._random_key_layers
                )
            ]
        )

        return self
