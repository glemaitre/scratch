from __future__ import annotations
from typing import (
    Callable,
    Tuple,
)

import haiku as hk
import jax
import jax.numpy as jnp
from sklearn.base import (  # type: ignore
    BaseEstimator,
    ClassifierMixin,
)
import optax  # type: ignore

LOSSES = {
    "cross_entropy": optax.softmax_cross_entropy,
}

OPTIMIZERS = {
    "sgd": optax.sgd,
}


class NeuralNetworkClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        build_func: Callable,
        optimizer: str = "sgd",
        loss: str = "cross_entropy",
        learning_rate: float = 0.5,
        max_epoch: int = 100,
        seed: int = 0,
    ) -> None:
        self.build_func = build_func
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.seed = seed

    def _loss(self, params: hk.Params, X: jnp.ndarray, Y: jnp.ndarray):
        logits = self.forward_network_.apply(params, self._key, X)
        return self.loss_(logits, Y)

    def _update(
        self,
        params: hk.Params,
        opt_state: optax.OptState,
        X: jnp.array,
        y: jnp.array,
    ) -> Tuple[hk.Params, optax.OptState]:
        """Learning rule (stochastic gradient descent)."""
        grads = jax.grad(self._loss)(params, X, y)
        updates, opt_state = self.optimizer_.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> NeuralNetworkClassifier:
        self.forward_network_ = hk.transform(self.build_func)
        self.optimizer_ = OPTIMIZERS[self.optimizer](
            learning_rate=self.learning_rate
        )
        self.loss_ = LOSSES[self.loss]
        self._key = jax.random.PRNGKey(self.seed)

        self.classes_ = jnp.unique(y)
        Y = jax.nn.one_hot(y, num_classes=len(self.classes_))

        self.params_ = self.forward_network_.init(self._key, X)
        self.opt_state_ = self.optimizer_.init(self.params_)

        for epoch in range(self.max_epoch):
            self.params_, self.opt_state_ = self._update(
                self.params_, self.opt_state_, X, Y
            )
        self.n_iter_ = epoch + 1

        return self
