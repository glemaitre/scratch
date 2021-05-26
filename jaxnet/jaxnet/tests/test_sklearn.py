import haiku as hk
import jax
import jax.numpy as jnp

from jaxnet.sklearn import NeuralNetworkClassifier


def test_neural_network_classifier():
    X = jnp.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ], dtype=jnp.float32)

    y = jnp.array([1, 0, 0, 1], dtype=jnp.float32)

    def build_func(X):
        sequential_model = hk.Sequential([
            hk.Linear(output_size=2), jax.nn.sigmoid,
            hk.Linear(output_size=2), jax.nn.sigmoid,
        ])
        return sequential_model(X)

    classifier = NeuralNetworkClassifier(
        build_func=build_func,
    )
    classifier.fit(X, y)
    print(classifier.params_)
    print(classifier.opt_state_)
