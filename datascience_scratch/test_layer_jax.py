from jax import value_and_grad
import jax.numpy as jnp

from .layer_jax import Linear, Layer


def test_linear():
    X = jnp.array([1, 2, 3, 4]).reshape((2, 2))
    linear = Linear(input_size=2, output_size=2)
    print(linear.forward(X))
    print(value_and_grad(linear.forward)(X))
