from jax import vmap
import jax.numpy as jnp

from .mlp_jax import MLPClassifier
from .mlp_jax import _forward_sample


def test_mlp_classifier_xor():
    X = jnp.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = jnp.array([0., 1., 1., 0.])

    mlp = MLPClassifier((2,))
    mlp.fit(X, y)

    # print([coef.shape for coef in mlp.coefs_])
    # print([intercept.shape for intercept in mlp.intercepts_])
    # print(mlp.forward(X))

    # print(mlp.coefs_)
    # print(mlp.intercepts_)

    mlp.coefs_ = (
        jnp.array([[20., 20.], [20., 20.]]),
        jnp.array([[-60., 60.], [60., -60.]]),
    )
    mlp.intercepts_ = (
        jnp.array([-30., -10.]),
        jnp.array([-30., 30.]),
    )

    # print(mlp.forward(X))
    y_encoded = mlp._target_encoder(y, 2)
    from jax import grad
    print(grad(mlp._loss)(X, y_encoded))


def test_xxx():
    X = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = jnp.array([0, 1, 1, 0])
    batch = vmap(_forward_sample, in_axes=(0, None, None), out_axes=0)
    mlp = MLPClassifier((2,)).fit(X, y)
    print(mlp.coefs_)
    print(mlp.intercepts_)
    print(batch(X, mlp.coefs_, mlp.intercepts_))
