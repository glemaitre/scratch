import numpy as np

from scipy.special import expit

from .layer import Sigmoid


def test_sigmoid_layer():
    layer = Sigmoid()

    input = np.array([1, 2, 3], dtype=np.float32)
    np.testing.assert_allclose(layer.forward(input), expit(input))

    gradient = np.array([3, 2, 1], dtype=np.float32)
    np.testing.assert_allclose(
        layer.backward(gradient),
        expit(input) * (1 - expit(input)) * gradient,
    )
