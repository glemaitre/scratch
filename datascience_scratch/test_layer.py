import numpy as np

from scipy.special import expit

from .layer import (
    Linear,
    Sigmoid,
)


def test_sigmoid_layer():
    layer = Sigmoid()

    inputs = np.array([1, 2, 3], dtype=np.float32)
    outputs = layer.forward(inputs)
    assert outputs.dtype == np.float32
    np.testing.assert_allclose(outputs, expit(inputs))

    grad = np.array([3, 2, 1], dtype=np.float32)
    outputs = layer.backward(grad)
    assert outputs.dtype == np.float32
    np.testing.assert_allclose(
        outputs, expit(inputs) * (1 - expit(inputs)) * grad,
    )


def test_linear_layer():
    layer = Linear(input_size=2, output_size=1)
    assert layer.params["w"].dtype == np.float32
    assert layer.params["b"].dtype == np.float32

    layer.params["w"] = np.array([2, 1], dtype=np.float32).reshape(-1, 1)
    layer.params["b"] = np.array([0], dtype=np.float32)

    inputs = np.array([1, 1], dtype=np.float32).reshape(1, -1)
    outputs = layer.forward(inputs)
    assert outputs == 3
    assert outputs.dtype == np.float32

    layer.params["b"] = np.array([10], dtype=np.float32)
    outputs = layer.forward(inputs)
    assert outputs == 13
    assert outputs.dtype == np.float32
