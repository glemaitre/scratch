import numpy as np

from .layer import Linear, Sigmoid

from .neural_network import NeuralNetwork


def test_neural_network_xor():
    nn = NeuralNetwork(
        layers=[
            Linear(input_size=2, output_size=2),
            Sigmoid(),
            Linear(input_size=2, output_size=1),
            Sigmoid(),
        ]
    )

    nn.layers[0].params["w"] = np.array([[20, 20], [20, 20]])
    nn.layers[0].params["b"] = np.array([-30, -10])
    nn.layers[2].params["w"] = np.array([[-60], [60]])
    nn.layers[2].params["b"] = np.array([-30])

    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    targets = np.array([0, 1, 1, 0]).reshape(-1, 1)
    predictions = nn.forward(inputs)
    assert targets.shape == predictions.shape
    np.testing.assert_allclose(predictions, targets, atol=1e-7)
