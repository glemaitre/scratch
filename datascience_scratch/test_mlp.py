import numpy as np
from sklearn.datasets import make_classification

from .mlp import MultiLayerPerceptronClassifier


def test_mlp_classifier():
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_classes=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=0,
    )
    X, y = X.astype(np.float32), y.astype(np.int8)

    classifier = MultiLayerPerceptronClassifier(hidden_layer_sizes=(10, 20))

    classifier.fit(X, y)

    assert len(classifier.coefs_) == 3
    assert len(classifier.intercepts_) == 3

    n_neurons_per_layer = (2, 10, 20, 1)
    for layer_idx, coef in enumerate(classifier.coefs_):
        assert isinstance(coef, np.ndarray)
        assert coef.dtype == np.float32
        assert coef.shape[0] == n_neurons_per_layer[layer_idx]
        assert coef.shape[1] == n_neurons_per_layer[layer_idx + 1]

    for layer_idx, intercept in enumerate(classifier.intercepts_):
        assert isinstance(intercept, np.ndarray)
        assert intercept.dtype == np.float32
        assert intercept.shape == (n_neurons_per_layer[layer_idx + 1],)


def test_mlp_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int8)
    classifier = MultiLayerPerceptronClassifier(hidden_layer_sizes=(2,)).fit(
        X, y
    )

    classifier.coefs_ = [
        np.array([[20, 20], [20, 20]]),
        np.array([[-60], [60]]),
    ]
    classifier.intercepts_ = [
        np.array([-30, -10]),
        np.array([-30]),
    ]

    y_pred = classifier.predict(X)
    assert y_pred.shape == y.shape
    np.testing.assert_allclose(y_pred, y, atol=1e-7)
