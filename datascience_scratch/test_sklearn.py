import numpy as np

from .sklearn import MLPClassifier


def test_mlp_classifier_xor():
    max_iter = 50_000
    mlp = MLPClassifier(
        hidden_layers_size=(2,),
        learning_rate=0.5,
        max_iter=max_iter,
        tol=0,
        seed=0,
    )

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0]).reshape(-1, 1)
    mlp.fit(X, y)
    y_pred = mlp.predict(X)

    assert y_pred[0] < 0.01
    assert y_pred[3] < 0.01
    assert y_pred[1] > 1 - 0.01
    assert y_pred[2] > 1 - 0.01

    assert mlp.n_iter_ == max_iter
