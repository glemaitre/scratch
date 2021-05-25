import numpy as np
import pytest

from .loss import MSE


def test_mse():
    mse = MSE()
    y_true = y_pred = np.array([0, 1, 1, 0], dtype=np.float32)
    mse_loss = mse.loss(y_true, y_pred)
    assert mse_loss == pytest.approx(0)

    y_pred = np.array([1, 0, 0, 1], dtype=np.float32)
    mse_loss = mse.loss(y_true, y_pred)
    assert mse_loss == pytest.approx(1)

    mse_grad = mse.grad(y_true, y_pred)
    np.testing.assert_allclose(mse_grad, [2, -2, -2, 2])
    assert mse_grad.dtype == np.float32
