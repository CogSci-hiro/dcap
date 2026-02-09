import numpy as np

from dcap.analysis.trf.predict_kernel import predict_from_kernel


def test_predict_lag0_valid_matches_matrix_product():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 3))
    coef = rng.standard_normal((1, 3, 2))
    intercept = rng.standard_normal((2,))
    y_hat = predict_from_kernel(X, coef=coef, intercept=intercept, lags_samp=np.array([0]), mode="valid")
    y_ref = X @ coef[0, :, :] + intercept[None, :]
    assert np.allclose(y_hat, y_ref)


def test_predict_valid_zero_padding_outside_keep():
    X = np.ones((100, 1))
    coef = np.ones((3, 1, 1))
    intercept = np.zeros((1,))
    lags = np.array([-1, 0, 1], dtype=int)
    y = predict_from_kernel(X, coef=coef, intercept=intercept, lags_samp=lags, mode="valid")
    assert y[0, 0] == 0.0
    assert y[-1, 0] == 0.0


def test_predict_same_returns_full_length():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 2))
    coef = rng.standard_normal((5, 2, 3))
    y = predict_from_kernel(X, coef=coef, intercept=np.zeros((3,)), lags_samp=np.arange(-2, 3), mode="same")
    assert y.shape == (100, 3)
