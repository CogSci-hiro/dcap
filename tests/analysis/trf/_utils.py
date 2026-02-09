from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class SimData:
    sfreq: float
    X: np.ndarray
    Y: np.ndarray
    coef_true: np.ndarray
    intercept_true: np.ndarray
    lags_samp: np.ndarray


def make_simulated_trf_data(
    *,
    seed: int,
    sfreq: float,
    n_times: int,
    n_features: int,
    n_outputs: int,
    lags_samp: np.ndarray,
    noise_std: float,
) -> SimData:
    """Generate X,Y from a known kernel via predict_from_kernel."""
    from dcap.analysis.trf.predict_kernel import predict_from_kernel

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_times, n_features))

    n_lags = int(len(lags_samp))
    coef_true = rng.standard_normal((n_lags, n_features, n_outputs))
    for _ in range(2):
        coef_true = (coef_true + np.roll(coef_true, 1, axis=0) + np.roll(coef_true, -1, axis=0)) / 3.0

    intercept_true = rng.standard_normal((n_outputs,)) * 0.05

    Y_clean = predict_from_kernel(
        X,
        coef=coef_true,
        intercept=intercept_true,
        lags_samp=lags_samp,
        mode="valid",
    )
    Y = Y_clean + rng.standard_normal(Y_clean.shape) * float(noise_std)

    return SimData(
        sfreq=float(sfreq),
        X=X,
        Y=Y,
        coef_true=coef_true,
        intercept_true=intercept_true,
        lags_samp=np.asarray(lags_samp, dtype=int),
    )


def corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    if den == 0:
        return 0.0
    return float((a * b).sum() / den)
