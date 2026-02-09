import numpy as np
import pytest

from dcap.analysis.trf.lags import LagSpec, compute_lags


def test_compute_lags_step_is_one_sample():
    sfreq = 100.0
    lags_samp, lags_s = compute_lags(LagSpec(-0.1, 0.4, include_0=True), sfreq)
    assert np.all(np.diff(lags_samp) == 1)
    assert np.allclose(lags_s, lags_samp / sfreq)
    assert 0 in set(lags_samp.tolist())


def test_compute_lags_exclude_zero():
    lags_samp, _ = compute_lags(LagSpec(-0.05, 0.05, include_0=False), sfreq=200.0)
    assert 0 not in set(lags_samp.tolist())


def test_compute_lags_bad_inputs_raise():
    with pytest.raises(ValueError):
        compute_lags(LagSpec(0.1, -0.1), sfreq=100.0)
    with pytest.raises(ValueError):
        compute_lags(LagSpec(-0.1, 0.1), sfreq=0.0)
