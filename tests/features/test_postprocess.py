# =============================================================================
# =============================================================================
#                 ############################################
#                 #       TESTS: POSTPROCESS DERIVATIVES     #
#                 ############################################
# =============================================================================
# =============================================================================

import numpy as np
import pytest

from dcap.features.postprocess import DerivativeSpec, apply_derivative, parse_derivative


def _ramp(*, n: int, slope: float = 1.0, intercept: float = 0.0) -> np.ndarray:
    t = np.arange(n, dtype=float)
    return intercept + slope * t


def _sine(*, sfreq: float, freq_hz: float, duration_s: float, amplitude: float = 1.0) -> np.ndarray:
    n = int(round(duration_s * sfreq))
    t = np.arange(n, dtype=float) / float(sfreq)
    return amplitude * np.sin(2.0 * np.pi * float(freq_hz) * t)


# =============================================================================
#                               BASIC CONTRACTS
# =============================================================================

@pytest.mark.parametrize("mode", ["none", "diff", "absdiff", "hr", "hr_absdiff"])
def test_apply_derivative_preserves_shape_1d(mode: str) -> None:
    x = _sine(sfreq=100.0, freq_hz=2.0, duration_s=2.0)
    y = apply_derivative(x, sfreq=100.0, spec=DerivativeSpec(mode=mode, hr_factor=4))
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize("mode", ["none", "diff", "absdiff", "hr", "hr_absdiff"])
def test_apply_derivative_preserves_shape_2d(mode: str) -> None:
    x1 = _sine(sfreq=100.0, freq_hz=2.0, duration_s=2.0)
    x2 = _sine(sfreq=100.0, freq_hz=3.0, duration_s=2.0)
    x = np.stack([x1, x2], axis=0)  # (F, T)
    y = apply_derivative(x, sfreq=100.0, spec=DerivativeSpec(mode=mode, hr_factor=4))
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


# =============================================================================
#                         NUMERICAL EXPECTATIONS
# =============================================================================

def test_diff_of_ramp_is_constant_scaled_by_sfreq() -> None:
    sfreq = 50.0
    slope_per_sample = 0.2  # x[t] = 0.2*t
    x = _ramp(n=200, slope=slope_per_sample)

    y = apply_derivative(x, sfreq=sfreq, spec=DerivativeSpec(mode="diff"))
    core = y[5:]  # ignore the first sample which is forced to 0

    expected = slope_per_sample * sfreq
    assert float(np.mean(core)) == pytest.approx(expected, abs=1e-10)
    assert float(np.std(core)) == pytest.approx(0.0, abs=1e-10)


def test_absdiff_is_nonnegative() -> None:
    sfreq = 100.0
    x = _sine(sfreq=sfreq, freq_hz=3.0, duration_s=2.0)
    y = apply_derivative(x, sfreq=sfreq, spec=DerivativeSpec(mode="absdiff"))
    assert np.all(y >= -1e-12)


def test_hr_factor_1_matches_diff() -> None:
    sfreq = 100.0
    x = _sine(sfreq=sfreq, freq_hz=4.0, duration_s=2.0)

    y_diff = apply_derivative(x, sfreq=sfreq, spec=DerivativeSpec(mode="diff"))
    y_hr1 = apply_derivative(x, sfreq=sfreq, spec=DerivativeSpec(mode="hr", hr_factor=1))

    assert np.allclose(y_diff, y_hr1, atol=1e-12, rtol=1e-10)


def test_hr_absdiff_is_nonnegative() -> None:
    sfreq = 100.0
    x = _sine(sfreq=sfreq, freq_hz=4.0, duration_s=2.0)
    y = apply_derivative(x, sfreq=sfreq, spec=DerivativeSpec(mode="hr_absdiff", hr_factor=4))
    assert np.all(y >= -1e-12)


# =============================================================================
#                            PARSER COMPATIBILITY
# =============================================================================

def test_parse_derivative_accepts_expected_strings() -> None:
    for s in ["none", "diff", "absdiff", "hr", "hr_absdiff"]:
        spec = parse_derivative(s, hr_factor=7)
        assert spec.mode == s
        assert spec.hr_factor == 7


def test_parse_derivative_invalid_raises() -> None:
    with pytest.raises(ValueError):
        _ = parse_derivative("derivative_pls", hr_factor=4)


# =============================================================================
#                               ERROR HANDLING
# =============================================================================

def test_invalid_sfreq_raises() -> None:
    x = np.zeros(10, dtype=float)
    with pytest.raises(ValueError):
        _ = apply_derivative(x, sfreq=0.0, spec=DerivativeSpec(mode="diff"))


def test_invalid_hr_factor_raises() -> None:
    x = np.zeros(10, dtype=float)
    with pytest.raises(ValueError):
        _ = apply_derivative(x, sfreq=100.0, spec=DerivativeSpec(mode="hr", hr_factor=0))


def test_invalid_shape_raises() -> None:
    x = np.zeros((2, 3, 4), dtype=float)
    with pytest.raises(ValueError):
        _ = apply_derivative(x, sfreq=100.0, spec=DerivativeSpec(mode="diff"))
