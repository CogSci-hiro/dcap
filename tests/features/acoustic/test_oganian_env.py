# =============================================================================
# =============================================================================
#                 ############################################
#                 #       TESTS: OGANIAN ENVELOPE FEATURE    #
#                 ############################################
# =============================================================================
# =============================================================================

from __future__ import annotations

import numpy as np
import pytest

from dcap.features.acoustic.oganian_env import OganianEnvelopeComputer, OganianEnvelopeConfig
from dcap.features.types import FeatureTimeBase


def _sine(*, sfreq: float, freq_hz: float, duration_s: float, amplitude: float = 1.0) -> np.ndarray:
    n = int(round(duration_s * sfreq))
    t = np.arange(n, dtype=float) / float(sfreq)
    return amplitude * np.sin(2.0 * np.pi * float(freq_hz) * t)


def _am_sine(
    *,
    sfreq: float,
    carrier_hz: float,
    mod_hz: float,
    duration_s: float,
    mod_depth: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    AM sinusoid and its ground-truth amplitude envelope.

    x(t) = a(t) * sin(2π f_c t), where a(t) = 1 + m sin(2π f_m t)
    """
    n = int(round(duration_s * sfreq))
    t = np.arange(n, dtype=float) / float(sfreq)
    env = 1.0 + float(mod_depth) * np.sin(2.0 * np.pi * float(mod_hz) * t)
    x = env * np.sin(2.0 * np.pi * float(carrier_hz) * t)
    return x, env


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd == 0.0:
        return x * 0.0
    return (x - mu) / sd


def _best_lagged_corr(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """
    Best Pearson correlation over integer lags in [-max_lag, +max_lag].

    This is useful because the Oganian "loudness" path includes smoothing
    that can introduce small phase shifts relative to a simple ground truth.
    """
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[: aa.shape[0]]
        elif lag > 0:
            aa = a[:-lag]
            bb = b[lag : lag + aa.shape[0]]
        else:
            aa = a
            bb = b
        if aa.size < 10:
            continue
        r = float(np.corrcoef(_zscore(aa), _zscore(bb))[0, 1])
        best = max(best, r)
    return best


# =============================================================================
#                               INVARIANT TESTS
# =============================================================================

@pytest.mark.parametrize("envtype", ["broadband", "loudness"])
def test_oganian_env_returns_exact_timebase_length(envtype: str) -> None:
    sfreq_audio = 16_000.0
    duration_s = 2.0
    x = _sine(sfreq=sfreq_audio, freq_hz=200.0, duration_s=duration_s, amplitude=1.0)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    cfg = OganianEnvelopeConfig(envtype=envtype, derivative="none")

    comp = OganianEnvelopeComputer()
    out = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)

    assert out.values.shape == (time.n_times,)
    assert out.time.sfreq == time.sfreq
    assert out.time.n_times == time.n_times
    assert out.kind == "acoustic"
    assert out.channel_names == ["env"]


def test_oganian_env_downmix_mean_multichannel() -> None:
    sfreq_audio = 10_000.0
    duration_s = 1.0
    a1 = _sine(sfreq=sfreq_audio, freq_hz=200.0, duration_s=duration_s, amplitude=1.0)
    a2 = _sine(sfreq=sfreq_audio, freq_hz=200.0, duration_s=duration_s, amplitude=0.5)
    audio = np.stack([a1, a2], axis=0)  # (channels, time)

    time = FeatureTimeBase(sfreq=200.0, n_times=int(round(duration_s * 200.0)), t0_s=0.0)
    cfg = OganianEnvelopeConfig(envtype="broadband", downmix="mean")

    comp = OganianEnvelopeComputer()
    out = comp.compute(time=time, audio=audio, audio_sfreq=sfreq_audio, config=cfg)

    assert out.values.shape == (time.n_times,)


# =============================================================================
#                          BROADBAND BEHAVIORAL TESTS
# =============================================================================

def test_oganian_broadband_tracks_amplitude_envelope_high_correlation() -> None:
    """
    Broadband envelope = LPF(|x|) should strongly track the AM envelope.

    We compare on the TRF grid with z-scoring and allow a small lag search.
    """
    sfreq_audio = 20_000.0
    duration_s = 3.0
    x, env_true = _am_sine(
        sfreq=sfreq_audio,
        carrier_hz=1000.0,
        mod_hz=2.5,
        duration_s=duration_s,
        mod_depth=0.8,
    )

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    cfg = OganianEnvelopeConfig(envtype="broadband", broadband_lowpass_hz=10.0, derivative="none")

    comp = OganianEnvelopeComputer()
    out = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)

    # Ground truth env on feature grid (test-only interpolation)
    t_audio = np.arange(env_true.shape[0], dtype=float) / sfreq_audio
    t_feat = np.arange(time.n_times, dtype=float) / time.sfreq
    env_true_rs = np.interp(t_feat, t_audio, env_true)

    edge = int(round(0.3 * time.sfreq))
    est_core = out.values[edge:-edge]
    true_core = env_true_rs[edge:-edge]

    max_lag = int(round(0.10 * time.sfreq))  # 100 ms
    r = _best_lagged_corr(est_core, true_core, max_lag=max_lag)
    assert r > 0.90


def test_oganian_broadband_invalid_lowpass_raises() -> None:
    sfreq_audio = 1_000.0
    x = _sine(sfreq=sfreq_audio, freq_hz=50.0, duration_s=1.0, amplitude=1.0)

    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)
    comp = OganianEnvelopeComputer()

    # lowpass must be < Nyquist(audio_sfreq) = 500 Hz
    with pytest.raises(ValueError):
        _ = comp.compute(
            time=time,
            audio=x,
            audio_sfreq=sfreq_audio,
            config=OganianEnvelopeConfig(envtype="broadband", broadband_lowpass_hz=600.0),
        )


# =============================================================================
#                          LOUDNESS BEHAVIORAL TESTS
# =============================================================================

def test_oganian_loudness_is_smooth_and_nonpathological() -> None:
    """
    Loudness proxy should yield a finite signal with non-trivial variance.

    We do NOT assert it matches the AM envelope perfectly because the method includes:
    - Bark-weighted bands
    - log/sqrt compression
    - signed band combination
    - heavy smoothing

    But it should still track slow modulations reasonably.
    """
    sfreq_audio = 20_000.0
    duration_s = 3.0
    x, env_true = _am_sine(
        sfreq=sfreq_audio,
        carrier_hz=800.0,
        mod_hz=2.0,
        duration_s=duration_s,
        mod_depth=0.7,
    )

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    cfg = OganianEnvelopeConfig(envtype="loudness", derivative="none")

    comp = OganianEnvelopeComputer()
    out = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)

    y = out.values
    assert np.all(np.isfinite(y))
    assert float(np.std(y)) > 1e-6

    # Loose tracking check: correlate with log-envelope (since loudness uses log compression)
    t_audio = np.arange(env_true.shape[0], dtype=float) / sfreq_audio
    t_feat = np.arange(time.n_times, dtype=float) / time.sfreq
    env_true_rs = np.interp(t_feat, t_audio, env_true)

    edge = int(round(0.4 * time.sfreq))
    est_core = y[edge:-edge]
    true_core = env_true_rs[edge:-edge]

    eps = 1e-12
    true_db_like = 20.0 * np.log10(np.maximum(true_core, eps))

    max_lag = int(round(0.20 * time.sfreq))  # allow more lag due to heavy smoothing
    r = _best_lagged_corr(est_core, true_db_like, max_lag=max_lag)

    # This threshold is intentionally modest: it's a sanity check, not a definition.
    assert r > 0.50


# =============================================================================
#                         DERIVATIVE / ABS-DERIVATIVE TESTS
# =============================================================================

@pytest.mark.parametrize("envtype", ["broadband", "loudness"])
def test_oganian_derivative_modes_basic_sanity(envtype: str) -> None:
    sfreq_audio = 16_000.0
    duration_s = 2.0
    x, _ = _am_sine(sfreq=sfreq_audio, carrier_hz=700.0, mod_hz=2.0, duration_s=duration_s, mod_depth=0.5)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    comp = OganianEnvelopeComputer()

    base = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=OganianEnvelopeConfig(envtype=envtype, derivative="none"))
    diff = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=OganianEnvelopeConfig(envtype=envtype, derivative="diff"))
    absdiff = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=OganianEnvelopeConfig(envtype=envtype, derivative="absdiff"))

    assert base.values.shape == diff.values.shape == absdiff.values.shape == (time.n_times,)
    assert np.all(absdiff.values >= -1e-12)
    assert np.any(diff.values > 0.0)
    assert np.any(diff.values < 0.0)


# =============================================================================
#                               ERROR HANDLING
# =============================================================================

def test_missing_audio_raises() -> None:
    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)
    comp = OganianEnvelopeComputer()
    with pytest.raises(ValueError):
        _ = comp.compute(time=time, config=OganianEnvelopeConfig(), audio=None, audio_sfreq=None)
