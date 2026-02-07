# =============================================================================
# =============================================================================
#                 ############################################
#                 #        TESTS: VARNET-STYLE ENVELOPE      #
#                 ############################################
# =============================================================================
# =============================================================================

from __future__ import annotations

import numpy as np
import pytest

from dcap.features.acoustic.varnet_env import VarnetEnvelopeComputer, VarnetEnvelopeConfig
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
    """AM sinusoid + its ground-truth amplitude envelope."""
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

def test_varnet_env_returns_exact_timebase_length() -> None:
    sfreq_audio = 48_000.0
    duration_s = 2.0
    audio = _sine(sfreq=sfreq_audio, freq_hz=200.0, duration_s=duration_s, amplitude=1.0)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    cfg = VarnetEnvelopeConfig(derivative="none")

    comp = VarnetEnvelopeComputer()
    out = comp.compute(time=time, audio=audio, audio_sfreq=sfreq_audio, config=cfg)

    assert out.values.shape == (time.n_times,)
    assert out.time.sfreq == time.sfreq
    assert out.time.n_times == time.n_times
    assert out.kind == "acoustic"
    assert out.channel_names == ["env"]


def test_varnet_env_downmix_mean_multichannel() -> None:
    sfreq_audio = 16_000.0
    duration_s = 1.0
    a1 = _sine(sfreq=sfreq_audio, freq_hz=220.0, duration_s=duration_s, amplitude=1.0)
    a2 = _sine(sfreq=sfreq_audio, freq_hz=220.0, duration_s=duration_s, amplitude=0.5)
    audio = np.stack([a1, a2], axis=0)

    time = FeatureTimeBase(sfreq=200.0, n_times=int(round(duration_s * 200.0)), t0_s=0.0)
    cfg = VarnetEnvelopeConfig(downmix="mean")

    comp = VarnetEnvelopeComputer()
    out = comp.compute(time=time, audio=audio, audio_sfreq=sfreq_audio, config=cfg)

    assert out.values.shape == (time.n_times,)


# =============================================================================
#                          BEHAVIORAL / SANITY TESTS
# =============================================================================

def test_varnet_env_tracks_am_envelope_with_lowpass() -> None:
    """
    Varnet-style envelope (filterbank Hilbert envelopes + lowpass + combine) should track
    an AM envelope strongly (up to smoothing and minor lag).
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

    cfg = VarnetEnvelopeConfig(
        n_bands=16,                 # keep test fast
        fmin_hz=80.0,
        fmax_hz=5000.0,
        envelope_lowpass_hz=8.0,    # "slow envelope"
        combine="mean",
        derivative="none",
    )

    comp = VarnetEnvelopeComputer()
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
    assert r > 0.85


def test_varnet_env_combine_modes_produce_different_scales() -> None:
    """
    mean/sum/rms should all run and produce same shape, but not identical outputs.
    """
    sfreq_audio = 16_000.0
    duration_s = 2.0
    x, _ = _am_sine(sfreq=sfreq_audio, carrier_hz=600.0, mod_hz=3.0, duration_s=duration_s, mod_depth=0.6)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    comp = VarnetEnvelopeComputer()

    base_cfg = dict(n_bands=12, fmin_hz=80.0, fmax_hz=5000.0, envelope_lowpass_hz=8.0)

    out_mean = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=VarnetEnvelopeConfig(**base_cfg, combine="mean"))
    out_sum = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=VarnetEnvelopeConfig(**base_cfg, combine="sum"))
    out_rms = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=VarnetEnvelopeConfig(**base_cfg, combine="rms"))

    assert out_mean.values.shape == out_sum.values.shape == out_rms.values.shape == (time.n_times,)

    # Not identical (scale/shape differs)
    assert not np.allclose(out_mean.values, out_sum.values)
    assert not np.allclose(out_mean.values, out_rms.values)


def test_varnet_env_no_lowpass_still_runs() -> None:
    sfreq_audio = 16_000.0
    duration_s = 1.5
    x = _sine(sfreq=sfreq_audio, freq_hz=400.0, duration_s=duration_s, amplitude=1.0)

    time = FeatureTimeBase(sfreq=200.0, n_times=int(round(duration_s * 200.0)), t0_s=0.0)
    cfg = VarnetEnvelopeConfig(n_bands=8, envelope_lowpass_hz=None, derivative="none")

    comp = VarnetEnvelopeComputer()
    out = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)

    assert np.all(np.isfinite(out.values))
    assert float(np.std(out.values)) > 1e-8


# =============================================================================
#                         DERIVATIVE / ABS-DERIVATIVE TESTS
# =============================================================================

def test_varnet_derivative_modes_basic_sanity() -> None:
    sfreq_audio = 16_000.0
    duration_s = 2.0
    x, _ = _am_sine(sfreq=sfreq_audio, carrier_hz=700.0, mod_hz=2.0, duration_s=duration_s, mod_depth=0.5)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    comp = VarnetEnvelopeComputer()

    base = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=VarnetEnvelopeConfig(n_bands=10, derivative="none"))
    diff = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=VarnetEnvelopeConfig(n_bands=10, derivative="diff"))
    absdiff = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=VarnetEnvelopeConfig(n_bands=10, derivative="absdiff"))

    assert base.values.shape == diff.values.shape == absdiff.values.shape == (time.n_times,)
    assert np.all(absdiff.values >= -1e-12)
    assert np.any(diff.values > 0.0)
    assert np.any(diff.values < 0.0)


# =============================================================================
#                               PARAMETER VALIDATION
# =============================================================================

def test_varnet_invalid_lowpass_cutoff_raises() -> None:
    sfreq_audio = 1_000.0
    duration_s = 1.0
    x = _sine(sfreq=sfreq_audio, freq_hz=50.0, duration_s=duration_s, amplitude=1.0)

    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)
    comp = VarnetEnvelopeComputer()

    # Lowpass cutoff must be < Nyquist(audio_sfreq) = 500 Hz
    with pytest.raises(ValueError):
        _ = comp.compute(
            time=time,
            audio=x,
            audio_sfreq=sfreq_audio,
            config=VarnetEnvelopeConfig(envelope_lowpass_hz=600.0),
        )


def test_varnet_invalid_frequency_range_raises() -> None:
    sfreq_audio = 16_000.0
    x = _sine(sfreq=sfreq_audio, freq_hz=200.0, duration_s=1.0)

    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)
    comp = VarnetEnvelopeComputer()

    with pytest.raises(ValueError):
        _ = comp.compute(
            time=time,
            audio=x,
            audio_sfreq=sfreq_audio,
            config=VarnetEnvelopeConfig(fmin_hz=5000.0, fmax_hz=80.0),
        )


def test_missing_audio_raises() -> None:
    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)
    comp = VarnetEnvelopeComputer()

    with pytest.raises(ValueError):
        _ = comp.compute(time=time, config=VarnetEnvelopeConfig(), audio=None, audio_sfreq=None)
