# =============================================================================
# =============================================================================
#                 ############################################
#                 #          TESTS: COCHLEOGRAM FEATURE      #
#                 ############################################
# =============================================================================
# =============================================================================

from __future__ import annotations

import numpy as np
import pytest

from dcap.features.acoustic.cochleogram import CochleogramComputer, CochleogramConfig
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
#                               INVARIANTS
# =============================================================================

@pytest.mark.parametrize("mode", ["efficient", "accurate"])
def test_cochleogram_returns_exact_timebase_length(mode: str) -> None:
    sfreq_audio = 16_000.0
    duration_s = 1.5
    audio = _sine(sfreq=sfreq_audio, freq_hz=200.0, duration_s=duration_s, amplitude=1.0)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    cfg = CochleogramConfig(
        mode=mode,
        # keep #channels small for test speed
        f_min_hz=100.0,
        f_max_hz=6000.0,
        octave_spacing=0.5,
        env_target_fs_hz=200.0,
        derivative="none",
    )

    comp = CochleogramComputer()
    out = comp.compute(time=time, audio=audio, audio_sfreq=sfreq_audio, config=cfg)

    # Contract: values are (C, T)
    assert out.values.ndim == 2
    assert out.values.shape[1] == time.n_times
    assert out.time.sfreq == time.sfreq
    assert out.time.n_times == time.n_times

    # Channel names match channels dimension
    assert len(out.channel_names) == out.values.shape[0]
    assert all(name.startswith("cf_") for name in out.channel_names)

    # Should be finite
    assert np.all(np.isfinite(out.values))


def test_cochleogram_downmix_mean_multichannel() -> None:
    sfreq_audio = 16_000.0
    duration_s = 1.0
    a1 = _sine(sfreq=sfreq_audio, freq_hz=220.0, duration_s=duration_s, amplitude=1.0)
    a2 = _sine(sfreq=sfreq_audio, freq_hz=220.0, duration_s=duration_s, amplitude=0.5)
    audio = np.stack([a1, a2], axis=0)  # (channels, time)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    cfg = CochleogramConfig(mode="efficient", octave_spacing=0.5, downmix="mean")

    comp = CochleogramComputer()
    out = comp.compute(time=time, audio=audio, audio_sfreq=sfreq_audio, config=cfg)

    assert out.values.shape[1] == time.n_times
    assert out.values.shape[0] == len(out.channel_names)


# =============================================================================
#                          MODE TOGGLE + NON-IDENTITY
# =============================================================================

def test_cochleogram_modes_both_run_and_are_not_identical() -> None:
    """
    Accurate vs efficient should both run; outputs should generally differ
    (they use different filter backends).
    """
    sfreq_audio = 16_000.0
    duration_s = 1.25
    x, _ = _am_sine(sfreq=sfreq_audio, carrier_hz=800.0, mod_hz=2.0, duration_s=duration_s, mod_depth=0.7)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    comp = CochleogramComputer()

    common = dict(
        f_min_hz=100.0,
        f_max_hz=6000.0,
        octave_spacing=0.5,
        env_target_fs_hz=200.0,
        synapse_lowpass_cutoff_hz=30.0,
        synapse_lowpass_order=2,
        derivative="none",
    )

    out_eff = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=CochleogramConfig(mode="efficient", **common))
    out_acc = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=CochleogramConfig(mode="accurate", **common))

    assert out_eff.values.shape == out_acc.values.shape
    # Not identical (very likely) — allow some tolerance because some configs might converge.
    assert not np.allclose(out_eff.values, out_acc.values, atol=1e-12, rtol=1e-8)


# =============================================================================
#                           BEHAVIORAL SANITY (BAND-AVG ENVELOPE)
# =============================================================================

def test_cochleogram_bandmean_tracks_am_envelope_reasonably() -> None:
    """
    If we average across cochleogram channels, the result should track the AM envelope
    reasonably well (after smoothing + resampling).
    """
    sfreq_audio = 20_000.0
    duration_s = 2.5
    x, env_true = _am_sine(
        sfreq=sfreq_audio,
        carrier_hz=1200.0,
        mod_hz=2.5,
        duration_s=duration_s,
        mod_depth=0.8,
    )

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    cfg = CochleogramConfig(
        mode="efficient",
        f_min_hz=100.0,
        f_max_hz=6000.0,
        octave_spacing=0.5,          # keep channels low for test speed
        synapse_lowpass_cutoff_hz=30.0,
        env_target_fs_hz=200.0,
        derivative="none",
    )

    comp = CochleogramComputer()
    out = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)

    # Combine channels -> 1D envelope proxy
    env_est = np.mean(out.values, axis=0)

    # Ground truth envelope on feature grid (test-only interpolation)
    t_audio = np.arange(env_true.shape[0], dtype=float) / sfreq_audio
    t_feat = np.arange(time.n_times, dtype=float) / time.sfreq
    env_true_rs = np.interp(t_feat, t_audio, env_true)

    edge = int(round(0.25 * time.sfreq))
    est_core = env_est[edge:-edge]
    true_core = env_true_rs[edge:-edge]

    r = _best_lagged_corr(est_core, true_core, max_lag=int(round(0.12 * time.sfreq)))
    assert r > 0.70


# =============================================================================
#                          DERIVATIVE / ABS-DERIVATIVE
# =============================================================================

@pytest.mark.parametrize("derivative", ["diff", "absdiff"])
def test_cochleogram_derivative_modes_sanity(derivative: str) -> None:
    sfreq_audio = 16_000.0
    duration_s = 2.0
    x, _ = _am_sine(sfreq=sfreq_audio, carrier_hz=700.0, mod_hz=2.0, duration_s=duration_s, mod_depth=0.6)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    cfg = CochleogramConfig(
        mode="efficient",
        f_min_hz=100.0,
        f_max_hz=6000.0,
        octave_spacing=0.5,
        env_target_fs_hz=200.0,
        derivative=derivative,
    )

    comp = CochleogramComputer()
    out = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)

    assert out.values.shape[1] == time.n_times
    assert np.all(np.isfinite(out.values))

    if derivative == "absdiff":
        assert np.all(out.values >= -1e-12)
    else:
        # diff should contain both signs in at least one channel
        assert np.any(out.values > 0.0)
        assert np.any(out.values < 0.0)


# =============================================================================
#                          PARAMETER VALIDATION / ERRORS
# =============================================================================

def test_invalid_freq_range_raises() -> None:
    sfreq_audio = 16_000.0
    x = _sine(sfreq=sfreq_audio, freq_hz=200.0, duration_s=1.0)

    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)
    comp = CochleogramComputer()

    with pytest.raises(ValueError):
        _ = comp.compute(
            time=time,
            audio=x,
            audio_sfreq=sfreq_audio,
            config=CochleogramConfig(f_min_hz=6000.0, f_max_hz=100.0),
        )


def test_invalid_octave_spacing_raises() -> None:
    sfreq_audio = 16_000.0
    x = _sine(sfreq=sfreq_audio, freq_hz=200.0, duration_s=1.0)

    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)
    comp = CochleogramComputer()

    with pytest.raises(ValueError):
        _ = comp.compute(
            time=time,
            audio=x,
            audio_sfreq=sfreq_audio,
            config=CochleogramConfig(octave_spacing=0.0),
        )


def test_missing_audio_raises() -> None:
    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)
    comp = CochleogramComputer()

    with pytest.raises(ValueError):
        _ = comp.compute(time=time, config=CochleogramConfig(), audio=None, audio_sfreq=None)
