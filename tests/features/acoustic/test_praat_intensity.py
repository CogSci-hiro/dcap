# =============================================================================
# =============================================================================
#                 ############################################
#                 #      TESTS: PRAAT INTENSITY FEATURE      #
#                 ############################################
# =============================================================================
# =============================================================================

import numpy as np
import pytest

from dcap.features.acoustic.praat_intensity import PraatIntensityComputer, PraatIntensityConfig
from dcap.features.types import FeatureTimeBase


parselmouth = pytest.importorskip("parselmouth")  # skip entire file if not installed


def _am_sine(
    *,
    sfreq: float,
    carrier_hz: float,
    mod_hz: float,
    duration_s: float,
    mod_depth: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """AM sinusoid and its ground-truth amplitude envelope."""
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
            bb = b[lag: lag + aa.shape[0]]
        else:
            aa = a
            bb = b
        if aa.size < 10:
            continue
        r = float(np.corrcoef(_zscore(aa), _zscore(bb))[0, 1])
        best = max(best, r)
    return best


def test_praat_intensity_returns_exact_timebase_length() -> None:
    sfreq_audio = 48_000.0
    duration_s = 2.0
    x, _ = _am_sine(sfreq=sfreq_audio, carrier_hz=800.0, mod_hz=3.0, duration_s=duration_s)

    time = FeatureTimeBase(sfreq=200.0, n_times=int(round(duration_s * 200.0)), t0_s=0.0)
    cfg = PraatIntensityConfig(time_step_s=0.01, minimum_pitch_hz=75.0, output_scale="db", derivative="none")

    comp = PraatIntensityComputer()
    out = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)

    assert out.values.shape == (time.n_times,)
    assert out.time.sfreq == time.sfreq
    assert out.time.n_times == time.n_times
    assert out.kind == "acoustic"
    assert out.channel_names == ["intensity"]


def test_praat_intensity_tracks_amplitude_envelope_high_correlation_db() -> None:
    # Praat intensity (dB) should strongly track an AM envelope (up to smoothing and log).
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
    cfg = PraatIntensityConfig(
        # make the analysis reasonably “dense”
        time_step_s=0.01,
        minimum_pitch_hz=75.0,
        output_scale="db",
        derivative="none",
    )

    comp = PraatIntensityComputer()
    out = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)

    # Resample ground-truth envelope to feature grid via interpolation (test-only).
    t_audio = np.arange(env_true.shape[0], dtype=float) / sfreq_audio
    t_feat = np.arange(time.n_times, dtype=float) / time.sfreq
    env_true_rs = np.interp(t_feat, t_audio, env_true)

    # Ignore edges: Praat windowing + our interpolation can have transient ends.
    edge = int(round(0.3 * time.sfreq))
    est_core = out.values[edge:-edge]
    true_core = env_true_rs[edge:-edge]

    # Because intensity is log(power), compare shapes via correlation after z-scoring.
    eps = 1e-12
    true_db_like = 20.0 * np.log10(np.maximum(true_core, eps))
    max_lag = int(round(0.10 * time.sfreq))  # 100 ms
    r = _best_lagged_corr(est_core, true_db_like, max_lag=max_lag)
    assert r > 0.85


def test_output_scale_linear_power_matches_db_transform() -> None:
    sfreq_audio = 16_000.0
    duration_s = 2.0
    x, _ = _am_sine(sfreq=sfreq_audio, carrier_hz=600.0, mod_hz=3.0, duration_s=duration_s)

    time = FeatureTimeBase(sfreq=200.0, n_times=int(round(duration_s * 200.0)), t0_s=0.0)
    comp = PraatIntensityComputer()

    out_db = comp.compute(
        time=time,
        audio=x,
        audio_sfreq=sfreq_audio,
        config=PraatIntensityConfig(time_step_s=0.005, minimum_pitch_hz=75.0, output_scale="db"),
    )
    out_lin = comp.compute(
        time=time,
        audio=x,
        audio_sfreq=sfreq_audio,
        config=PraatIntensityConfig(time_step_s=0.005, minimum_pitch_hz=75.0, output_scale="linear_power"),
    )

    edge = int(round(0.3 * time.sfreq))
    db = out_db.values[edge:-edge]
    lin = out_lin.values[edge:-edge]

    # linear_power should equal 10**(db/10) up to numerical tolerance
    lin_from_db = np.power(10.0, db / 10.0)

    # Compare relative error (ignore very tiny values)
    denom = np.maximum(lin, 1e-12)
    rel_err = np.median(np.abs(lin_from_db - lin) / denom)
    assert rel_err < 1e-6


def test_derivative_modes_basic_sanity() -> None:
    sfreq_audio = 20_000.0
    duration_s = 2.0
    x, _ = _am_sine(sfreq=sfreq_audio, carrier_hz=700.0, mod_hz=2.0, duration_s=duration_s)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    comp = PraatIntensityComputer()

    base = comp.compute(
        time=time,
        audio=x,
        audio_sfreq=sfreq_audio,
        config=PraatIntensityConfig(time_step_s=0.01, derivative="none"),
    )
    diff = comp.compute(
        time=time,
        audio=x,
        audio_sfreq=sfreq_audio,
        config=PraatIntensityConfig(time_step_s=0.01, derivative="diff"),
    )
    absdiff = comp.compute(
        time=time,
        audio=x,
        audio_sfreq=sfreq_audio,
        config=PraatIntensityConfig(time_step_s=0.01, derivative="absdiff"),
    )

    assert base.values.shape == diff.values.shape == absdiff.values.shape == (time.n_times,)
    assert np.all(absdiff.values >= -1e-12)
    assert np.any(diff.values > 0.0)
    assert np.any(diff.values < 0.0)


def test_missing_audio_raises() -> None:
    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)
    comp = PraatIntensityComputer()
    with pytest.raises(ValueError):
        _ = comp.compute(time=time, config=PraatIntensityConfig(), audio=None, audio_sfreq=None)
