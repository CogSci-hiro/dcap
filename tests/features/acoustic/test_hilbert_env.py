# =============================================================================
# =============================================================================
#                 ############################################
#                 #     TESTS: HILBERT SPEECH ENVELOPE       #
#                 ############################################
# =============================================================================
# =============================================================================

import numpy as np
import pytest

from dcap.features.acoustic.hilbert_env import HilbertEnvelopeComputer, HilbertEnvelopeConfig
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
    """Return (signal, ground_truth_envelope) for AM sinusoid.

    Envelope is 1 + mod_depth*sin(2π f_mod t), always positive if mod_depth < 1.
    """
    n = int(round(duration_s * sfreq))
    t = np.arange(n, dtype=float) / float(sfreq)
    env = 1.0 + float(mod_depth) * np.sin(2.0 * np.pi * float(mod_hz) * t)
    x = env * np.sin(2.0 * np.pi * float(carrier_hz) * t)
    return x, env


def test_hilbert_env_returns_exact_timebase_length() -> None:
    sfreq_audio = 48_000.0
    duration_s = 2.0
    audio = _sine(sfreq=sfreq_audio, freq_hz=200.0, duration_s=duration_s)

    time = FeatureTimeBase(sfreq=100.0, n_times=200, t0_s=0.0)  # exactly 2 seconds at 100 Hz
    cfg = HilbertEnvelopeConfig(lowpass_hz=None, derivative="none")

    comp = HilbertEnvelopeComputer()
    out = comp.compute(time=time, audio=audio, audio_sfreq=sfreq_audio, config=cfg)

    assert out.values.shape == (time.n_times,)
    assert out.time.sfreq == time.sfreq
    assert out.time.n_times == time.n_times
    assert out.channel_names == ["env"]
    assert out.kind == "acoustic"


def test_hilbert_env_constant_amplitude_sine_is_nearly_constant() -> None:
    # For a pure sine wave, |hilbert(sin)| should be ~1 everywhere (edge effects aside).
    sfreq_audio = 10_000.0
    audio = _sine(sfreq=sfreq_audio, freq_hz=123.0, duration_s=2.0, amplitude=1.0)

    time = FeatureTimeBase(sfreq=200.0, n_times=400, t0_s=0.0)
    cfg = HilbertEnvelopeConfig(lowpass_hz=None, derivative="none")

    comp = HilbertEnvelopeComputer()
    out = comp.compute(time=time, audio=audio, audio_sfreq=sfreq_audio, config=cfg)

    env = out.values

    # Ignore ~100 ms at each edge to avoid Hilbert boundary artifacts + resample ringing.
    edge = int(round(0.1 * time.sfreq))
    core = env[edge:-edge]

    # Should be approximately flat around 1.0.
    assert np.mean(core) == pytest.approx(1.0, abs=0.05)
    assert np.std(core) < 0.10


def test_hilbert_env_tracks_am_envelope_with_lowpass() -> None:
    # AM signal: envelope is known; Hilbert magnitude should recover it.
    sfreq_audio = 20_000.0
    duration_s = 2.0
    x, env_true = _am_sine(
        sfreq=sfreq_audio,
        carrier_hz=800.0,
        mod_hz=3.0,
        duration_s=duration_s,
        mod_depth=0.8,
    )

    # Target grid for TRF-style regressors
    time = FeatureTimeBase(sfreq=200.0, n_times=int(round(duration_s * 200.0)), t0_s=0.0)

    # Lowpass above modulation frequency but way below carrier (envelope-only).
    cfg = HilbertEnvelopeConfig(lowpass_hz=10.0, derivative="none")

    comp = HilbertEnvelopeComputer()
    out = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)

    env_est = out.values

    # Resample ground-truth envelope to match time grid for comparison.
    # Use simple interpolation for test (not production).
    t_audio = np.arange(env_true.shape[0], dtype=float) / sfreq_audio
    t_feat = np.arange(time.n_times, dtype=float) / time.sfreq
    env_true_rs = np.interp(t_feat, t_audio, env_true)

    # Ignore edges (filtering + resampling transients).
    edge = int(round(0.2 * time.sfreq))
    core_est = env_est[edge:-edge]
    core_true = env_true_rs[edge:-edge]

    # Compare via correlation: should be very high.
    r = np.corrcoef(core_est, core_true)[0, 1]
    assert r > 0.95


def test_derivative_modes() -> None:
    sfreq_audio = 10_000.0
    x, _ = _am_sine(sfreq=sfreq_audio, carrier_hz=500.0, mod_hz=2.0, duration_s=2.0, mod_depth=0.5)

    time = FeatureTimeBase(sfreq=100.0, n_times=200, t0_s=0.0)
    comp = HilbertEnvelopeComputer()

    base = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=HilbertEnvelopeConfig(derivative="none"))
    diff = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=HilbertEnvelopeConfig(derivative="diff"))
    absdiff = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=HilbertEnvelopeConfig(derivative="absdiff"))

    assert base.values.shape == diff.values.shape == absdiff.values.shape == (time.n_times,)
    # absdiff should be >= 0 everywhere
    assert np.all(absdiff.values >= -1e-12)
    # diff should have both positive and negative values for a varying envelope
    assert np.any(diff.values > 0.0)
    assert np.any(diff.values < 0.0)


def test_lowpass_invalid_cutoff_raises() -> None:
    sfreq_audio = 1_000.0
    audio = _sine(sfreq=sfreq_audio, freq_hz=50.0, duration_s=1.0)

    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)
    comp = HilbertEnvelopeComputer()

    # Nyquist at target grid is 50 Hz
    with pytest.raises(ValueError):
        _ = comp.compute(
            time=time,
            audio=audio,
            audio_sfreq=sfreq_audio,
            config=HilbertEnvelopeConfig(lowpass_hz=60.0),
        )


def test_multichannel_downmix_mean() -> None:
    sfreq_audio = 10_000.0
    a1 = _sine(sfreq=sfreq_audio, freq_hz=200.0, duration_s=1.0, amplitude=1.0)
    a2 = _sine(sfreq=sfreq_audio, freq_hz=200.0, duration_s=1.0, amplitude=0.5)
    audio = np.stack([a1, a2], axis=0)  # (channels, time)

    time = FeatureTimeBase(sfreq=200.0, n_times=200, t0_s=0.0)
    comp = HilbertEnvelopeComputer()

    out = comp.compute(
        time=time,
        audio=audio,
        audio_sfreq=sfreq_audio,
        config=HilbertEnvelopeConfig(downmix="mean"),
    )

    assert out.values.shape == (time.n_times,)
