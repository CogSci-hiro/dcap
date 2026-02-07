# =============================================================================
# =============================================================================
#                 ############################################
#                 #           TESTS: SPECTROGRAM FEATURE     #
#                 ############################################
# =============================================================================
# =============================================================================

from __future__ import annotations

import numpy as np
import pytest

from dcap.features.acoustic.spectrogram import SpectrogramComputer, SpectrogramConfig
from dcap.features.types import FeatureTimeBase


def _sine(*, sfreq: float, freq_hz: float, duration_s: float, amplitude: float = 1.0) -> np.ndarray:
    n = int(round(duration_s * sfreq))
    t = np.arange(n, dtype=float) / float(sfreq)
    return amplitude * np.sin(2.0 * np.pi * float(freq_hz) * t)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd == 0.0:
        return x * 0.0
    return (x - mu) / sd


# =============================================================================
#                               INVARIANTS
# =============================================================================

@pytest.mark.parametrize("mode", ["efficient", "accurate"])
def test_spectrogram_returns_exact_timebase_length(mode: str) -> None:
    sfreq_audio = 16_000.0
    duration_s = 1.2
    x = _sine(sfreq=sfreq_audio, freq_hz=440.0, duration_s=duration_s)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    cfg = SpectrogramConfig(mode=mode, n_fft=512, hop_length=160, output="log_power")

    out = SpectrogramComputer().compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)

    assert out.values.ndim == 2
    assert out.values.shape[1] == time.n_times
    assert len(out.channel_names) == out.values.shape[0]
    assert np.all(np.isfinite(out.values))


def test_spectrogram_missing_audio_raises() -> None:
    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)
    with pytest.raises(ValueError):
        _ = SpectrogramComputer().compute(time=time, config=SpectrogramConfig(), audio=None, audio_sfreq=None)


# =============================================================================
#                          BASIC FREQUENCY LOCALIZATION
# =============================================================================

def test_spectrogram_has_peak_near_sine_frequency() -> None:
    sfreq_audio = 16_000.0
    duration_s = 1.5
    f0 = 500.0
    x = _sine(sfreq=sfreq_audio, freq_hz=f0, duration_s=duration_s)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    cfg = SpectrogramConfig(mode="efficient", n_fft=1024, hop_length=160, output="power")

    out = SpectrogramComputer().compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)

    # Average power over time; peak bin should be near f0
    mean_power = np.mean(out.values, axis=1)

    # Parse frequencies from channel names (f_{hz}hz)
    freqs = np.array([float(name.split("_")[1].replace("hz", "")) for name in out.channel_names], dtype=float)
    peak_f = float(freqs[int(np.argmax(mean_power))])

    # Resolution is fs/n_fft; allow a couple bins
    bin_hz = sfreq_audio / cfg.n_fft
    assert abs(peak_f - f0) < 2.5 * bin_hz


# =============================================================================
#                          OUTPUT SCALE CONSISTENCY
# =============================================================================

def test_log_power_is_monotonic_transform_of_power() -> None:
    sfreq_audio = 16_000.0
    duration_s = 1.2
    x = _sine(sfreq=sfreq_audio, freq_hz=440.0, duration_s=duration_s)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    comp = SpectrogramComputer()

    cfg_pow = SpectrogramConfig(mode="efficient", n_fft=512, hop_length=160, output="power")
    cfg_log = SpectrogramConfig(mode="efficient", n_fft=512, hop_length=160, output="log_power", log_floor=1e-12)

    out_pow = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg_pow)
    out_log = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg_log)

    # Compare after flattening; log10(power) should correlate strongly with log output
    p = out_pow.values.ravel()
    l = out_log.values.ravel()

    l2 = np.log10(np.maximum(p, 1e-12))
    r = float(np.corrcoef(_zscore(l), _zscore(l2))[0, 1])
    assert r > 0.99


# =============================================================================
#                          FREQUENCY CROPPING
# =============================================================================

def test_frequency_cropping_reduces_bins() -> None:
    sfreq_audio = 16_000.0
    duration_s = 1.0
    x = _sine(sfreq=sfreq_audio, freq_hz=440.0, duration_s=duration_s)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    comp = SpectrogramComputer()

    out_full = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=SpectrogramConfig(n_fft=512, hop_length=160))
    out_crop = comp.compute(
        time=time,
        audio=x,
        audio_sfreq=sfreq_audio,
        config=SpectrogramConfig(n_fft=512, hop_length=160, fmin_hz=300.0, fmax_hz=2000.0),
    )

    assert out_crop.values.shape[0] < out_full.values.shape[0]
    assert np.all(np.isfinite(out_crop.values))
