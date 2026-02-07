# =============================================================================
# =============================================================================
#                 ############################################
#                 #        TESTS: MEL SPECTROGRAM FEATURE     #
#                 ############################################
# =============================================================================
# =============================================================================

from __future__ import annotations

import numpy as np
import pytest

from dcap.features.acoustic.mel_spectrogram import MelSpectrogramComputer, MelSpectrogramConfig
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


@pytest.mark.parametrize("mode", ["efficient", "accurate"])
def test_mel_spectrogram_returns_exact_timebase_length(mode: str) -> None:
    sfreq_audio = 16_000.0
    duration_s = 1.2
    x = _sine(sfreq=sfreq_audio, freq_hz=440.0, duration_s=duration_s)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    cfg = MelSpectrogramConfig(mode=mode, n_fft=512, hop_length=160, n_mels=40, output="log_power")

    out = MelSpectrogramComputer().compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)

    assert out.values.ndim == 2
    assert out.values.shape[1] == time.n_times
    assert out.values.shape[0] == 40
    assert len(out.channel_names) == 40
    assert np.all(np.isfinite(out.values))


def test_mel_spectrogram_missing_audio_raises() -> None:
    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)
    with pytest.raises(ValueError):
        _ = MelSpectrogramComputer().compute(time=time, config=MelSpectrogramConfig(), audio=None, audio_sfreq=None)


def test_mel_log_power_is_monotonic_transform_of_power() -> None:
    sfreq_audio = 16_000.0
    duration_s = 1.2
    x = _sine(sfreq=sfreq_audio, freq_hz=440.0, duration_s=duration_s)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    comp = MelSpectrogramComputer()

    cfg_pow = MelSpectrogramConfig(mode="efficient", n_fft=512, hop_length=160, n_mels=40, output="power")
    cfg_log = MelSpectrogramConfig(mode="efficient", n_fft=512, hop_length=160, n_mels=40, output="log_power", log_floor=1e-12)

    out_pow = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg_pow)
    out_log = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg_log)

    p = out_pow.values.ravel()
    l = out_log.values.ravel()
    l2 = np.log10(np.maximum(p, 1e-12))

    r = float(np.corrcoef(_zscore(l), _zscore(l2))[0, 1])
    assert r > 0.99


def test_mel_band_energy_peaks_near_tone_frequency() -> None:
    """
    For a pure tone, mel energy should concentrate around a mel band whose center is near the tone frequency.
    """
    sfreq_audio = 16_000.0
    duration_s = 1.5
    f0 = 1000.0
    x = _sine(sfreq=sfreq_audio, freq_hz=f0, duration_s=duration_s)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    cfg = MelSpectrogramConfig(
        mode="efficient",
        n_fft=1024,
        hop_length=160,
        n_mels=64,
        fmin_hz=0.0,
        fmax_hz=8000.0,
        output="power",
        norm="slaney",
    )

    out = MelSpectrogramComputer().compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)

    mel_centers = np.asarray(out.meta["mel_centers_hz"], dtype=float)
    mean_energy = np.mean(out.values, axis=1)
    peak_idx = int(np.argmax(mean_energy))
    peak_center = float(mel_centers[peak_idx])

    # We don't expect exact; mel spacing is coarse. Allow broad tolerance.
    assert abs(peak_center - f0) < 600.0


def test_mel_n_mels_controls_feature_count() -> None:
    sfreq_audio = 16_000.0
    x = _sine(sfreq=sfreq_audio, freq_hz=440.0, duration_s=1.0)
    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)

    out_20 = MelSpectrogramComputer().compute(
        time=time,
        audio=x,
        audio_sfreq=sfreq_audio,
        config=MelSpectrogramConfig(n_mels=20),
    )
    out_80 = MelSpectrogramComputer().compute(
        time=time,
        audio=x,
        audio_sfreq=sfreq_audio,
        config=MelSpectrogramConfig(n_mels=80),
    )

    assert out_20.values.shape[0] == 20
    assert out_80.values.shape[0] == 80
