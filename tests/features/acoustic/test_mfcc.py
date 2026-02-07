# =============================================================================
# =============================================================================
#                 ############################################
#                 #              TESTS: MFCC                #
#                 ############################################
# =============================================================================
# =============================================================================

from __future__ import annotations

import numpy as np
import pytest

from dcap.features.acoustic.mfcc import MfccComputer, MfccConfig
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
def test_mfcc_returns_exact_timebase_length(mode: str) -> None:
    sfreq_audio = 16_000.0
    duration_s = 1.2
    x = _sine(sfreq=sfreq_audio, freq_hz=440.0, duration_s=duration_s)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)
    cfg = MfccConfig(mode=mode, n_fft=512, hop_length=160, n_mels=40, n_mfcc=13)

    out = MfccComputer().compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)

    assert out.values.ndim == 2
    assert out.values.shape[1] == time.n_times
    assert out.values.shape[0] == 13
    assert len(out.channel_names) == 13
    assert np.all(np.isfinite(out.values))


def test_mfcc_delta_stacks_increase_feature_dim() -> None:
    sfreq_audio = 16_000.0
    x = _sine(sfreq=sfreq_audio, freq_hz=500.0, duration_s=1.0)
    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)

    base = MfccComputer().compute(
        time=time,
        audio=x,
        audio_sfreq=sfreq_audio,
        config=MfccConfig(n_mels=40, n_mfcc=13, include_delta=False, include_delta2=False),
    )
    d1 = MfccComputer().compute(
        time=time,
        audio=x,
        audio_sfreq=sfreq_audio,
        config=MfccConfig(n_mels=40, n_mfcc=13, include_delta=True, include_delta2=False),
    )
    d2 = MfccComputer().compute(
        time=time,
        audio=x,
        audio_sfreq=sfreq_audio,
        config=MfccConfig(n_mels=40, n_mfcc=13, include_delta=True, include_delta2=True),
    )

    assert base.values.shape[0] == 13
    assert d1.values.shape[0] == 26
    assert d2.values.shape[0] == 39

    assert len(d1.channel_names) == 26
    assert len(d2.channel_names) == 39


def test_mfcc_lifter_changes_values_but_not_shape() -> None:
    sfreq_audio = 16_000.0
    duration_s = 1.0
    x = _sine(sfreq=sfreq_audio, freq_hz=800.0, duration_s=duration_s)

    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)

    out0 = MfccComputer().compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=MfccConfig(n_mels=40, n_mfcc=13, lifter=0))
    outL = MfccComputer().compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=MfccConfig(n_mels=40, n_mfcc=13, lifter=22))

    assert out0.values.shape == outL.values.shape
    assert not np.allclose(out0.values, outL.values, atol=1e-12, rtol=1e-8)


def test_invalid_n_mfcc_gt_n_mels_raises() -> None:
    sfreq_audio = 16_000.0
    x = _sine(sfreq=sfreq_audio, freq_hz=440.0, duration_s=1.0)
    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)

    with pytest.raises(ValueError):
        _ = MfccComputer().compute(
            time=time,
            audio=x,
            audio_sfreq=sfreq_audio,
            config=MfccConfig(n_mels=20, n_mfcc=40),
        )


def test_missing_audio_raises() -> None:
    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)
    with pytest.raises(ValueError):
        _ = MfccComputer().compute(time=time, config=MfccConfig(), audio=None, audio_sfreq=None)
