# =============================================================================
# =============================================================================
#                 ############################################
#                 #            TESTS: MIDBRAIN FEATURE       #
#                 ############################################
# =============================================================================
# =============================================================================

import numpy as np
import pytest

from dcap.features.acoustic.cochleogram import CochleogramConfig
from dcap.features.acoustic.midbrain import MidbrainComputer, MidbrainConfig
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


# =============================================================================
#                               INVARIANTS
# =============================================================================

@pytest.mark.parametrize("mid_mode", ["efficient", "accurate"])
def test_midbrain_returns_exact_timebase_length(mid_mode: str) -> None:
    sfreq_audio = 16_000.0
    duration_s = 1.2
    x = _sine(sfreq=sfreq_audio, freq_hz=300.0, duration_s=duration_s, amplitude=1.0)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)

    cfg = MidbrainConfig(
        cochleogram=CochleogramConfig(
            mode="efficient",
            f_min_hz=100.0,
            f_max_hz=6000.0,
            octave_spacing=0.5,     # keep channels small
            env_target_fs_hz=200.0,
            synapse_lowpass_cutoff_hz=30.0,
        ),
        mode=mid_mode,
        temporal_mods_hz=(2.0, 4.0),            # keep small for test speed
        spectral_mods_cyc_per_oct=(0.5, 1.0),   # keep small for test speed
        temporal_bandwidth_hz=2.0,
        spectral_bandwidth_cyc_per_oct=0.8,
        temporal_kernel_cycles=4.0,
        spectral_kernel_cycles=4.0,
        power=True,
    )

    comp = MidbrainComputer()
    out = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)

    assert out.values.ndim == 2
    assert out.values.shape[1] == time.n_times
    assert out.time.sfreq == time.sfreq
    assert out.time.n_times == time.n_times
    assert out.kind == "acoustic"
    assert len(out.channel_names) == out.values.shape[0]
    assert np.all(np.isfinite(out.values))


def test_midbrain_feature_count_matches_expected_product() -> None:
    sfreq_audio = 16_000.0
    duration_s = 1.0
    x = _sine(sfreq=sfreq_audio, freq_hz=500.0, duration_s=duration_s, amplitude=1.0)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)

    coch_cfg = CochleogramConfig(
        mode="efficient",
        f_min_hz=100.0,
        f_max_hz=6000.0,
        octave_spacing=0.5,
        env_target_fs_hz=200.0,
        synapse_lowpass_cutoff_hz=30.0,
    )

    temporal_mods = (2.0, 4.0, 8.0)
    spectral_mods = (0.5, 1.0)

    cfg = MidbrainConfig(
        cochleogram=coch_cfg,
        mode="efficient",
        temporal_mods_hz=temporal_mods,
        spectral_mods_cyc_per_oct=spectral_mods,
        temporal_bandwidth_hz=2.0,
        spectral_bandwidth_cyc_per_oct=0.8,
        temporal_kernel_cycles=4.0,
        spectral_kernel_cycles=4.0,
        power=True,
    )

    out = MidbrainComputer().compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)

    # Midbrain flattens: C * n_tm * n_sm
    n_coch_channels = int(out.meta["n_coch_channels"])
    expected = n_coch_channels * len(temporal_mods) * len(spectral_mods)
    assert out.values.shape[0] == expected
    assert len(out.channel_names) == expected


# =============================================================================
#                       COCHLEOGRAM MODE PROPAGATION
# =============================================================================

@pytest.mark.parametrize("coch_mode", ["efficient", "accurate"])
def test_midbrain_runs_with_both_cochleogram_modes(coch_mode: str) -> None:
    sfreq_audio = 16_000.0
    duration_s = 1.1
    x, _ = _am_sine(sfreq=sfreq_audio, carrier_hz=800.0, mod_hz=2.0, duration_s=duration_s, mod_depth=0.7)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)

    cfg = MidbrainConfig(
        cochleogram=CochleogramConfig(
            mode=coch_mode,
            f_min_hz=100.0,
            f_max_hz=6000.0,
            octave_spacing=0.5,
            env_target_fs_hz=200.0,
            synapse_lowpass_cutoff_hz=30.0,
        ),
        mode="efficient",
        temporal_mods_hz=(2.0, 4.0),
        spectral_mods_cyc_per_oct=(0.5, 1.0),
        temporal_bandwidth_hz=2.0,
        spectral_bandwidth_cyc_per_oct=0.8,
        temporal_kernel_cycles=4.0,
        spectral_kernel_cycles=4.0,
        power=True,
    )

    out = MidbrainComputer().compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=cfg)
    assert out.meta["cochleogram_mode"] == coch_mode
    assert np.all(np.isfinite(out.values))


# =============================================================================
#                         MIDBRAIN MODE NON-IDENTITY
# =============================================================================

def test_midbrain_modes_both_run_and_report_mode() -> None:
    sfreq_audio = 16_000.0
    duration_s = 1.2
    x, _ = _am_sine(sfreq=sfreq_audio, carrier_hz=900.0, mod_hz=2.5, duration_s=duration_s, mod_depth=0.8)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)

    coch_cfg = CochleogramConfig(
        mode="efficient",
        f_min_hz=100.0,
        f_max_hz=6000.0,
        octave_spacing=0.5,
        env_target_fs_hz=200.0,
        synapse_lowpass_cutoff_hz=30.0,
    )

    base = dict(
        cochleogram=coch_cfg,
        temporal_mods_hz=(2.0, 4.0),
        spectral_mods_cyc_per_oct=(0.5, 1.0),
        temporal_bandwidth_hz=2.0,
        spectral_bandwidth_cyc_per_oct=0.8,
        temporal_kernel_cycles=4.0,
        spectral_kernel_cycles=4.0,
        power=True,
    )

    comp = MidbrainComputer()

    out_eff = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=MidbrainConfig(mode="efficient", **base))
    out_acc = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=MidbrainConfig(mode="accurate", **base))

    assert out_eff.values.shape == out_acc.values.shape
    assert out_eff.meta["midbrain_mode"] == "efficient"
    assert out_acc.meta["midbrain_mode"] == "accurate"
    assert np.all(np.isfinite(out_eff.values))
    assert np.all(np.isfinite(out_acc.values))


# =============================================================================
#                       POWER VS MAGNITUDE CONSISTENCY
# =============================================================================

def test_power_vs_magnitude_relationship() -> None:
    """
    If power=False returns |resp| and power=True returns resp^2 (approximately),
    then the two should be monotonically related.

    We check correlation between:
      log(power + eps) and log(|mag|^2 + eps)
    """
    sfreq_audio = 16_000.0
    duration_s = 1.0
    x, _ = _am_sine(sfreq=sfreq_audio, carrier_hz=800.0, mod_hz=2.0, duration_s=duration_s, mod_depth=0.7)

    time = FeatureTimeBase(sfreq=100.0, n_times=int(round(duration_s * 100.0)), t0_s=0.0)

    coch_cfg = CochleogramConfig(
        mode="efficient",
        f_min_hz=100.0,
        f_max_hz=6000.0,
        octave_spacing=0.5,
        env_target_fs_hz=200.0,
        synapse_lowpass_cutoff_hz=30.0,
    )

    common = dict(
        cochleogram=coch_cfg,
        mode="efficient",
        temporal_mods_hz=(2.0,),
        spectral_mods_cyc_per_oct=(0.5,),
        temporal_bandwidth_hz=2.0,
        spectral_bandwidth_cyc_per_oct=0.8,
        temporal_kernel_cycles=4.0,
        spectral_kernel_cycles=4.0,
    )

    comp = MidbrainComputer()
    out_pow = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=MidbrainConfig(power=True, **common))
    out_mag = comp.compute(time=time, audio=x, audio_sfreq=sfreq_audio, config=MidbrainConfig(power=False, **common))

    a = out_pow.values.ravel()
    b = out_mag.values.ravel()

    eps = 1e-12
    log_a = np.log(a + eps)
    log_b2 = np.log((b * b) + eps)

    r = float(np.corrcoef(_zscore(log_a), _zscore(log_b2))[0, 1])
    assert r > 0.95


# =============================================================================
#                               ERROR HANDLING
# =============================================================================

def test_missing_audio_raises() -> None:
    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)
    comp = MidbrainComputer()
    with pytest.raises(ValueError):
        _ = comp.compute(time=time, config=MidbrainConfig(), audio=None, audio_sfreq=None)


def test_empty_mod_lists_raise() -> None:
    sfreq_audio = 16_000.0
    x = _sine(sfreq=sfreq_audio, freq_hz=400.0, duration_s=1.0)

    time = FeatureTimeBase(sfreq=100.0, n_times=100, t0_s=0.0)
    comp = MidbrainComputer()

    with pytest.raises(ValueError):
        _ = comp.compute(
            time=time,
            audio=x,
            audio_sfreq=sfreq_audio,
            config=MidbrainConfig(
                cochleogram=CochleogramConfig(octave_spacing=0.5),
                temporal_mods_hz=(),
                spectral_mods_cyc_per_oct=(0.5,),
            ),
        )

    with pytest.raises(ValueError):
        _ = comp.compute(
            time=time,
            audio=x,
            audio_sfreq=sfreq_audio,
            config=MidbrainConfig(
                cochleogram=CochleogramConfig(octave_spacing=0.5),
                temporal_mods_hz=(2.0,),
                spectral_mods_cyc_per_oct=(),
            ),
        )
