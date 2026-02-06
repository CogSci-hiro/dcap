# =============================================================================
# =============================================================================
#                     ########################################
#                     #           TEST: FILTERING            #
#                     ########################################
# =============================================================================
# =============================================================================
"""
Tests for filtering utilities:
- High-pass filter (drift removal)
- Gamma/HFA envelope extraction

Covers both:
- analysis API: highpass(...), gamma_envelope(...)
- clinical wrappers (if available): highpass_view(...), gamma_envelope_view(...)

Test strategy
-------------
Highpass:
- Construct a signal with strong low-frequency drift + a higher-frequency component.
- Verify low-frequency band power is reduced after highpass.

Gamma envelope:
- Construct amplitude-modulated high-frequency carrier:
    x(t) = (1 + m*sin(2π f_mod t)) * sin(2π f_carrier t)
- Envelope should correlate with the modulation (f_mod).
- Channel naming suffix behavior is verified (if suffix is used in implementation).
- Metadata invariants: sfreq, n_times, annotations preserved.

Notes
-----
- These are "physics-ish" tests: thresholds are permissive to avoid brittleness.
- If your implementation chooses different output channel naming for envelopes,
  adjust the corresponding assertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import numpy.testing as npt
import pytest

import mne


# =============================================================================
#                     ########################################
#                     #              Constants               #
#                     ########################################
# =============================================================================
_SFREQ_HZ: float = 1000.0
_DURATION_S: float = 5.0
_TOL: float = 1e-8


# =============================================================================
#                     ########################################
#                     #          Import compatibility         #
#                     ########################################
# =============================================================================
def _import_analysis_highpass() -> Callable[..., mne.io.BaseRaw]:
    try:
        from dcap.seeg.preprocessing.blocks.filtering import highpass  # type: ignore
        return highpass
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import analysis API highpass from dcap.seeg.preprocessing.blocks.filtering."
        ) from exc


def _import_analysis_gamma_envelope() -> Callable[..., mne.io.BaseRaw]:
    try:
        from dcap.seeg.preprocessing.blocks.filtering import gamma_envelope  # type: ignore
        return gamma_envelope
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import analysis API gamma_envelope from dcap.seeg.preprocessing.blocks.filtering."
        ) from exc


def _import_wrapper_highpass() -> Optional[Callable[..., Tuple[mne.io.BaseRaw, Any]]]:
    try:
        from dcap.seeg.preprocessing.blocks.filtering import highpass_view  # type: ignore
        return highpass_view
    except Exception:
        return None


def _import_wrapper_gamma_envelope() -> Optional[Callable[..., Tuple[mne.io.BaseRaw, Any]]]:
    try:
        from dcap.seeg.preprocessing.blocks.filtering import gamma_envelope_view  # type: ignore
        return gamma_envelope_view
    except Exception:
        return None


# =============================================================================
#                     ########################################
#                     #               Helpers                #
#                     ########################################
# =============================================================================
def _bandpower_fft(x: np.ndarray, *, sfreq: float, fmin: float, fmax: float) -> float:
    """
    Integrate FFT power over [fmin, fmax] (very simple PSD proxy).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    psd = (np.abs(X) ** 2) / max(n, 1)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.sum(psd[mask]))


def _make_raw(data_ch_time: np.ndarray, *, sfreq: float, ch_names: list[str], ch_types: list[str]) -> mne.io.BaseRaw:
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data_ch_time, info, verbose=False)
    raw.set_annotations(mne.Annotations(onset=[0.2], duration=[0.1], description=["test"]))
    return raw


# =============================================================================
#                     ########################################
#                     #         Dummy wrapper objects         #
#                     ########################################
# =============================================================================
@dataclass(frozen=True)
class _DummyHighpassCfg:
    l_freq: float = 1.0
    phase: str = "zero"


@dataclass(frozen=True)
class _DummyGammaEnvCfg:
    band_hz: tuple[float, float] = (70.0, 150.0)
    method: str = "hilbert"  # "hilbert" | "rectified_smooth"
    smoothing_sec: float = 0.05


class _DummyCtx:
    def __init__(self) -> None:
        self.records: list[tuple[str, Dict[str, Any]]] = []

    def add_record(self, name: str, payload: Dict[str, Any]) -> None:
        self.records.append((name, payload))


# =============================================================================
#                     ########################################
#                     #                Tests                 #
#                     ########################################
# =============================================================================
def test_highpass_reduces_low_frequency_drift_power() -> None:
    highpass = _import_analysis_highpass()

    n_times = int(round(_SFREQ_HZ * _DURATION_S))
    t = np.arange(n_times, dtype=float) / _SFREQ_HZ

    # Strong drift (0.2 Hz) + a 10 Hz component
    drift = 5.0 * np.sin(2.0 * np.pi * 0.2 * t)
    sig10 = 1.0 * np.sin(2.0 * np.pi * 10.0 * t)
    x = drift + sig10

    raw = _make_raw(x[None, :], sfreq=_SFREQ_HZ, ch_names=["A1"], ch_types=["seeg"])

    before_low = _bandpower_fft(x, sfreq=_SFREQ_HZ, fmin=0.0, fmax=0.8)
    before_10 = _bandpower_fft(x, sfreq=_SFREQ_HZ, fmin=9.0, fmax=11.0)

    raw_hp = highpass(raw, l_freq=1.0, phase="zero", copy=True)
    y = raw_hp.get_data(picks=[0])[0]

    after_low = _bandpower_fft(y, sfreq=_SFREQ_HZ, fmin=0.0, fmax=0.8)
    after_10 = _bandpower_fft(y, sfreq=_SFREQ_HZ, fmin=9.0, fmax=11.0)

    # Low frequencies should reduce significantly
    assert after_low < 0.2 * before_low

    # 10 Hz should be mostly preserved
    assert after_10 > 0.7 * before_10

    # Metadata invariants
    assert raw_hp.info["sfreq"] == raw.info["sfreq"]
    assert raw_hp.n_times == raw.n_times
    assert len(raw_hp.annotations) == len(raw.annotations)
    assert np.isfinite(raw_hp.get_data()).all()


def test_gamma_envelope_tracks_modulation_hilbert() -> None:
    gamma_envelope = _import_analysis_gamma_envelope()

    n_times = int(round(_SFREQ_HZ * _DURATION_S))
    t = np.arange(n_times, dtype=float) / _SFREQ_HZ

    f_carrier = 100.0
    f_mod = 2.0
    m = 0.8

    env_true = 1.0 + m * np.sin(2.0 * np.pi * f_mod * t)
    x = env_true * np.sin(2.0 * np.pi * f_carrier * t)

    raw = _make_raw(x[None, :], sfreq=_SFREQ_HZ, ch_names=["A1"], ch_types=["seeg"])

    env_raw = gamma_envelope(
        raw,
        band_hz=(70.0, 150.0),
        method="hilbert",
        smoothing_sec=0.05,
        suffix="HFAenv",
        out_ch_type="misc",
        copy=True,
    )

    y = env_raw.get_data(picks=[0])[0]

    y_n = (y - np.mean(y)) / (np.std(y) + 1e-12)
    e_n = (env_true - np.mean(env_true)) / (np.std(env_true) + 1e-12)

    corr = float(np.corrcoef(y_n, e_n)[0, 1])
    assert corr > 0.8

    assert env_raw.info["sfreq"] == raw.info["sfreq"]
    assert env_raw.n_times == raw.n_times
    assert len(env_raw.annotations) == len(raw.annotations)
    assert np.isfinite(env_raw.get_data()).all()


def test_gamma_envelope_channel_names_are_suffixed_when_requested() -> None:
    gamma_envelope = _import_analysis_gamma_envelope()

    n_times = int(round(_SFREQ_HZ * 1.0))
    t = np.arange(n_times, dtype=float) / _SFREQ_HZ
    x = np.sin(2.0 * np.pi * 100.0 * t)
    raw = _make_raw(x[None, :], sfreq=_SFREQ_HZ, ch_names=["A1"], ch_types=["seeg"])

    env_raw = gamma_envelope(
        raw,
        band_hz=(70.0, 150.0),
        method="rectified_smooth",
        smoothing_sec=0.0,
        suffix="HFAenv",
        out_ch_type="misc",
        copy=True,
    )

    assert env_raw.ch_names[0].endswith("_HFAenv")


@pytest.mark.skipif(_import_wrapper_highpass() is None, reason="highpass_view wrapper not available")
def test_highpass_view_records_provenance_and_returns_artifact() -> None:
    highpass_view = _import_wrapper_highpass()
    assert highpass_view is not None

    n_times = int(round(_SFREQ_HZ * 1.0))
    t = np.arange(n_times, dtype=float) / _SFREQ_HZ
    x = np.sin(2.0 * np.pi * 10.0 * t)
    raw = _make_raw(x[None, :], sfreq=_SFREQ_HZ, ch_names=["A1"], ch_types=["seeg"])

    cfg = _DummyHighpassCfg(l_freq=1.0, phase="zero")
    ctx = _DummyCtx()

    raw_out, artifact = highpass_view(raw, cfg, ctx)  # type: ignore[arg-type]

    assert len(ctx.records) == 1
    assert ctx.records[0][0] == "highpass"
    assert getattr(artifact, "name") == "highpass"
    assert raw_out.n_times == raw.n_times


@pytest.mark.skipif(_import_wrapper_gamma_envelope() is None, reason="gamma_envelope_view wrapper not available")
def test_gamma_envelope_view_records_provenance_and_returns_artifact() -> None:
    gamma_envelope_view = _import_wrapper_gamma_envelope()
    assert gamma_envelope_view is not None

    n_times = int(round(_SFREQ_HZ * 1.0))
    t = np.arange(n_times, dtype=float) / _SFREQ_HZ
    x = np.sin(2.0 * np.pi * 100.0 * t)
    raw = _make_raw(x[None, :], sfreq=_SFREQ_HZ, ch_names=["A1"], ch_types=["seeg"])

    cfg = _DummyGammaEnvCfg(band_hz=(70.0, 150.0), method="hilbert", smoothing_sec=0.05)
    ctx = _DummyCtx()

    raw_out, artifact = gamma_envelope_view(raw, cfg, ctx)  # type: ignore[arg-type]

    assert len(ctx.records) == 1
    assert ctx.records[0][0] == "gamma_envelope"
    assert getattr(artifact, "name") == "gamma_envelope"
    assert raw_out.n_times == raw.n_times
