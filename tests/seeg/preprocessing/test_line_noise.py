# =============================================================================
# =============================================================================
#                     ########################################
#                     #        TEST: LINE NOISE REMOVAL      #
#                     ########################################
# =============================================================================
# =============================================================================
"""
Tests for line-noise removal (notch + zapline) with both:
- analysis-friendly API (MNE-like): remove_line_noise(raw, method=..., ...)
- clinical wrapper API: remove_line_noise_view(raw, cfg, ctx)

These tests focus on:
- spectral effect at the line frequency and harmonics
- preservation of non-target frequencies (e.g., 10 Hz)
- channel picks semantics (only picked channels change)
- metadata invariants (sfreq, n_times, annotations)
- wrapper provenance + artifact fields (basic)

ZapLine tests are skipped automatically if `meegkit` is not installed.

Usage
-----
    pytest -q
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable

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
_LINE_HZ: float = 50.0
_NON_TARGET_HZ: float = 10.0
_HARMONICS_HZ: tuple[float, ...] = (50.0, 100.0, 150.0)
_MAX_HARMONIC_HZ: float = 150.0

_TOL: float = 1e-8


# =============================================================================
#                     ########################################
#                     #          Import compatibility         #
#                     ########################################
# =============================================================================
def _import_analysis_api() -> Callable[..., mne.io.BaseRaw]:
    """
    Import the analysis-friendly API entrypoint.

    Expected after refactor:
        from dcap.seeg.preprocessing.blocks.line_noise import remove_line_noise
    """
    try:
        from dcap.seeg.preprocessing.blocks.line_noise import remove_line_noise  # type: ignore
        return remove_line_noise
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import analysis API remove_line_noise from "
            "dcap.seeg.preprocessing.blocks.line_noise. Adjust this helper if paths changed."
        ) from exc


def _import_wrapper_api() -> Optional[Callable[..., Tuple[mne.io.BaseRaw, Any]]]:
    """
    Import the clinical wrapper API entrypoint if it exists.

    Expected after refactor:
        from dcap.seeg.preprocessing.blocks.line_noise import remove_line_noise_view
    """
    try:
        from dcap.seeg.preprocessing.blocks.line_noise import remove_line_noise_view  # type: ignore
        return remove_line_noise_view
    except Exception:
        return None


# =============================================================================
#                     ########################################
#                     #               Helpers                #
#                     ########################################
# =============================================================================
def _make_toy_raw(*, include_harmonics: bool = True) -> mne.io.BaseRaw:
    """
    Create a tiny Raw with:
    - A1: line noise + non-target component
    - A2: same signal (used for picks tests)
    - STI 014: stim channel (should remain unchanged by default picks)
    """
    n_times = int(round(_SFREQ_HZ * _DURATION_S))
    t = np.arange(n_times, dtype=float) / _SFREQ_HZ

    # Compose signal: strong line + moderate 10 Hz
    sig = 5.0 * np.sin(2.0 * np.pi * _LINE_HZ * t) + 1.0 * np.sin(2.0 * np.pi * _NON_TARGET_HZ * t)

    if include_harmonics:
        sig = sig + 2.0 * np.sin(2.0 * np.pi * 100.0 * t) + 1.5 * np.sin(2.0 * np.pi * 150.0 * t)

    rng = np.random.default_rng(0)
    sig = sig + 0.05 * rng.standard_normal(size=n_times)

    data = np.vstack([sig, sig.copy(), np.zeros_like(sig)])
    ch_names = ["A1", "A2", "STI 014"]
    ch_types = ["seeg", "seeg", "stim"]
    info = mne.create_info(ch_names=ch_names, sfreq=_SFREQ_HZ, ch_types=ch_types)

    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(mne.Annotations(onset=[0.2], duration=[0.1], description=["test"]))
    return raw


def _bandpower_fft(x: np.ndarray, *, sfreq: float, f0: float, bw_hz: float = 1.0) -> float:
    """
    Compute band power around f0 using FFT magnitude squared.

    Power is integrated over [f0-bw_hz, f0+bw_hz].
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    psd = (np.abs(X) ** 2) / max(n, 1)

    mask = (freqs >= (f0 - bw_hz)) & (freqs <= (f0 + bw_hz))
    return float(np.sum(psd[mask]))


# =============================================================================
#                     ########################################
#                     #         Dummy wrapper objects         #
#                     ########################################
# =============================================================================
@dataclass(frozen=True)
class _DummyLineNoiseCfg:
    method: str = "notch"            # "notch" | "zapline"
    freq_base: float = 50.0
    max_harmonic_hz: float = 150.0
    picks: Optional[np.ndarray] = None
    # zapline-only
    chunk_sec: float = 60.0
    nremove: int = 1


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
def test_compute_line_freqs_harmonics() -> None:
    try:
        from dcap.seeg.preprocessing.blocks.line_noise import _compute_line_freqs  # type: ignore
    except Exception:
        pytest.skip("Could not import _compute_line_freqs; skipping direct helper test.")

    freqs = _compute_line_freqs(50.0, 260.0)
    assert freqs == [50.0, 100.0, 150.0, 200.0, 250.0]


def test_notch_reduces_line_and_harmonics_power() -> None:
    remove_line_noise = _import_analysis_api()
    raw = _make_toy_raw(include_harmonics=True)

    x_before = raw.get_data(picks=[0])[0]
    before = {f: _bandpower_fft(x_before, sfreq=_SFREQ_HZ, f0=f, bw_hz=1.0) for f in _HARMONICS_HZ}

    raw_clean = remove_line_noise(
        raw,
        method="notch",
        freq_base=_LINE_HZ,
        max_harmonic_hz=_MAX_HARMONIC_HZ,
        picks=None,
        copy=True,
    )

    x_after = raw_clean.get_data(picks=[0])[0]
    after = {f: _bandpower_fft(x_after, sfreq=_SFREQ_HZ, f0=f, bw_hz=1.0) for f in _HARMONICS_HZ}

    for f in _HARMONICS_HZ:
        assert after[f] < 0.5 * before[f], f"Expected notch to reduce power near {f} Hz"


def test_notch_preserves_non_target_frequency() -> None:
    remove_line_noise = _import_analysis_api()
    raw = _make_toy_raw(include_harmonics=True)

    x_before = raw.get_data(picks=[0])[0]
    p10_before = _bandpower_fft(x_before, sfreq=_SFREQ_HZ, f0=_NON_TARGET_HZ, bw_hz=1.0)

    raw_clean = remove_line_noise(
        raw,
        method="notch",
        freq_base=_LINE_HZ,
        max_harmonic_hz=_MAX_HARMONIC_HZ,
        picks=None,
        copy=True,
    )

    x_after = raw_clean.get_data(picks=[0])[0]
    p10_after = _bandpower_fft(x_after, sfreq=_SFREQ_HZ, f0=_NON_TARGET_HZ, bw_hz=1.0)

    assert p10_after > 0.7 * p10_before


def test_picks_only_modify_selected_channels() -> None:
    remove_line_noise = _import_analysis_api()
    raw = _make_toy_raw(include_harmonics=False)

    picks = np.array([0], dtype=int)

    before_a1 = raw.get_data(picks=[0])[0].copy()
    before_a2 = raw.get_data(picks=[1])[0].copy()

    raw_clean = remove_line_noise(
        raw,
        method="notch",
        freq_base=_LINE_HZ,
        max_harmonic_hz=_MAX_HARMONIC_HZ,
        picks=picks,
        copy=True,
    )

    after_a1 = raw_clean.get_data(picks=[0])[0]
    after_a2 = raw_clean.get_data(picks=[1])[0]

    npt.assert_allclose(after_a2, before_a2, atol=_TOL)
    assert not np.allclose(after_a1, before_a1, atol=_TOL)


def test_default_picks_do_not_modify_stim_channel() -> None:
    remove_line_noise = _import_analysis_api()
    raw = _make_toy_raw(include_harmonics=False)

    stim_idx = raw.ch_names.index("STI 014")
    stim_before = raw.get_data(picks=[stim_idx])[0].copy()

    raw_clean = remove_line_noise(
        raw,
        method="notch",
        freq_base=_LINE_HZ,
        max_harmonic_hz=_MAX_HARMONIC_HZ,
        picks=None,
        copy=True,
    )

    stim_after = raw_clean.get_data(picks=[stim_idx])[0]
    npt.assert_allclose(stim_after, stim_before, atol=_TOL)


def test_metadata_invariants_notch() -> None:
    remove_line_noise = _import_analysis_api()
    raw = _make_toy_raw(include_harmonics=False)

    raw_clean = remove_line_noise(
        raw,
        method="notch",
        freq_base=_LINE_HZ,
        max_harmonic_hz=_MAX_HARMONIC_HZ,
        picks=None,
        copy=True,
    )

    assert raw_clean.info["sfreq"] == raw.info["sfreq"]
    assert raw_clean.n_times == raw.n_times
    assert raw_clean.ch_names == raw.ch_names
    assert len(raw_clean.annotations) == len(raw.annotations)
    assert np.isfinite(raw_clean.get_data()).all()


@pytest.mark.skipif(_import_wrapper_api() is None, reason="remove_line_noise_view wrapper not available")
def test_wrapper_records_provenance_and_returns_artifact() -> None:
    remove_line_noise_view = _import_wrapper_api()
    assert remove_line_noise_view is not None

    raw = _make_toy_raw(include_harmonics=False)
    cfg = _DummyLineNoiseCfg(method="notch", freq_base=_LINE_HZ, max_harmonic_hz=_MAX_HARMONIC_HZ, picks=None)
    ctx = _DummyCtx()

    raw_out, artifact = remove_line_noise_view(raw, cfg, ctx)  # type: ignore[arg-type]

    assert len(ctx.records) == 1
    assert ctx.records[0][0] == "line_noise"

    assert getattr(artifact, "name") == "line_noise"
    assert "freqs_applied" in getattr(artifact, "parameters")
    assert getattr(artifact, "summary_metrics")["n_freqs"] >= 1.0

    assert raw_out.info["sfreq"] == raw.info["sfreq"]
    assert raw_out.n_times == raw.n_times


# =============================================================================
#                     ########################################
#                     #               ZapLine                #
#                     ########################################
# =============================================================================
def test_zapline_reduces_line_power_if_available() -> None:
    pytest.importorskip("meegkit")
    remove_line_noise = _import_analysis_api()

    raw = _make_toy_raw(include_harmonics=False)
    raw.load_data()

    x_before = raw.get_data(picks=[0])[0]
    before = _bandpower_fft(x_before, sfreq=_SFREQ_HZ, f0=_LINE_HZ, bw_hz=1.0)

    raw_clean = remove_line_noise(
        raw,
        method="zapline",
        freq_base=_LINE_HZ,
        picks=None,
        chunk_sec=1.0,
        nremove=1,
        copy=True,
    )

    x_after = raw_clean.get_data(picks=[0])[0]
    after = _bandpower_fft(x_after, sfreq=_SFREQ_HZ, f0=_LINE_HZ, bw_hz=1.0)

    assert after < 0.8 * before


def test_zapline_chunk_size_consistency_if_available() -> None:
    pytest.importorskip("meegkit")
    remove_line_noise = _import_analysis_api()

    raw = _make_toy_raw(include_harmonics=False)
    raw.load_data()

    clean_1 = remove_line_noise(
        raw,
        method="zapline",
        freq_base=_LINE_HZ,
        picks=None,
        chunk_sec=1.0,
        nremove=1,
        copy=True,
    )
    clean_2 = remove_line_noise(
        raw,
        method="zapline",
        freq_base=_LINE_HZ,
        picks=None,
        chunk_sec=2.0,
        nremove=1,
        copy=True,
    )

    x1 = clean_1.get_data(picks=[0])[0]
    x2 = clean_2.get_data(picks=[0])[0]

    denom = float(np.linalg.norm(x1)) + 1e-12
    rel = float(np.linalg.norm(x1 - x2)) / denom
    assert rel < 0.2
