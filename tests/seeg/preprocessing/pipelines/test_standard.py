# =============================================================================
# =============================================================================
#                     ########################################
#                     #         TEST: STANDARD PIPELINE       #
#                     ########################################
# =============================================================================
# =============================================================================
"""
pytest smoke tests for dcap.seeg.preprocessing.pipelines.standard

Strategy
--------
- Create tiny synthetic MNE RawArray
- Run pipeline with multiple profiles/modes
- Assert outputs are saved and provenance exists
- Optionally monkeypatch heavy DSP steps to keep tests stable/fast

Usage example
-------------
    pytest -q
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import mne
import numpy as np
import pytest

from dcap.seeg.preprocessing.pipelines.standard import (
    PreprocessOutputs,
    StandardPipelineConfig,
    run_preprocess_single_raw,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _make_synthetic_raw(*, sfreq_hz: float = 1024.0, duration_sec: float = 2.0, n_channels: int = 4) -> mne.io.Raw:
    """
    Make a small synthetic RawArray suitable for quick tests.

    Parameters
    ----------
    sfreq_hz
        Sampling frequency.
    duration_sec
        Duration in seconds.
    n_channels
        Number of channels.

    Returns
    -------
    raw
        MNE RawArray.

    Usage example
    -------------
        raw = _make_synthetic_raw(sfreq_hz=512.0, duration_sec=1.0, n_channels=2)
    """
    n_times = int(np.round(duration_sec * sfreq_hz))
    rng = np.random.default_rng(0)

    # A bit of structure: add a 10 Hz oscillation + weak 50 Hz line noise component.
    t = np.arange(n_times) / sfreq_hz
    data = 1e-6 * rng.standard_normal(size=(n_channels, n_times)).astype(np.float64)
    data += (1e-6 * np.sin(2.0 * np.pi * 10.0 * t))[None, :]
    data += (2e-7 * np.sin(2.0 * np.pi * 50.0 * t))[None, :]

    ch_names = [f"SEEG{ch:03d}" for ch in range(1, n_channels + 1)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq_hz, ch_types="seeg")

    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def _base_cfg_dict() -> Dict[str, Any]:
    """
    Base config dict used across tests.

    Returns
    -------
    cfg
        Dict suitable for StandardPipelineConfig(raw=cfg).

    Usage example
    -------------
        cfg = StandardPipelineConfig(raw=_base_cfg_dict())
    """
    return {
        "version": 1,
        "io": {"overwrite": True, "preload": True},
        "pipeline": {"profile": "canonical", "stop_after": None},
        "resample": {"enabled": True, "sfreq_out_hz": 512.0},
        "line_noise": {"enabled": True, "method": "notch", "base_freq_hz": 50.0, "max_harmonic_hz": 250.0},
        "filtering": {
            "enabled": True,
            "mode": "broadband",
            "broadband": {"highpass": {"enabled": True, "l_freq_hz": 1.0, "phase": "zero"}},
            "high_gamma_envelope": {
                "enabled": False,
                "bandpass": {"l_freq_hz": 70.0, "h_freq_hz": 150.0},
                "smooth": {"enabled": True, "window_ms": 50.0},
                "downsample": {"enabled": True, "sfreq_out_hz": 128.0},
            },
        },
        "rereference": {
            "enabled": True,
            "save_all_views": True,
            "default_view": "car",
            "views": [
                {"method": "car", "name": "car", "params": {"scope": "global", "exclude_bads": True}},
                # Keep bipolar optional; include if your rereference module supports it robustly
                # {"method": "bipolar", "name": "bipolar", "params": {"exclude_bads": True}},
            ],
        },
    }


# -----------------------------------------------------------------------------
# Optional: monkeypatch DSP blocks to keep tests robust while iterating
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class _FakeArtifact:
    name: str
    parameters: Dict[str, Any]
    summary_metrics: Dict[str, float]
    warnings: List[str]
    figures: List[Any]


def _identity_step(raw: mne.io.BaseRaw, *, name: str) -> Tuple[mne.io.BaseRaw, Any]:
    artifact = _FakeArtifact(
        name=name,
        parameters={"fake": True},
        summary_metrics={"ok": 1.0},
        warnings=[],
        figures=[],
    )
    return raw.copy(), artifact


def _fake_reref_step(raw: mne.io.BaseRaw) -> Tuple[Mapping[str, mne.io.BaseRaw], Any]:
    # Provide two views to test save-branching.
    views = {
        "original": raw.copy(),
        "car": raw.copy(),
    }
    artifact = _FakeArtifact(
        name="rereference",
        parameters={"fake": True, "views": list(views.keys())},
        summary_metrics={"ok": 1.0},
        warnings=[],
        figures=[],
    )
    return views, artifact


@pytest.fixture()
def synthetic_raw() -> mne.io.Raw:
    return _make_synthetic_raw(sfreq_hz=1024.0, duration_sec=2.0, n_channels=4)


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    p = tmp_path / "derivatives"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_standard_pipeline_smoke_canonical_broadband(
    synthetic_raw: mne.io.Raw,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Canonical order + broadband filtering should produce:
    - at least one FIF output
    - a provenance JSON
    """
    # Patch step functions if you want the test to be resistant to DSP changes.
    from dcap.seeg.preprocessing.pipelines import standard as standard_mod

    monkeypatch.setattr(standard_mod, "resample_raw", lambda raw, cfg, ctx: _identity_step(raw, name="resample"))
    monkeypatch.setattr(standard_mod, "remove_line_noise_view", lambda raw, cfg, ctx: _identity_step(raw, name="line_noise"))
    monkeypatch.setattr(standard_mod, "highpass_view", lambda raw, cfg, ctx: _identity_step(raw, name="filtering"))
    monkeypatch.setattr(standard_mod, "gamma_envelope_view", lambda raw, cfg, ctx: _identity_step(raw, name="gamma_envelope"))
    monkeypatch.setattr(standard_mod, "rereference_view", lambda raw, cfg, ctx: _fake_reref_step(raw))

    cfg = StandardPipelineConfig(raw=_base_cfg_dict())
    outputs: PreprocessOutputs = run_preprocess_single_raw(
        raw=synthetic_raw,
        cfg=cfg,
        out_dir=out_dir,
        base_stem="sub-TEST_task-conversation_run-1",
    )

    assert outputs.provenance_path.exists()
    assert len(outputs.saved_paths) >= 1
    for p in outputs.saved_paths:
        assert p.exists()
        assert p.suffix == ".fif"

    # Validate provenance JSON is parseable and references outputs
    prov = json.loads(outputs.provenance_path.read_text(encoding="utf-8"))
    assert "config" in prov
    assert "outputs" in prov
    assert isinstance(prov["outputs"], list)
    assert len(prov["outputs"]) == len(outputs.saved_paths)


def test_standard_pipeline_smoke_zapline_optimized(
    synthetic_raw: mne.io.Raw,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Zapline-optimized should run without errors and still save outputs.

    This test mostly exercises:
    - order resolution logic
    - optional pre-zapline highpass branch
    - zapline placement relative to resample
    """
    from dcap.seeg.preprocessing.pipelines import standard as standard_mod

    monkeypatch.setattr(standard_mod, "resample_raw", lambda raw, cfg, ctx: _identity_step(raw, name="resample"))
    monkeypatch.setattr(standard_mod, "remove_line_noise_view", lambda raw, cfg, ctx: _identity_step(raw, name="line_noise"))
    monkeypatch.setattr(standard_mod, "highpass_view", lambda raw, cfg, ctx: _identity_step(raw, name="highpass"))
    monkeypatch.setattr(standard_mod, "gamma_envelope_view", lambda raw, cfg, ctx: _identity_step(raw, name="gamma_envelope"))
    monkeypatch.setattr(standard_mod, "rereference_view", lambda raw, cfg, ctx: _fake_reref_step(raw))

    cfg_dict = _base_cfg_dict()
    cfg_dict["pipeline"]["profile"] = "zapline_optimized"
    cfg_dict["pipeline"]["zapline_at_native_sfreq"] = True
    cfg_dict["pipeline"]["pre_zapline_highpass_hz"] = 1.0
    cfg_dict["line_noise"]["method"] = "zapline"

    cfg = StandardPipelineConfig(raw=cfg_dict)
    outputs = run_preprocess_single_raw(
        raw=synthetic_raw,
        cfg=cfg,
        out_dir=out_dir,
        base_stem="sub-TEST_task-conversation_run-2",
    )

    assert outputs.provenance_path.exists()
    assert any(p.name.endswith("_raw.fif") or p.name.endswith("_desc-car_raw.fif") for p in outputs.saved_paths)


def test_standard_pipeline_high_gamma_envelope_mode(
    synthetic_raw: mne.io.Raw,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    HGA envelope mode should run and produce outputs.
    """
    from dcap.seeg.preprocessing.pipelines import standard as standard_mod

    monkeypatch.setattr(standard_mod, "resample_raw", lambda raw, cfg, ctx: _identity_step(raw, name="resample"))
    monkeypatch.setattr(standard_mod, "remove_line_noise_view", lambda raw, cfg, ctx: _identity_step(raw, name="line_noise"))
    monkeypatch.setattr(standard_mod, "highpass_view", lambda raw, cfg, ctx: _identity_step(raw, name="highpass"))
    monkeypatch.setattr(standard_mod, "gamma_envelope_view", lambda raw, cfg, ctx: _identity_step(raw, name="gamma_envelope"))
    monkeypatch.setattr(standard_mod, "rereference_view", lambda raw, cfg, ctx: _fake_reref_step(raw))

    cfg_dict = _base_cfg_dict()
    cfg_dict["filtering"]["mode"] = "high_gamma_envelope"
    cfg_dict["filtering"]["high_gamma_envelope"]["enabled"] = True

    cfg = StandardPipelineConfig(raw=cfg_dict)
    outputs = run_preprocess_single_raw(
        raw=synthetic_raw,
        cfg=cfg,
        out_dir=out_dir,
        base_stem="sub-TEST_task-conversation_run-3",
    )

    assert outputs.provenance_path.exists()
    assert len(outputs.saved_paths) >= 1


def test_standard_pipeline_respects_save_all_views_false(
    synthetic_raw: mne.io.Raw,
    out_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When save_all_views=false, only default_view should be saved (plus original fallback handling).
    """
    from dcap.seeg.preprocessing.pipelines import standard as standard_mod

    monkeypatch.setattr(standard_mod, "resample_raw", lambda raw, cfg, ctx: _identity_step(raw, name="resample"))
    monkeypatch.setattr(standard_mod, "remove_line_noise_view", lambda raw, cfg, ctx: _identity_step(raw, name="line_noise"))
    monkeypatch.setattr(standard_mod, "highpass_view", lambda raw, cfg, ctx: _identity_step(raw, name="filtering"))
    monkeypatch.setattr(standard_mod, "gamma_envelope_view", lambda raw, cfg, ctx: _identity_step(raw, name="gamma_envelope"))
    monkeypatch.setattr(standard_mod, "rereference_view", lambda raw, cfg, ctx: _fake_reref_step(raw))

    cfg_dict = _base_cfg_dict()
    cfg_dict["rereference"]["save_all_views"] = False
    cfg_dict["rereference"]["default_view"] = "car"

    cfg = StandardPipelineConfig(raw=cfg_dict)
    outputs = run_preprocess_single_raw(
        raw=synthetic_raw,
        cfg=cfg,
        out_dir=out_dir,
        base_stem="sub-TEST_task-conversation_run-4",
    )

    assert outputs.provenance_path.exists()
    # With our fake reref, pipeline should save exactly one view
    assert len(outputs.saved_paths) == 1
