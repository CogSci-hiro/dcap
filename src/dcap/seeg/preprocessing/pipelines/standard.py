# =============================================================================
# =============================================================================
#                     ########################################
#                     #     STANDARD (RESEARCH) PIPELINE      #
#                     ########################################
# =============================================================================
# =============================================================================
"""
Config-driven, research-friendly sEEG preprocessing pipeline.

This pipeline is intended as the reusable counterpart to:
    dcap.seeg.preprocessing.pipelines.clinical

Core design
-----------
- Orchestrates existing "block" functions (core logic lives elsewhere)
- Supports two step orders:
    1) "canonical"
    2) "zapline_optimized"
- Produces one or multiple rereference "views"
- Saves derivatives + a provenance JSON bundle (config + artifacts + ctx snapshot)

Key assumptions
---------------
- You already have core step implementations available in dcap:
    - dcap.seeg.preprocessing.block.resample.resample_raw
    - dcap.seeg.preprocessing.block.line_noise.remove_line_noise_view
    - dcap.seeg.preprocessing.block.filtering.highpass_view
    - dcap.seeg.preprocessing.block.filtering.gamma_envelope_view
    - dcap.seeg.preprocessing.block.rereference.rereference_view

- Config dataclasses exist in:
    dcap.seeg.preprocessing.configs
  (ResampleConfig, LineNoiseConfig, HighpassConfig, GammaEnvelopeConfig, RereferenceConfig)

- Types exist in:
    dcap.seeg.preprocessing.types
  (PreprocContext, BlockArtifact)

Usage example
-------------
    from pathlib import Path
    import mne

    from dcap.seeg.preprocessing.pipelines.standard import (
        load_preprocess_yaml,
        run_preprocess_single_raw,
    )

    cfg = load_preprocess_yaml(Path("dcap_preprocess.yaml"))

    raw = mne.io.read_raw_fif("sub-001_task-conversation_run-1_raw.fif", preload=True)

    outputs = run_preprocess_single_raw(
        raw=raw,
        cfg=cfg,
        out_dir=Path("derivatives/preproc/sub-001"),
        base_stem="sub-001_task-conversation_run-1",
    )
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import mne

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError("YAML support requires PyYAML (pip install pyyaml).") from exc

from dcap.seeg.preprocessing.configs import (
    GammaEnvelopeConfig,
    HighpassConfig,
    LineNoiseConfig,
    RereferenceConfig,
    ResampleConfig,
)
from dcap.seeg.preprocessing.types import BlockArtifact, PreprocContext

from dcap.seeg.preprocessing.blocks.resample import resample_raw
from dcap.seeg.preprocessing.blocks.line_noise import remove_line_noise_view
from dcap.seeg.preprocessing.blocks.filtering import gamma_envelope_view, highpass_view
from dcap.seeg.preprocessing.blocks.rereference import rereference_view


PipelineProfile = Literal["canonical", "zapline_optimized"]
FilteringMode = Literal["broadband", "high_gamma_envelope"]


# =============================================================================
#                               YAML config model
# =============================================================================
@dataclass(frozen=True)
class StandardPipelineConfig:
    """
    Parsed YAML configuration for the standard pipeline.

    This is intentionally a *thin* schema: it stores the YAML dict and provides
    typed accessors for the pipeline runner. You can later replace this with a
    stricter dataclass hierarchy if desired.

    Attributes
    ----------
    raw
        The raw YAML dictionary (normalized).
    """

    raw: Dict[str, Any]

    @property
    def profile(self) -> PipelineProfile:
        return str(self.raw.get("pipeline", {}).get("profile", "canonical"))

    @property
    def filtering_mode(self) -> FilteringMode:
        return str(self.raw.get("filtering", {}).get("mode", "broadband"))

    @property
    def stop_after(self) -> Optional[str]:
        value = self.raw.get("pipeline", {}).get("stop_after", None)
        return None if value in (None, "null") else str(value)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.raw)


def load_preprocess_yaml(path: Path) -> StandardPipelineConfig:
    """
    Load preprocessing config YAML.

    Parameters
    ----------
    path
        Path to YAML file.

    Returns
    -------
    cfg
        Parsed StandardPipelineConfig.

    Usage example
    -------------
        cfg = load_preprocess_yaml(Path("dcap_preprocess.yaml"))
    """
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected YAML top-level to be a mapping/dict.")

    return StandardPipelineConfig(raw=_normalize_yaml_dict(data))


def _normalize_yaml_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize common YAML pitfalls (None/strings) without being overly clever.
    """
    # For now, shallow-copy only; deeper normalization can be added later.
    return dict(data)


# =============================================================================
#                               Public pipeline API
# =============================================================================
@dataclass(frozen=True)
class PreprocessOutputs:
    """
    Outputs of a preprocessing run.

    Attributes
    ----------
    saved_paths
        Paths of saved derivative files (one per view).
    artifacts
        BlockArtifacts returned by step wrappers (one per executed step).
    provenance_path
        Path to JSON provenance bundle.
    """

    saved_paths: List[Path]
    artifacts: List[BlockArtifact]
    provenance_path: Path


def run_preprocess_single_raw(
    *,
    raw: mne.io.BaseRaw,
    cfg: StandardPipelineConfig,
    out_dir: Path,
    base_stem: str,
    ctx: Optional[PreprocContext] = None,
) -> PreprocessOutputs:
    """
    Run the standard pipeline on a single already-loaded Raw.

    Parameters
    ----------
    raw
        Input MNE Raw.
    cfg
        Parsed standard pipeline config.
    out_dir
        Output directory for derivatives and provenance JSON.
    base_stem
        Base filename stem used for outputs (e.g., "sub-001_task-conversation_run-1").
    ctx
        Optional PreprocContext. If None, a new context is created.

    Returns
    -------
    outputs
        Saved paths + artifacts + provenance JSON path.

    Usage example
    -------------
        outputs = run_preprocess_single_raw(
            raw=raw,
            cfg=cfg,
            out_dir=Path("derivatives/preproc/sub-001"),
            base_stem="sub-001_task-conversation_run-1",
        )
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw must be an mne.io.BaseRaw.")
    if ctx is None:
        ctx = PreprocContext()  # type: ignore[call-arg]

    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: List[BlockArtifact] = []

    # Decide which step order to execute
    steps = _resolve_step_order(cfg)

    raw_working = raw
    views: Dict[str, mne.io.BaseRaw] = {"original": raw_working}

    for step_name in steps:
        if step_name == "pre_zapline_highpass":
            raw_working, artifact = _run_pre_zapline_highpass(raw_working, cfg=cfg, ctx=ctx)
            artifacts.append(artifact)
            views = {"original": raw_working}

        elif step_name == "resample":
            raw_working, artifact = _run_resample(raw_working, cfg=cfg, ctx=ctx)
            artifacts.append(artifact)
            views = {"original": raw_working}

        elif step_name == "line_noise":
            raw_working, artifact = _run_line_noise(raw_working, cfg=cfg, ctx=ctx)
            artifacts.append(artifact)
            views = {"original": raw_working}

        elif step_name == "filtering":
            raw_working, artifact = _run_filtering(raw_working, cfg=cfg, ctx=ctx)
            artifacts.append(artifact)
            views = {"original": raw_working}

        elif step_name == "rereference":
            views, artifact = _run_rereference(raw_working, cfg=cfg, ctx=ctx)
            artifacts.append(artifact)

        else:
            raise ValueError(f"Unknown step in resolved order: {step_name!r}")

        # Optional early stop for debugging
        if cfg.stop_after is not None and step_name == cfg.stop_after:
            break

    saved_paths = _save_views(
        views=views,
        cfg=cfg,
        out_dir=out_dir,
        base_stem=base_stem,
    )

    provenance_path = _write_provenance_bundle(
        out_dir=out_dir,
        base_stem=base_stem,
        cfg=cfg,
        ctx=ctx,
        artifacts=artifacts,
        saved_paths=saved_paths,
    )

    return PreprocessOutputs(saved_paths=saved_paths, artifacts=artifacts, provenance_path=provenance_path)


# =============================================================================
#                               Step order resolver
# =============================================================================
def _resolve_step_order(cfg: StandardPipelineConfig) -> List[str]:
    """
    Resolve step order based on profile + zapline switches.

    Returns a list of step names from:
        pre_zapline_highpass, resample, line_noise, filtering, rereference
    """
    profile = cfg.profile
    line_noise_method = str(cfg.raw.get("line_noise", {}).get("method", "notch"))

    if profile == "canonical":
        return ["resample", "line_noise", "filtering", "rereference"]

    if profile == "zapline_optimized":
        # If zapline isn't requested, this profile is equivalent to canonical.
        if line_noise_method != "zapline":
            return ["resample", "line_noise", "filtering", "rereference"]

        pre_hp = cfg.raw.get("pipeline", {}).get("pre_zapline_highpass_hz", None)
        zapline_at_native = bool(cfg.raw.get("pipeline", {}).get("zapline_at_native_sfreq", True))

        order: List[str] = []
        if pre_hp not in (None, "null"):
            order.append("pre_zapline_highpass")

        # If zapline_at_native_sfreq: zapline before resample
        if zapline_at_native:
            order.extend(["line_noise", "resample", "filtering", "rereference"])
        else:
            order.extend(["resample", "line_noise", "filtering", "rereference"])

        return order

    raise ValueError(f"Unknown pipeline profile: {profile!r}")


# =============================================================================
#                               Individual step runners
# =============================================================================
def _run_resample(raw: mne.io.BaseRaw, *, cfg: StandardPipelineConfig, ctx: PreprocContext) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    section = cfg.raw.get("resample", {})
    if not bool(section.get("enabled", True)):
        artifact = BlockArtifact(
            name="resample",
            parameters={"enabled": False},
            summary_metrics={"skipped": 1.0},
            warnings=["Resample disabled in config; step skipped."],
            figures=[],
        )
        return raw, artifact

    sfreq_out = float(section.get("sfreq_out_hz", section.get("sfreq_out", raw.info["sfreq"])))
    res_cfg = ResampleConfig(sfreq_out=sfreq_out)  # type: ignore[call-arg]
    return resample_raw(raw, cfg=res_cfg, ctx=ctx)


def _run_line_noise(raw: mne.io.BaseRaw, *, cfg: StandardPipelineConfig, ctx: PreprocContext) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    section = cfg.raw.get("line_noise", {})
    if not bool(section.get("enabled", True)):
        artifact = BlockArtifact(
            name="line_noise",
            parameters={"enabled": False},
            summary_metrics={"skipped": 1.0},
            warnings=["Line-noise removal disabled in config; step skipped."],
            figures=[],
        )
        return raw, artifact

    method = str(section.get("method", "notch"))
    freq_base = float(section.get("base_freq_hz", section.get("freq_base", 50.0)))
    max_harmonic_hz = float(section.get("max_harmonic_hz", 250.0))

    zap = section.get("zapline", {}) if isinstance(section.get("zapline", {}), dict) else {}
    chunk_sec = float(zap.get("chunk_sec", 60.0))
    nremove = int(zap.get("nremove", 1))

    # Config object used by your wrapper
    ln_cfg = LineNoiseConfig(
        method=method,
        freq_base=freq_base,
        max_harmonic_hz=max_harmonic_hz,
        picks=None,
        chunk_sec=chunk_sec,
        nremove=nremove,
    )  # type: ignore[call-arg]

    return remove_line_noise_view(raw, cfg=ln_cfg, ctx=ctx)


def _run_pre_zapline_highpass(raw: mne.io.BaseRaw, *, cfg: StandardPipelineConfig, ctx: PreprocContext) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    """
    Optional light highpass *only* used in zapline_optimized profile.
    """
    l_freq = cfg.raw.get("pipeline", {}).get("pre_zapline_highpass_hz", None)
    if l_freq in (None, "null"):
        artifact = BlockArtifact(
            name="pre_zapline_highpass",
            parameters={"enabled": False},
            summary_metrics={"skipped": 1.0},
            warnings=["pre_zapline_highpass_hz is null; step skipped."],
            figures=[],
        )
        return raw, artifact

    hp_cfg = HighpassConfig(l_freq=float(l_freq), phase="zero")  # type: ignore[call-arg]
    raw_out, hp_artifact = highpass_view(raw, cfg=hp_cfg, ctx=ctx)

    artifact = BlockArtifact(
        name="pre_zapline_highpass",
        parameters={"l_freq_hz": float(l_freq), "phase": "zero"},
        summary_metrics=hp_artifact.summary_metrics,
        warnings=["Pre-zapline highpass applied (zapline_optimized profile)."],
        figures=[],
    )
    return raw_out, artifact


def _run_filtering(raw: mne.io.BaseRaw, *, cfg: StandardPipelineConfig, ctx: PreprocContext) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    section = cfg.raw.get("filtering", {})
    if not bool(section.get("enabled", True)):
        artifact = BlockArtifact(
            name="filtering",
            parameters={"enabled": False},
            summary_metrics={"skipped": 1.0},
            warnings=["Filtering disabled in config; step skipped."],
            figures=[],
        )
        return raw, artifact

    mode = str(section.get("mode", "broadband"))
    if mode == "broadband":
        bb = section.get("broadband", {}) if isinstance(section.get("broadband", {}), dict) else {}
        hp = bb.get("highpass", {}) if isinstance(bb.get("highpass", {}), dict) else {}
        if bool(hp.get("enabled", True)):
            hp_cfg = HighpassConfig(
                l_freq=float(hp.get("l_freq_hz", hp.get("l_freq", 1.0))),
                phase=str(hp.get("phase", "zero")),
            )  # type: ignore[call-arg]
            return highpass_view(raw, cfg=hp_cfg, ctx=ctx)

        artifact = BlockArtifact(
            name="filtering",
            parameters={"mode": "broadband", "highpass_enabled": False},
            summary_metrics={"skipped": 1.0},
            warnings=["Broadband filtering configured but highpass disabled; step skipped."],
            figures=[],
        )
        return raw, artifact

    if mode == "high_gamma_envelope":
        hga = section.get("high_gamma_envelope", {}) if isinstance(section.get("high_gamma_envelope", {}), dict) else {}
        if not bool(hga.get("enabled", True)):
            artifact = BlockArtifact(
                name="gamma_envelope",
                parameters={"enabled": False},
                summary_metrics={"skipped": 1.0},
                warnings=["high_gamma_envelope mode selected but enabled=false; step skipped."],
                figures=[],
            )
            return raw, artifact

        bandpass = hga.get("bandpass", {}) if isinstance(hga.get("bandpass", {}), dict) else {}
        l_freq = float(bandpass.get("l_freq_hz", 70.0))
        h_freq = float(bandpass.get("h_freq_hz", 150.0))

        smooth = hga.get("smooth", {}) if isinstance(hga.get("smooth", {}), dict) else {}
        smooth_enabled = bool(smooth.get("enabled", True))
        smoothing_sec = 0.0
        if smooth_enabled:
            # Prefer lowpass if your wrapper supports it later; for now map to moving-average sec.
            window_ms = float(smooth.get("window_ms", 50.0))
            smoothing_sec = float(window_ms) / 1000.0

        gamma_cfg = GammaEnvelopeConfig(
            band_hz=(l_freq, h_freq),
            method="hilbert",
            smoothing_sec=float(smoothing_sec),
        )  # type: ignore[call-arg]

        env_raw, artifact = gamma_envelope_view(raw, cfg=gamma_cfg, ctx=ctx)

        # Optional envelope downsample (recommended for TRF)
        down = hga.get("downsample", {}) if isinstance(hga.get("downsample", {}), dict) else {}
        if bool(down.get("enabled", False)):
            sfreq_out = float(down.get("sfreq_out_hz", 128.0))
            env_raw, res_artifact = resample_raw(env_raw, cfg=ResampleConfig(sfreq_out=sfreq_out), ctx=ctx)  # type: ignore[call-arg]
            # Merge artifacts into one (keep both visible)
            merged = BlockArtifact(
                name="gamma_envelope(+downsample)",
                parameters={"gamma": artifact.parameters, "downsample": res_artifact.parameters},
                summary_metrics={**artifact.summary_metrics, **res_artifact.summary_metrics},
                warnings=artifact.warnings + res_artifact.warnings,  # noqa
                figures=[],
            )
            return env_raw, merged

        return env_raw, artifact

    raise ValueError(f"Unknown filtering mode: {mode!r}")


def _run_rereference(raw: mne.io.BaseRaw, *, cfg: StandardPipelineConfig, ctx: PreprocContext) -> Tuple[Dict[str, mne.io.BaseRaw], BlockArtifact]:
    section = cfg.raw.get("rereference", {})
    if not bool(section.get("enabled", True)):
        artifact = BlockArtifact(
            name="rereference",
            parameters={"enabled": False},
            summary_metrics={"skipped": 1.0},
            warnings=["Rereference disabled in config; returning original only."],
            figures=[],
        )
        return {"original": raw}, artifact

    views_section = section.get("views", [])
    save_all = bool(section.get("save_all_views", True))
    default_view_name = str(section.get("default_view", "car"))

    methods: List[str] = []
    for view in views_section:
        if not isinstance(view, dict):
            continue
        methods.append(str(view.get("method", "")))
    methods = [m for m in methods if m]

    if not methods:
        # Safe fallback
        methods = ["car"]

    reref_cfg = RereferenceConfig(
        methods=methods,
        car_scope=str(_first_view_param(views_section, key="scope", default="global")),
        laplacian_mode=str(_first_view_param(views_section, key="mode", default="shaft_1d")),
    )  # type: ignore[call-arg]

    views, artifact = rereference_view(raw, cfg=reref_cfg, ctx=ctx)

    if not save_all:
        # Reduce to one chosen view, fall back to original if missing.
        chosen = views.get(default_view_name, None)
        if chosen is None:
            chosen = views.get("original", raw)
            artifact = BlockArtifact(
                name=artifact.name,
                parameters=artifact.parameters,
                summary_metrics=artifact.summary_metrics,
                warnings=artifact.warnings + [f"default_view={default_view_name!r} missing; fell back to 'original'."],  # noqa
                figures=[],
            )
        return {default_view_name: chosen}, artifact

    return views, artifact


def _first_view_param(views_section: Any, *, key: str, default: Any) -> Any:
    if not isinstance(views_section, list):
        return default
    for item in views_section:
        if not isinstance(item, dict):
            continue
        params = item.get("params", None)
        if isinstance(params, dict) and key in params:
            return params[key]
    return default


# =============================================================================
#                               Saving + provenance
# =============================================================================
def _save_views(
    *,
    views: Mapping[str, mne.io.BaseRaw],
    cfg: StandardPipelineConfig,
    out_dir: Path,
    base_stem: str,
) -> List[Path]:
    """
    Save each view as a FIF file. This keeps saving boring and reliable.

    Naming
    ------
    {base_stem}_desc-{view}_raw.fif

    If view is "original", desc is omitted:
    {base_stem}_raw.fif
    """
    io_cfg = cfg.raw.get("io", {})
    overwrite = bool(io_cfg.get("overwrite", False))

    saved: List[Path] = []
    for view_name, raw in views.items():
        desc = "" if view_name in ("original", "", None) else f"_desc-{view_name}"
        fname = f"{base_stem}{desc}_raw.fif"
        path = out_dir / fname

        raw.save(path, overwrite=overwrite, verbose=False)
        saved.append(path)

    return saved


def _write_provenance_bundle(
    *,
    out_dir: Path,
    base_stem: str,
    cfg: StandardPipelineConfig,
    ctx: PreprocContext,
    artifacts: Sequence[BlockArtifact],
    saved_paths: Sequence[Path],
) -> Path:
    """
    Write a single JSON bundle capturing:
    - config (as loaded)
    - ctx snapshot (best-effort)
    - artifacts (best-effort)
    - outputs list
    """
    now = datetime.now(timezone.utc).isoformat()

    bundle: Dict[str, Any] = {
        "created_utc": now,
        "base_stem": base_stem,
        "config": cfg.to_dict(),
        "outputs": [str(p) for p in saved_paths],
        "ctx": _ctx_snapshot(ctx),
        "artifacts": [_artifact_to_dict(a) for a in artifacts],
    }

    path = out_dir / f"{base_stem}_provenance.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, sort_keys=True)

    return path


def _ctx_snapshot(ctx: PreprocContext) -> Dict[str, Any]:
    """
    Best-effort serialization of PreprocContext without assuming its internals.
    """
    out: Dict[str, Any] = {}
    for attr in ("records", "decisions"):
        if hasattr(ctx, attr):
            try:
                out[attr] = getattr(ctx, attr)
            except Exception:  # noqa
                out[attr] = "<unserializable>"
    # geometry might be complex; store a minimal hint
    if hasattr(ctx, "geometry"):
        try:
            geo = getattr(ctx, "geometry")
            out["geometry_present"] = geo is not None
        except Exception:  # noqa
            out["geometry_present"] = "<unknown>"
    return out


def _artifact_to_dict(artifact: BlockArtifact) -> Dict[str, Any]:
    """
    Best-effort conversion of BlockArtifact to dict.
    """
    out: Dict[str, Any] = {}
    for attr in ("name", "parameters", "summary_metrics", "warnings", "figures"):
        if hasattr(artifact, attr):
            try:
                out[attr] = getattr(artifact, attr)
            except Exception:  # noqa
                out[attr] = "<unserializable>"
    return out
