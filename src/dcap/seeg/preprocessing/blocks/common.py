
# =============================================================================
#                             PREPROCESSING BLOCKS
# =============================================================================
#
# Reusable signal-processing blocks (skeleton).
#
# Each block:
# - takes (raw, cfg, ctx)
# - returns (raw_out, artifact)
# - appends to ctx.proc_history
# - performs no file I/O
#
# NOTE: This file intentionally contains placeholder implementations. It is a
# stable API surface for tests and future logic.
#
# =============================================================================

from dataclasses import asdict
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from dcap.seeg.preprocessing.configs import (
    BadChannelsConfig,
    CoordinatesConfig,
    GammaEnvelopeConfig,
    HighpassConfig,
    LineNoiseConfig,
    ResampleConfig,
    RereferenceConfig,
)
from dcap.seeg.preprocessing.raw_types import BaseRaw
from dcap.seeg.preprocessing.types import BlockArtifact, Geometry, PreprocContext


# =============================================================================
#                               BLOCK 2: COORDS
# =============================================================================
def attach_coordinates(
    raw: BaseRaw,
    electrodes_table: Mapping[str, Sequence[float]],
    cfg: CoordinatesConfig,
    ctx: PreprocContext,
) -> Tuple[BaseRaw, BlockArtifact]:
    """
    Attach electrode coordinates to channels and enrich context geometry.

    Parameters
    ----------
    raw
        Raw recording.
    electrodes_table
        Mapping from channel name to (x, y, z) coordinates.
        Unit is declared by `cfg.unit`.
    cfg
        Coordinates block configuration.
    ctx
        Preprocessing context.

    Returns
    -------
    raw_out
        Raw (currently passthrough).
    artifact
        Block artifact.

    Usage example
    -------------
        raw_out, artifact = attach_coordinates(
            raw=raw,
            electrodes_table={"A1": (0.0, 0.0, 0.0)},
            cfg=CoordinatesConfig(unit="mm"),
            ctx=ctx,
        )
    """
    ctx.add_record("coordinates", asdict(cfg))

    ctx.geometry = Geometry(coords_m=dict(electrodes_table))
    artifact = BlockArtifact(
        name="coordinates",
        parameters=asdict(cfg),
        summary_metrics={"n_coords": len(electrodes_table)},
        warnings=["coordinates attachment not implemented; geometry stored in ctx only"],
        figures=[],
    )
    return raw, artifact


# =============================================================================
#                           BLOCK 3: LINE NOISE
# =============================================================================
def remove_line_noise(
    raw: BaseRaw,
    cfg: LineNoiseConfig,
    ctx: PreprocContext,
) -> Tuple[BaseRaw, BlockArtifact]:
    """
    Remove line noise using notch or zapline (meegkit).

    Returns a passthrough Raw in this skeleton.

    Usage example
    -------------
        raw_out, artifact = remove_line_noise(raw, LineNoiseConfig(method="notch"), ctx)
    """
    ctx.add_record("line_noise", asdict(cfg))

    artifact = BlockArtifact(
        name="line_noise",
        parameters=asdict(cfg),
        summary_metrics={},
        warnings=[f"{cfg.method} not implemented; passthrough"],
        figures=[],
    )
    return raw, artifact


# =============================================================================
#                          BLOCK 4A: HIGH-PASS
# =============================================================================
def highpass_filter(
    raw: BaseRaw,
    cfg: HighpassConfig,
    ctx: PreprocContext,
) -> Tuple[BaseRaw, BlockArtifact]:
    """
    High-pass filter for drift removal (skeleton passthrough).

    Usage example
    -------------
        raw_out, artifact = highpass_filter(raw, HighpassConfig(l_freq=0.5), ctx)
    """
    ctx.add_record("highpass", asdict(cfg))

    artifact = BlockArtifact(
        name="highpass",
        parameters=asdict(cfg),
        summary_metrics={},
        warnings=["high-pass filtering not implemented; passthrough"],
        figures=[],
    )
    return raw, artifact


# =============================================================================
#                      BLOCK 4B: GAMMA ENVELOPE PATH
# =============================================================================
def compute_gamma_envelope(
    raw: BaseRaw,
    cfg: GammaEnvelopeConfig,
    ctx: PreprocContext,
) -> Tuple[BaseRaw, BlockArtifact]:
    """
    Compute gamma/HFA envelope time series (skeleton passthrough).

    In v1, this will likely return a derived Raw whose channels represent
    envelope values.

    Usage example
    -------------
        env_raw, artifact = compute_gamma_envelope(raw, GammaEnvelopeConfig(), ctx)
    """
    ctx.add_record("gamma_envelope", asdict(cfg))

    artifact = BlockArtifact(
        name="gamma_envelope",
        parameters=asdict(cfg),
        summary_metrics={},
        warnings=["gamma envelope not implemented; passthrough"],
        figures=[],
    )
    return raw, artifact


# =============================================================================
#                           BLOCK 5: RESAMPLE
# =============================================================================
def resample_raw(
    raw: BaseRaw,
    cfg: ResampleConfig,
    ctx: PreprocContext,
) -> Tuple[BaseRaw, BlockArtifact]:
    """
    Resample the recording to a target sampling rate (skeleton passthrough).

    Usage example
    -------------
        raw_out, artifact = resample_raw(raw, ResampleConfig(sfreq_out=512.0), ctx)
    """
    ctx.add_record("resample", asdict(cfg))

    artifact = BlockArtifact(
        name="resample",
        parameters=asdict(cfg),
        summary_metrics={},
        warnings=["resampling not implemented; passthrough"],
        figures=[],
    )
    return raw, artifact


# =============================================================================
#                      BLOCK 6: BAD CHANNEL SUGGESTION
# =============================================================================
def suggest_bad_channels(
    raw: BaseRaw,
    cfg: BadChannelsConfig,
    ctx: PreprocContext,
) -> Tuple[BaseRaw, BlockArtifact]:
    """
    Suggest bad channels with human-readable reasons (skeleton).

    In v1, this will compute metrics and store:
    - ctx.decisions["suggested_bad_channels"]
    - ctx.decisions["bad_channel_reasons"]

    Usage example
    -------------
        raw_out, artifact = suggest_bad_channels(raw, BadChannelsConfig(), ctx)
    """
    ctx.add_record("bad_channels", asdict(cfg))

    ctx.decisions.setdefault("suggested_bad_channels", [])
    ctx.decisions.setdefault("bad_channel_reasons", {})

    artifact = BlockArtifact(
        name="bad_channels",
        parameters=asdict(cfg),
        summary_metrics={"n_suggested": 0},
        warnings=["bad channel detection not implemented; no suggestions"],
        figures=[],
    )
    return raw, artifact


# =============================================================================
#                         BLOCK 7: REREFERENCING
# =============================================================================
def rereference(
    raw: BaseRaw,
    cfg: RereferenceConfig,
    ctx: PreprocContext,
) -> Tuple[Dict[str, BaseRaw], BlockArtifact]:
    """
    Generate one or more rereferenced views (CAR, bipolar, Laplacian, etc.).

    Returns
    -------
    views
        Mapping from view name to Raw object. In skeleton, returns {"original": raw}.
    artifact
        Block artifact.

    Usage example
    -------------
        views, artifact = rereference(raw, RereferenceConfig(methods=("car",)), ctx)
        raw_car = views.get("car")
    """
    ctx.add_record("rereference", asdict(cfg))

    views: Dict[str, BaseRaw] = {"original": raw}

    artifact = BlockArtifact(
        name="rereference",
        parameters=asdict(cfg),
        summary_metrics={"views": list(views.keys())},
        warnings=["rereferencing not implemented; only 'original' returned"],
        figures=[],
    )
    return views, artifact
