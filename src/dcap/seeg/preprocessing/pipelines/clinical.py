# =============================================================================
#                   ############################################
#                   #   PIPELINE: CLINICAL COMMON PREPROCESS   #
#                   ############################################
# =============================================================================
#
# Logic-only orchestration of reusable blocks for a "clinical-style" baseline.
#
# This module:
# - does not read or write any files
# - does not parse CLI arguments
# - does not print
#
# The CLI layer should:
# - load Raw + electrode metadata from disk
# - call `run_clinical_preproc`
# - write outputs + build reports
#
# =============================================================================

from dataclasses import asdict, dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

from dcap.seeg.preprocessing.blocks import (
    attach_coordinates,
    compute_gamma_envelope,
    highpass_filter,
    rereference,
    remove_line_noise,
    resample_raw,
    suggest_bad_channels,
)
from dcap.seeg.preprocessing.configs import ClinicalPreprocConfig, CoordinatesConfig
from dcap.seeg.preprocessing.types import BlockArtifact, PreprocContext


@dataclass(frozen=True)
class ClinicalPreprocResult:
    """
    Return object for the clinical preprocessing pipeline.

    Attributes
    ----------
    views
        Mapping of view name -> Raw (e.g., "original", "car", "bipolar", "laplacian").
    artifacts
        Ordered list of block artifacts (one per executed step).
    ctx
        Context containing provenance + decisions.

    Usage example
    -------------
        result = run_clinical_preproc(raw, electrodes_table, cfg, coords_cfg)
        views = result.views
        artifacts = result.artifacts
    """

    views: Mapping[str, "mne.io.BaseRaw"]
    artifacts: Sequence[BlockArtifact]
    ctx: PreprocContext


def run_clinical_preproc(
    raw: "mne.io.BaseRaw",
    electrodes_table: Optional[Mapping[str, Sequence[float]]],
    cfg: ClinicalPreprocConfig,
    coords_cfg: Optional[CoordinatesConfig] = None,
    ctx: Optional[PreprocContext] = None,
) -> ClinicalPreprocResult:
    """
    Run the clinical-style common preprocessing pipeline (logic-only).

    Parameters
    ----------
    raw
        MNE Raw object.
    electrodes_table
        Optional mapping channel -> (x,y,z) coordinates. If provided, coordinates are attached.
    cfg
        Pipeline configuration.
    coords_cfg
        Coordinates block configuration (required if electrodes_table is provided).
    ctx
        Optional existing context; if None, a fresh context is created.

    Returns
    -------
    result
        Pipeline result with views + artifacts + context.

    Usage example
    -------------
        cfg = ClinicalPreprocConfig()
        result = run_clinical_preproc(
            raw=raw,
            electrodes_table={"A1": (12.3, -4.5, 6.7)},
            cfg=cfg,
            coords_cfg=CoordinatesConfig(unit="mm"),
        )
    """
    import mne
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("run_clinical_preproc expects an mne.io.BaseRaw.")

    if ctx is None:
        ctx = PreprocContext()

    artifacts: List[BlockArtifact] = []

    ctx.add_record("pipeline_clinical_preproc", asdict(cfg))

    if electrodes_table is not None:
        if coords_cfg is None:
            raise ValueError("coords_cfg must be provided when electrodes_table is provided.")
        raw, artifact = attach_coordinates(raw, electrodes_table, coords_cfg, ctx)
        artifacts.append(artifact)

    raw, artifact = remove_line_noise(raw, cfg.line_noise, ctx)
    artifacts.append(artifact)

    raw, artifact = highpass_filter(raw, cfg.highpass, ctx)
    artifacts.append(artifact)

    if cfg.gamma_envelope is not None:
        raw, artifact = compute_gamma_envelope(raw, cfg.gamma_envelope, ctx)
        artifacts.append(artifact)

    if cfg.resample is not None:
        raw, artifact = resample_raw(raw, cfg.resample, ctx)
        artifacts.append(artifact)

    raw, artifact = suggest_bad_channels(raw, cfg.bad_channels, ctx)
    artifacts.append(artifact)

    views, artifact = rereference(raw, cfg.rereference, ctx)
    artifacts.append(artifact)

    return ClinicalPreprocResult(views=views, artifacts=artifacts, ctx=ctx)
