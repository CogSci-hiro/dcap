# =============================================================================
# =============================================================================
#                     ########################################
#                     #   PIPELINE: CLINICAL PREPROCESSING   #
#                     ########################################
# =============================================================================
# =============================================================================

from typing import Mapping, Optional, Sequence

import mne

from dcap.seeg.preprocessing.blocks.coordinates import attach_coordinates
from dcap.seeg.preprocessing.blocks.filtering import highpass_filter
from dcap.seeg.preprocessing.blocks.line_noise import remove_line_noise
from dcap.seeg.preprocessing.blocks.resample import resample_raw
from dcap.seeg.preprocessing.blocks.rereference import rereference
from dcap.seeg.preprocessing.configs import ClinicalPreprocConfig, CoordinatesConfig
from dcap.seeg.preprocessing.types import BlockArtifact, PreprocContext, PreprocResult


def run_clinical_preproc(
    *,
    raw: mne.io.BaseRaw,
    cfg: ClinicalPreprocConfig,
    electrodes_table: Optional[Mapping[str, Sequence[float]]] = None,
    coords_cfg: Optional[CoordinatesConfig] = None,
    ctx: Optional[PreprocContext] = None,
) -> PreprocResult:
    """
    Run the standard clinical preprocessing pipeline (logic-only).

    Parameters
    ----------
    raw
        Input Raw.
    cfg
        Clinical preprocessing configuration bundle.
    electrodes_table
        Optional mapping contact -> (x, y, z) in cfg.coords.unit.
    coords_cfg
        Optional override for coordinate config; if None, uses cfg.coords.
    ctx
        Optional existing context.

    Returns
    -------
    result
        PreprocResult with views (includes "original"), artifacts, and context.

    Usage example
    -------------
        result = run_clinical_preproc(raw=raw, cfg=ClinicalPreprocConfig())
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("run_clinical_preproc expects an mne.io.BaseRaw.")

    ctx_final = ctx if ctx is not None else PreprocContext()
    artifacts: list[BlockArtifact] = []

    current_raw = raw.copy()
    views = {"original": current_raw}

    # Coordinates
    if cfg.do_coordinates and electrodes_table is not None:
        coords_cfg_final = coords_cfg if coords_cfg is not None else cfg.coords
        current_raw, artifact = attach_coordinates(
            raw=current_raw,
            electrodes_table=electrodes_table,
            cfg=coords_cfg_final,
            ctx=ctx_final,
        )
        artifacts.append(artifact)
        views["original"] = current_raw

    # Line noise
    if cfg.do_line_noise:
        current_raw, artifact = remove_line_noise(raw=current_raw, cfg=cfg.line_noise, ctx=ctx_final)
        artifacts.append(artifact)
        views["original"] = current_raw

    # High-pass
    if cfg.do_highpass:
        current_raw, artifact = highpass_filter(raw=current_raw, cfg=cfg.highpass, ctx=ctx_final)
        artifacts.append(artifact)
        views["original"] = current_raw

    # Resample
    if cfg.do_resample:
        current_raw, artifact = resample_raw(raw=current_raw, cfg=cfg.resample, ctx=ctx_final)
        artifacts.append(artifact)
        views["original"] = current_raw

    # Rereference (generates additional views)
    if cfg.do_rereference:
        reref_views, artifact = rereference(raw=current_raw, cfg=cfg.rereference, ctx=ctx_final)
        artifacts.append(artifact)
        # Ensure original reflects latest voltage-space raw
        if "original" in reref_views:
            views["original"] = reref_views["original"]
        for name, v in reref_views.items():
            if name == "original":
                continue
            views[name] = v
        reref_views["car"].save("test_raw.fif", overwrite=True)

    return PreprocResult(views=views, artifacts=artifacts, ctx=ctx_final)
