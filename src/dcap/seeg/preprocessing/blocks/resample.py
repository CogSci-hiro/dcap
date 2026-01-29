# =============================================================================
#                     ########################################
#                     #        BLOCK 5: RESAMPLING           #
#                     ########################################
# =============================================================================
#
# Logic only:
# - No file I/O
# - No CLI / printing
#
# =============================================================================

from dataclasses import asdict
from typing import Tuple

import numpy as np
import mne

from dcap.seeg.preprocessing.configs import ResampleConfig
from dcap.seeg.preprocessing.types import BlockArtifact, PreprocContext


# =============================================================================
#                              INTERNAL CONSTANTS
# =============================================================================
_SFREQ_RTOL: float = 1e-6
_SFREQ_ATOL: float = 1e-9


def resample_raw(
    raw: mne.io.BaseRaw,
    cfg: ResampleConfig,
    ctx: PreprocContext,
) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    """
    Resample recording to a target sampling rate.

    Parameters
    ----------
    raw
        MNE Raw object.
    cfg
        Resampling configuration.
    ctx
        Preprocessing context.

    Returns
    -------
    raw_out
        Resampled Raw (a copy). If sfreq is already equal to target (within tolerance),
        returns a copy without resampling.
    artifact
        Block artifact.

    Notes
    -----
    - Uses `mne.io.BaseRaw.resample`, which applies appropriate anti-alias filtering.
    - MNE preserves annotation timing in seconds. Events in sample indices must be
      recomputed downstream (but that is outside this block).

    Usage example
    -------------
        ctx = PreprocContext()
        raw_rs, artifact = resample_raw(raw, ResampleConfig(sfreq_out=512.0), ctx)
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("resample_raw expects an mne.io.BaseRaw.")

    sfreq_in = float(raw.info["sfreq"])
    sfreq_out = float(cfg.sfreq_out)

    if sfreq_out <= 0:
        raise ValueError(f"sfreq_out must be > 0, got {sfreq_out}.")

    ctx.add_record("resample", asdict(cfg))

    # If already at target (within tolerance), just return a copy.
    if np.isclose(sfreq_in, sfreq_out, rtol=_SFREQ_RTOL, atol=_SFREQ_ATOL):
        raw_out = raw.copy()
        artifact = BlockArtifact(
            name="resample",
            parameters=asdict(cfg),
            summary_metrics={
                "sfreq_in": sfreq_in,
                "sfreq_out": sfreq_in,
                "changed": False,
                "n_times_in": int(raw.n_times),
                "n_times_out": int(raw.n_times),
            },
            warnings=["Sampling rate already matches target; no resampling applied."],
            figures=[],
        )
        return raw_out, artifact

    raw_out = raw.copy()
    n_times_in = int(raw_out.n_times)

    # `pad="auto"` is MNE's safe default to mitigate edge artifacts.
    raw_out.resample(sfreq=sfreq_out, npad="auto", verbose=False)

    n_times_out = int(raw_out.n_times)
    sfreq_after = float(raw_out.info["sfreq"])

    # Sanity check: MNE should set sfreq exactly, but keep this robust.
    warnings: list[str] = []
    if not np.isclose(sfreq_after, sfreq_out, rtol=_SFREQ_RTOL, atol=_SFREQ_ATOL):
        warnings.append(
            f"Post-resample sfreq ({sfreq_after}) differs from requested ({sfreq_out})."
        )

    artifact = BlockArtifact(
        name="resample",
        parameters=asdict(cfg),
        summary_metrics={
            "sfreq_in": sfreq_in,
            "sfreq_out": sfreq_after,
            "changed": True,
            "n_times_in": n_times_in,
            "n_times_out": n_times_out,
            "duration_sec_in": float(n_times_in) / sfreq_in,
            "duration_sec_out": float(n_times_out) / sfreq_after,
        },
        warnings=warnings,
        figures=[],
    )
    return raw_out, artifact
