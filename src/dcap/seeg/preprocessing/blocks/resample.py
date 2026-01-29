# =============================================================================
# =============================================================================
#                     ########################################
#                     #        BLOCK 5: RESAMPLING           #
#                     ########################################
# =============================================================================
# =============================================================================

from dataclasses import asdict
from typing import Tuple

import numpy as np
import mne

from dcap.seeg.preprocessing.configs import ResampleConfig
from dcap.seeg.preprocessing.types import BlockArtifact, PreprocContext


_SFREQ_RTOL: float = 1e-6
_SFREQ_ATOL: float = 1e-9


def resample_raw(
    raw: mne.io.BaseRaw,
    cfg: ResampleConfig,
    ctx: PreprocContext,
) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    """Resample recording to a target sampling rate."""
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("resample_raw expects an mne.io.BaseRaw.")

    ctx.add_record("resample", asdict(cfg))

    sfreq_in = float(raw.info["sfreq"])
    sfreq_out = float(cfg.sfreq_out)

    raw_out = raw.copy()

    if np.isclose(sfreq_in, sfreq_out, rtol=_SFREQ_RTOL, atol=_SFREQ_ATOL):
        artifact = BlockArtifact(
            name="resample",
            parameters=asdict(cfg),
            summary_metrics={"changed": 0.0, "sfreq_in": sfreq_in, "sfreq_out": sfreq_in},
            warnings=["Sampling rate already matches target; no resampling applied."],
            figures=[],
        )
        return raw_out, artifact

    n_times_in = int(raw_out.n_times)
    raw_out.resample(sfreq=sfreq_out, npad="auto", verbose=False)
    n_times_out = int(raw_out.n_times)

    artifact = BlockArtifact(
        name="resample",
        parameters=asdict(cfg),
        summary_metrics={
            "changed": 1.0,
            "sfreq_in": sfreq_in,
            "sfreq_out": float(raw_out.info["sfreq"]),
            "n_times_in": float(n_times_in),
            "n_times_out": float(n_times_out),
        },
        warnings=[],
        figures=[],
    )
    return raw_out, artifact
