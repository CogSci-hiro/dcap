# =============================================================================
#                     ########################################
#                     #        BLOCK 5: RESAMPLING           #
#                     ########################################
# =============================================================================
#
# v0: passthrough API skeleton.
#
# =============================================================================

from dataclasses import asdict
from typing import Tuple

from dcap.seeg.preprocessing.configs import ResampleConfig
from dcap.seeg.preprocessing.types import BlockArtifact, PreprocContext


def resample_raw(
    raw: "mne.io.BaseRaw",
    cfg: ResampleConfig,
    ctx: PreprocContext,
) -> Tuple["mne.io.BaseRaw", BlockArtifact]:
    """
    Resample recording to a target sampling rate (v0 passthrough).

    Usage example
    -------------
        ctx = PreprocContext()
        raw_out, artifact = resample_raw(raw, ResampleConfig(sfreq_out=512.0), ctx)
    """
    import mne
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("resample_raw expects an mne.io.BaseRaw.")

    ctx.add_record("resample", asdict(cfg))

    artifact = BlockArtifact(
        name="resample",
        parameters=asdict(cfg),
        summary_metrics={},
        warnings=["Resampling not implemented yet; returning passthrough raw."],
        figures=[],
    )
    return raw, artifact
