# =============================================================================
#                     ########################################
#                     #    BLOCK 3: LINE-NOISE REMOVAL       #
#                     ########################################
# =============================================================================
#
# Notch filtering (MNE) or Zapline (meegkit) line-noise removal.
#
# v0: passthrough API skeleton. Implements provenance + artifact only.
#
# =============================================================================

from dataclasses import asdict
from typing import Tuple

import mne

from dcap.preprocessing.configs import LineNoiseConfig
from dcap.preprocessing.types import BlockArtifact, PreprocContext


def remove_line_noise(
    raw: "mne.io.BaseRaw",
    cfg: LineNoiseConfig,
    ctx: PreprocContext,
) -> Tuple["mne.io.BaseRaw", BlockArtifact]:
    """
    Remove line noise using notch or zapline.

    Parameters
    ----------
    raw
        MNE Raw object.
    cfg
        Line-noise configuration.
    ctx
        Preprocessing context.

    Returns
    -------
    raw_out
        Raw object (passthrough in v0).
    artifact
        Block artifact.

    Usage example
    -------------
        ctx = PreprocContext()
        raw_out, artifact = remove_line_noise(
            raw=raw,
            cfg=LineNoiseConfig(method="notch", freq_base=50),
            ctx=ctx,
        )
    """
    import mne  # local import
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("remove_line_noise expects an mne.io.BaseRaw.")

    ctx.add_record("line_noise", asdict(cfg))

    artifact = BlockArtifact(
        name="line_noise",
        parameters=asdict(cfg),
        summary_metrics={},
        warnings=[f"{cfg.method} is not implemented yet; returning passthrough raw."],
        figures=[],
    )
    return raw, artifact
