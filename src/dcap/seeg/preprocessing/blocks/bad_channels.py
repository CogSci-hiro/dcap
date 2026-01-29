# =============================================================================
#                     ########################################
#                     #     BLOCK 6: BAD CHANNEL SUGGESTION  #
#                     ########################################
# =============================================================================
#
# Semi-automatic bad channel suggestion with human-readable reasons.
#
# v0: API skeleton only.
#
# =============================================================================

from dataclasses import asdict
from typing import Tuple

import mne

from dcap.seeg.preprocessing.configs import BadChannelsConfig
from dcap.seeg.preprocessing.types import BadChannelReason, BlockArtifact, PreprocContext


def suggest_bad_channels(
    raw: "mne.io.BaseRaw",
    cfg: BadChannelsConfig,
    ctx: PreprocContext,
) -> Tuple["mne.io.BaseRaw", BlockArtifact]:
    """
    Suggest bad channels with reasons (v0 returns none).

    Side effects
    ------------
    Initializes:
    - ctx.decisions["suggested_bad_channels"]: List[str]
    - ctx.decisions["bad_channel_reasons"]: Dict[str, List[BadChannelReason]]

    Usage example
    -------------
        ctx = PreprocContext()
        raw_out, artifact = suggest_bad_channels(raw, BadChannelsConfig(), ctx)
        suggested = ctx.decisions["suggested_bad_channels"]
    """
    import mne
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("suggest_bad_channels expects an mne.io.BaseRaw.")

    ctx.add_record("bad_channels", asdict(cfg))

    ctx.decisions.setdefault("suggested_bad_channels", [])
    ctx.decisions.setdefault("bad_channel_reasons", {})

    artifact = BlockArtifact(
        name="bad_channels",
        parameters=asdict(cfg),
        summary_metrics={"n_suggested": 0},
        warnings=["Bad channel detection not implemented yet; no suggestions produced."],
        figures=[],
    )
    return raw, artifact
