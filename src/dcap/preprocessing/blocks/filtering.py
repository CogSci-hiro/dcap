# =============================================================================
#                     ########################################
#                     #       BLOCK 4: FILTERING PATHS       #
#                     ########################################
# =============================================================================
#
# - High-pass filtering (drift removal)
# - Gamma/HFA envelope branch (feature-ready)
#
# v0: passthrough API skeleton.
#
# =============================================================================

from dataclasses import asdict
from typing import Tuple

from dcap.seeg.preprocessing.configs import GammaEnvelopeConfig, HighpassConfig
from dcap.seeg.preprocessing.types import BlockArtifact, PreprocContext


def highpass_filter(
    raw: "mne.io.BaseRaw",
    cfg: HighpassConfig,
    ctx: PreprocContext,
) -> Tuple["mne.io.BaseRaw", BlockArtifact]:
    """
    High-pass filter for drift removal (v0 passthrough).

    Usage example
    -------------
        ctx = PreprocContext()
        raw_out, artifact = highpass_filter(raw, HighpassConfig(l_freq=0.5), ctx)
    """
    import mne
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("highpass_filter expects an mne.io.BaseRaw.")

    ctx.add_record("highpass", asdict(cfg))

    artifact = BlockArtifact(
        name="highpass",
        parameters=asdict(cfg),
        summary_metrics={},
        warnings=["High-pass filtering not implemented yet; returning passthrough raw."],
        figures=[],
    )
    return raw, artifact


def compute_gamma_envelope(
    raw: "mne.io.BaseRaw",
    cfg: GammaEnvelopeConfig,
    ctx: PreprocContext,
) -> Tuple["mne.io.BaseRaw", BlockArtifact]:
    """
    Compute gamma/HFA envelope time series (v0 passthrough).

    Notes
    -----
    In v1, this will likely return a derived Raw whose channels represent envelope values
    (and must be clearly labeled so clinicians don't confuse it with voltage).

    Usage example
    -------------
        ctx = PreprocContext()
        env_raw, artifact = compute_gamma_envelope(raw, GammaEnvelopeConfig(), ctx)
    """
    import mne
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("compute_gamma_envelope expects an mne.io.BaseRaw.")

    ctx.add_record("gamma_envelope", asdict(cfg))

    artifact = BlockArtifact(
        name="gamma_envelope",
        parameters=asdict(cfg),
        summary_metrics={},
        warnings=["Gamma envelope not implemented yet; returning passthrough raw."],
        figures=[],
    )
    return raw, artifact
