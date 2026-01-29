# =============================================================================
#                     ########################################
#                     #       BLOCK 7: REREFERENCING         #
#                     ########################################
# =============================================================================
#
# Generate rereferenced views:
# - CAR (common average)
# - bipolar (shaft-local)
# - WM reference
# - Laplacian (knn_3d or shaft_1d)
#
# v0: returns {"original": raw} only.
#
# =============================================================================

from dataclasses import asdict
from typing import Dict, Tuple

from dcap.preprocessing.configs import RereferenceConfig
from dcap.preprocessing.types import BlockArtifact, PreprocContext


def rereference(
    raw: "mne.io.BaseRaw",
    cfg: RereferenceConfig,
    ctx: PreprocContext,
) -> Tuple[Dict[str, "mne.io.BaseRaw"], BlockArtifact]:
    """
    Generate rereferenced views (v0 skeleton).

    Returns
    -------
    views
        Mapping from view name to Raw object. v0 returns {"original": raw}.
    artifact
        Block artifact.

    Usage example
    -------------
        ctx = PreprocContext()
        views, artifact = rereference(raw, RereferenceConfig(methods=("car",)), ctx)
        raw_original = views["original"]
    """
    import mne
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("rereference expects an mne.io.BaseRaw.")

    ctx.add_record("rereference", asdict(cfg))

    views: Dict[str, "mne.io.BaseRaw"] = {"original": raw}

    artifact = BlockArtifact(
        name="rereference",
        parameters=asdict(cfg),
        summary_metrics={"views": list(views.keys())},
        warnings=["Rereferencing not implemented yet; only 'original' view returned."],
        figures=[],
    )
    return views, artifact
