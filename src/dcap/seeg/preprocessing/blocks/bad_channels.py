# =============================================================================
#                     ########################################
#                     #     BLOCK 6: BAD CHANNEL SUGGESTION  #
#                     ########################################
# =============================================================================
#
# Prefer BIDS channels.tsv over heuristics.
#
# =============================================================================

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import pandas as pd

from dcap.seeg.io.sidecars import find_neighbor_sidecar
from dcap.seeg.preprocessing.configs.bad_channels import BadChannelsConfig
from dcap.seeg.preprocessing.types import BadChannelReason, BlockArtifact, PreprocContext


# =============================================================================
# Helpers
# =============================================================================

def _infer_bids_channels_tsv(raw: mne.io.BaseRaw) -> Optional[Path]:
    """
    Infer the BIDS *_channels.tsv path from raw.filenames.

    Notes
    -----
    This assumes the raw was loaded from a BIDS-named file like:
        sub-XXX[_ses-YYY]_task-ZZZ[_run-N]_ieeg.edf
    and that the channels sidecar lives alongside it:
        sub-XXX[_ses-YYY]_task-ZZZ[_run-N]_channels.tsv
    """
    if not getattr(raw, "filenames", None):
        return None
    if raw.filenames is None or len(raw.filenames) == 0:
        return None
    if raw.filenames[0] is None:
        return None

    data_path = Path(raw.filenames[0])
    return find_neighbor_sidecar(data_path, sidecar_suffix="_channels.tsv")


def _read_bad_channels_from_bids_tsv(channels_tsv: Path) -> Tuple[List[str], Dict[str, str]]:
    """
    Read BIDS channels.tsv and return:
    - bad channel names
    - mapping channel -> status_description (if present)
    """
    df = pd.read_csv(channels_tsv, sep="\t")

    # BIDS requires "name" and "status" for channels.tsv in iEEG/EEG.
    if "name" not in df.columns or "status" not in df.columns:
        raise ValueError(
            f"channels.tsv missing required columns. Found columns={list(df.columns)} "
            f"at {channels_tsv}"
        )

    status = df["status"].astype(str).str.strip().str.lower()
    bad_mask = status == "bad"

    bad_names = df.loc[bad_mask, "name"].astype(str).str.strip().tolist()

    desc_map: Dict[str, str] = {}
    if "status_description" in df.columns:
        sub = df.loc[bad_mask, ["name", "status_description"]].copy()
        sub["name"] = sub["name"].astype(str).str.strip()
        sub["status_description"] = sub["status_description"].astype(str).fillna("").str.strip()
        desc_map = dict(zip(sub["name"].tolist(), sub["status_description"].tolist()))

    return bad_names, desc_map


def _make_reason(*, description: str) -> BadChannelReason:
    """
    Construct a BadChannelReason from BIDS status fields.

    NOTE: Adjust field names here if your BadChannelReason dataclass differs.
    """
    # Common patterns are:
    # - BadChannelReason(code=..., message=..., source=...)
    # - BadChannelReason(reason=..., details=...)
    #
    # This implementation assumes (code: str, message: str).
    message = description if description else "Marked bad in BIDS channels.tsv (no description)."
    return BadChannelReason(code="bids_channels_tsv", message=message)  # type: ignore[arg-type]


# =============================================================================
# Main
# =============================================================================

def suggest_bad_channels(
    raw: "mne.io.BaseRaw",
    cfg: BadChannelsConfig,
    ctx: PreprocContext,
) -> Tuple["mne.io.BaseRaw", BlockArtifact]:
    """
    Suggest bad channels using BIDS channels.tsv (status == "bad").

    Side effects
    ------------
    Initializes/updates:
    - ctx.decisions["suggested_bad_channels"]: List[str]
    - ctx.decisions["bad_channel_reasons"]: Dict[str, List[BadChannelReason]]

    Also updates:
    - raw.info["bads"] (merged; unique; preserves existing)

    Usage example
    -------------
        ctx = PreprocContext()
        raw_out, artifact = suggest_bad_channels(raw, BadChannelsConfig(), ctx)
        suggested = ctx.decisions["suggested_bad_channels"]
        reasons = ctx.decisions["bad_channel_reasons"]
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("suggest_bad_channels expects an mne.io.BaseRaw.")

    ctx.add_record("bad_channels", asdict(cfg))
    ctx.decisions.setdefault("suggested_bad_channels", [])
    ctx.decisions.setdefault("bad_channel_reasons", {})

    warnings: List[str] = []
    suggested: List[str] = []
    reasons: Dict[str, List[BadChannelReason]] = {}

    channels_tsv = _infer_bids_channels_tsv(raw)
    if channels_tsv is None:
        warnings.append(
            "Could not infer *_channels.tsv from raw.filenames; no bad channels suggested."
        )
        artifact = BlockArtifact(
            name="bad_channels",
            parameters=asdict(cfg),
            summary_metrics={
                "n_suggested": 0,
                "n_in_tsv": 0,
                "n_missing_from_raw": 0,
            },
            warnings=warnings,
            figures=[],
        )
        return raw, artifact

    try:
        bad_in_tsv, desc_map = _read_bad_channels_from_bids_tsv(channels_tsv)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Failed to read BIDS channels.tsv at {channels_tsv}: {e}")
        artifact = BlockArtifact(
            name="bad_channels",
            parameters={**asdict(cfg), "channels_tsv": str(channels_tsv)},
            summary_metrics={
                "n_suggested": 0,
                "n_in_tsv": 0,
                "n_missing_from_raw": 0,
            },
            warnings=warnings,
            figures=[],
        )
        return raw, artifact

    raw_chs = set(raw.ch_names)
    missing = [ch for ch in bad_in_tsv if ch not in raw_chs]
    present = [ch for ch in bad_in_tsv if ch in raw_chs]

    if missing:
        warnings.append(
            f"{len(missing)} channel(s) marked bad in channels.tsv were not found in raw.ch_names "
            f"(examples: {missing[:5]})."
        )

    for ch in present:
        suggested.append(ch)
        reasons[ch] = [_make_reason(description=desc_map.get(ch, ""))]

    # Merge into ctx
    ctx.decisions["suggested_bad_channels"] = sorted(set(ctx.decisions["suggested_bad_channels"]) | set(suggested))
    # Merge reasons, but don’t blow away existing entries
    existing_reasons = ctx.decisions["bad_channel_reasons"]
    for ch, rlist in reasons.items():
        if ch not in existing_reasons:
            existing_reasons[ch] = rlist
        else:
            existing_reasons[ch].extend(rlist)

    # Merge into raw.info["bads"]
    existing_bads = list(getattr(raw.info, "bads", []) or [])
    raw.info["bads"] = sorted(set(existing_bads) | set(suggested))

    artifact = BlockArtifact(
        name="bad_channels",
        parameters={**asdict(cfg), "channels_tsv": str(channels_tsv)},
        summary_metrics={
            "n_suggested": len(suggested),
            "n_in_tsv": len(bad_in_tsv),
            "n_missing_from_raw": len(missing),
        },
        warnings=warnings,
        figures=[],
    )
    return raw, artifact
