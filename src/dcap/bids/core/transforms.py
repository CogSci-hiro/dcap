# src/dcap/bids/core/transforms.py
# =============================================================================
#                         BIDS Core: Raw transforms
# =============================================================================
#
# Task-agnostic transforms that are broadly reusable across tasks.
#
# Scope
# -----
# - Set mandatory BIDS fields (e.g., line frequency)
# - Apply channel renaming maps
# - Apply channel type maps
# - Provide small, safe utilities around channel selection
#
# Non-goals
# ---------
# - Task-specific channel policies (e.g., "drop NULL electrodes") belong in tasks
# - Atlas/montage construction belongs in tasks (or a shared non-core module),
#   because assumptions vary heavily across tasks/datasets.
#
# REVIEW
# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import mne
import pandas as pd


# =============================================================================
# Mandatory / common metadata
# =============================================================================

def apply_line_frequency(raw: mne.io.BaseRaw, line_freq_hz: float) -> None:
    """
    Set the line frequency (Hz) in-place.

    Parameters
    ----------
    raw
        Raw object to modify.
    line_freq_hz
        Power line frequency (e.g., 50 in EU, 60 in US).

    Returns
    -------
    None

    Usage example
    -------------
        apply_line_frequency(raw, 50.0)
    """
    raw.info["line_freq"] = float(line_freq_hz)


# =============================================================================
# Channel renaming
# =============================================================================

def apply_channel_renaming(raw: mne.io.BaseRaw, mapping: Mapping[str, str]) -> None:
    """
    Rename channels in-place.

    Parameters
    ----------
    raw
        Raw object to modify.
    mapping
        Dict-like mapping {old_name: new_name}.

    Returns
    -------
    None

    Notes
    -----
    - Only keys present in raw.ch_names will be applied.
    - MNE will raise if the renaming produces duplicate channel names.

    Usage example
    -------------
        apply_channel_renaming(raw, {"A1": "A01", "A2": "A02"})
    """
    if not mapping:
        return

    existing = set(raw.ch_names)
    filtered: Dict[str, str] = {k: v for k, v in mapping.items() if k in existing}
    if not filtered:
        return

    raw.rename_channels(filtered)


def load_channel_rename_mapping_from_tsv(channels_tsv: Path, subject: str) -> Dict[str, str]:
    """
    Load a channel rename mapping from a TSV file, filtered by subject.

    Expected columns
    ----------------
    - subject
    - old
    - new

    Parameters
    ----------
    channels_tsv
        TSV path.
    subject
        Subject identifier as stored in the TSV (typically BIDS subject label).

    Returns
    -------
    Dict[str, str]
        Rename mapping {old: new}. May be empty if no rows match.

    Example TSV format
    ------------------
    +---------+------+------+
    | subject | old  | new  |
    +---------+------+------+
    | NicEle  | A1   | A01  |
    | NicEle  | A2   | A02  |
    +---------+------+------+

    Usage example
    -------------
        mapping = load_channel_rename_mapping_from_tsv(Path("channels.tsv"), subject="NicEle")
        apply_channel_renaming(raw, mapping)
    """
    df = pd.read_csv(channels_tsv, sep="\t")
    df_sub = df[df["subject"].astype(str) == str(subject)]

    mapping: Dict[str, str] = {}
    for old, new in zip(df_sub["old"].astype(str), df_sub["new"].astype(str)):
        mapping[old] = new
    return mapping


# =============================================================================
# Channel types
# =============================================================================

def apply_channel_types(raw: mne.io.BaseRaw, mapping: Mapping[str, str]) -> None:
    """
    Apply channel types in-place.

    Parameters
    ----------
    raw
        Raw object to modify.
    mapping
        Mapping {channel_name: mne_type}, e.g. {"ECG": "ecg", "A01": "seeg"}.

    Returns
    -------
    None

    Notes
    -----
    - Unknown channel names are ignored.
    - Channel types must be valid MNE channel type strings.

    Usage example
    -------------
        apply_channel_types(raw, {"ECG": "ecg"})
    """
    if not mapping:
        return

    existing = set(raw.ch_names)
    filtered: Dict[str, str] = {k: v for k, v in mapping.items() if k in existing}
    if not filtered:
        return

    raw.set_channel_types(filtered)


def build_default_seeg_channel_types(
    channel_names: Sequence[str],
    ecg_channel_name: str = "ECG",
) -> Dict[str, str]:
    """
    Build a conservative default channel-type mapping for sEEG.

    Parameters
    ----------
    channel_names
        Channel names to type.
    ecg_channel_name
        If present, set this channel to "ecg" and all others to "seeg".

    Returns
    -------
    Dict[str, str]
        Mapping {ch: type}.

    Usage example
    -------------
        types = build_default_seeg_channel_types(raw.ch_names, ecg_channel_name="ECG")
        apply_channel_types(raw, types)
    """
    mapping: Dict[str, str] = {str(ch): "seeg" for ch in channel_names}
    if ecg_channel_name in mapping:
        mapping[ecg_channel_name] = "ecg"
    return mapping


def build_default_channel_types(
    *,
    channel_names: Sequence[str],
    datatype: str,
    ecg_channel_name: str = "ECG",
) -> Dict[str, str]:
    """
    Build a conservative default channel-type mapping for the requested datatype.

    Supported defaults are intentionally narrow:
    - ``ieeg`` -> all non-ECG channels become ``seeg``
    - ``eeg`` -> all non-ECG channels become ``eeg``

    Other datatypes currently fall back to the existing iEEG-safe behavior so we
    do not silently broaden policy beyond what this repo already supports.
    """
    datatype_norm = str(datatype).strip().lower()
    if datatype_norm == "eeg":
        mapping: Dict[str, str] = {str(ch): "eeg" for ch in channel_names}
        if ecg_channel_name in mapping:
            mapping[ecg_channel_name] = "ecg"
        return mapping

    return build_default_seeg_channel_types(
        channel_names=channel_names,
        ecg_channel_name=ecg_channel_name,
    )


# =============================================================================
# Channel selection helpers (safe, task-agnostic)
# =============================================================================

def pick_channels_present(raw: mne.io.BaseRaw, keep: Iterable[str]) -> list[str]:
    """
    Return channel names in `keep` that are present in `raw`, preserving order.

    Usage example
    -------------
        picks = pick_channels_present(raw, ["ECG", "A01", "A02"])
    """
    existing = set(raw.ch_names)
    return [ch for ch in keep if ch in existing]


def drop_channels_if_present(raw: mne.io.BaseRaw, drop: Iterable[str]) -> None:
    """
    Drop channels if they exist. No error if they do not.

    Parameters
    ----------
    raw
        Raw object to modify.
    drop
        Iterable of channel names to drop.

    Returns
    -------
    None

    Usage example
    -------------
        drop_channels_if_present(raw, ["TRIG", "DC01"])
    """
    to_drop = [ch for ch in drop if ch in raw.ch_names]
    if to_drop:
        raw.drop_channels(to_drop)
