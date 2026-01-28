# =============================================================================
#                          BIDS: Raw transforms
# =============================================================================
#
# Apply channel renaming/types and montage. Keep each transform small and testable.
#
# REVIEW
# =============================================================================

from pathlib import Path
from typing import Dict, Optional

import mne
import pandas as pd

# You already have these in your older project; wire them in here when ready.
# from dcap.electrodes import get_mni_mono_coordinates, get_montage


def apply_line_frequency(raw: mne.io.BaseRaw, line_freq_hz: float) -> None:
    """
    Set required line frequency in-place.

    Usage example
    -------------
        apply_line_frequency(raw, 50.0)
    """
    raw.info["line_freq"] = float(line_freq_hz)


def apply_channel_renaming_from_tsv(raw: mne.io.BaseRaw, channels_tsv: Path, subject: str) -> None:
    """
    Rename channels using a TSV mapping file filtered by subject.

    Expected columns
    ----------------
    - subject
    - old
    - new

    Parameters
    ----------
    raw
        Raw object to rename in-place.
    channels_tsv
        TSV file path.
    subject
        BIDS subject label used in the TSV "subject" column.

    Returns
    -------
    None

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
        apply_channel_renaming_from_tsv(raw, Path("channels.tsv"), subject="NicEle")
    """
    df = pd.read_csv(channels_tsv, sep="\t")
    df_sub = df[df["subject"].astype(str) == str(subject)]

    mapping: Dict[str, str] = dict(zip(df_sub["old"].astype(str), df_sub["new"].astype(str)))
    if mapping:
        raw.rename_channels(mapping)


def apply_channel_types_default_seeg(raw: mne.io.BaseRaw, ecg_channel_name: str = "ECG") -> None:
    """
    Default channel type assignment for sEEG: set all to 'seeg' except ECG.

    Usage example
    -------------
        apply_channel_types_default_seeg(raw, ecg_channel_name="ECG")
    """
    mapping: Dict[str, str] = {ch: "seeg" for ch in raw.ch_names}
    if ecg_channel_name in mapping:
        mapping[ecg_channel_name] = "ecg"
    raw.set_channel_types(mapping)


def apply_montage_from_atlas(raw: mne.io.BaseRaw, atlas_path: Path) -> None:
    """
    Apply monopolar montage using an atlas file (placeholder).

    Notes
    -----
    This is intentionally a stub in the skeleton: wire in your existing
    `get_mni_mono_coordinates()` + `get_montage()` here.

    Usage example
    -------------
        apply_montage_from_atlas(raw, Path("elec2atlas.mat"))
    """
    _ = atlas_path
    # TODO: implement using your existing pipeline:
    # atlas = mat73.loadmat(atlas_path)
    # mni_coords = get_mni_mono_coordinates(atlas)
    # montage, _, _ = get_montage(raw, mni_coords, montage_type="monopolar")
    # raw.set_montage(montage)
    return
