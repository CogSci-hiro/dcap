# =============================================================================
#                           DCAP: Source I/O loaders
# =============================================================================
# > Convert a SourceItem into an MNE Raw.
# > Keep this narrow: loading only. No fancy preprocessing.

from pathlib import Path
from typing import Callable, Dict

import mne

from dcap_bids.heuristics import SourceItem
from dcap_bids.logging import get_logger

LOGGER = get_logger(__name__)


def load_source_raw(item: SourceItem) -> mne.io.BaseRaw:
    """
    Load a SourceItem into an MNE Raw object.

    Parameters
    ----------
    item
        Discovered source item.

    Returns
    -------
    mne.io.BaseRaw
        Loaded raw recording.

    Usage example
        raw = load_source_raw(item)
    """
    loaders: Dict[str, Callable[[Path], mne.io.BaseRaw]] = {
        "edf": _load_edf,
        "brainvision": _load_brainvision,
        "fif": _load_fif,
    }

    if item.kind not in loaders:
        raise ValueError(f"Unsupported source kind: {item.kind}")

    raw = loaders[item.kind](item.source_path)
    return raw


def _load_edf(path: Path) -> mne.io.BaseRaw:
    """
    Load EDF using MNE.

    Usage example
        raw = _load_edf(Path("recording.edf"))
    """
    LOGGER.debug("Loading EDF: %s", path)
    return mne.io.read_raw_edf(path, preload=False, verbose=False)


def _load_brainvision(path: Path) -> mne.io.BaseRaw:
    """
    Load BrainVision using MNE.

    Usage example
        raw = _load_brainvision(Path("recording.vhdr"))
    """
    LOGGER.debug("Loading BrainVision: %s", path)
    return mne.io.read_raw_brainvision(path, preload=False, verbose=False)


def _load_fif(path: Path) -> mne.io.BaseRaw:
    """
    Load FIF using MNE.

    Usage example
        raw = _load_fif(Path("recording_raw.fif"))
    """
    LOGGER.debug("Loading FIF: %s", path)
    return mne.io.read_raw_fif(path, preload=False, verbose=False)
