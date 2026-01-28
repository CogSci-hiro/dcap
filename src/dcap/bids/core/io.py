# =============================================================================
#                            BIDS: Source I/O
# =============================================================================

import codecs
from pathlib import Path
from typing import Tuple

import mne
import numpy as np
from scipy.io import wavfile

from dcap.bids.tasks.diapix.heuristics import SourceItem


def fix_brainvision_header_utf8(vhdr_path: Path) -> None:
    """
    Fix common BrainVision header encoding issues (µ -> u) in-place.

    Parameters
    ----------
    vhdr_path
        Path to the .vhdr file.

    Usage example
    -------------
        fix_brainvision_header_utf8(Path("conversation_1.vhdr"))
    """
    with codecs.open(vhdr_path, "rb") as f:
        data = f.read()

    # BrainVision sometimes uses byte 0xB5 (µ) which breaks UTF-8 parsing in some contexts.
    fixed = data.replace(b"\xb5", b"u")

    if fixed != data:
        with open(vhdr_path, "wb") as f:
            f.write(fixed)


def load_raw_brainvision(item: SourceItem, preload: bool) -> mne.io.BaseRaw:
    """
    Load BrainVision raw for a SourceItem.

    Parameters
    ----------
    item
        The SourceItem containing the .vhdr path.
    preload
        Whether to preload data into memory.

    Returns
    -------
    mne.io.BaseRaw
        Loaded raw data.

    Usage example
    -------------
        raw = load_raw_brainvision(item, preload=True)
    """
    if item.raw_vhdr is None:
        raise ValueError("SourceItem has no raw_vhdr; cannot load BrainVision raw.")

    fix_brainvision_header_utf8(item.raw_vhdr)
    raw = mne.io.read_raw_brainvision(item.raw_vhdr, preload=preload, verbose=False)
    return raw


def load_wav(path: Path) -> Tuple[int, np.ndarray]:
    """
    Load WAV audio via scipy.

    Parameters
    ----------
    path
        Path to a .wav file.

    Returns
    -------
    sr : int
        Sampling rate.
    wav : np.ndarray
        Audio array (shape [n_samples, n_channels] or [n_samples]).

    Usage example
    -------------
        sr, wav = load_wav(Path("conversation_1.wav"))
    """
    sr, wav = wavfile.read(path)
    return int(sr), wav
