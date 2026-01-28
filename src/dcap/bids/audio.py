# =============================================================================
#                          BIDS: Audio handling
# =============================================================================
#
# Crop and (optionally) normalize WAV audio.
#
# REVIEW
# =============================================================================

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pydub
from scipy.io import wavfile


def load_audio_onsets_tsv(audio_onsets_tsv: Path) -> pd.DataFrame:
    """
    Load audio onset table.

    Expected columns
    ----------------
    - subject
    - run
    - onset

    Returns
    -------
    pd.DataFrame

    Example format
    --------------
    +---------+-----+-------+
    | subject | run | onset |
    +---------+-----+-------+
    | NicEle  | 1   | 12.34 |
    | NicEle  | 2   | 10.10 |
    +---------+-----+-------+

    Usage example
    -------------
        df = load_audio_onsets_tsv(Path("audio_onsets.tsv"))
    """
    return pd.read_csv(audio_onsets_tsv, sep="\t")


def crop_and_write_audio(
    src_wav: Path,
    dst_wav: Path,
    onset_s: float,
    duration_s: float,
    normalize_headroom_db: Optional[float] = 10.0,
) -> None:
    """
    Crop a WAV file to [onset, onset+duration] and write it out. Optionally normalize per channel.

    Parameters
    ----------
    src_wav
        Input WAV path.
    dst_wav
        Output WAV path.
    onset_s
        Crop start in seconds.
    duration_s
        Crop duration in seconds.
    normalize_headroom_db
        If not None, normalize each channel using pydub with given headroom (dB).

    Usage example
    -------------
        crop_and_write_audio(Path("in.wav"), Path("out.wav"), onset_s=12.3, duration_s=240.0)
    """
    sr, wav = wavfile.read(src_wav)

    start = int(round(float(onset_s) * sr))
    end = int(round((float(onset_s) + float(duration_s)) * sr))

    cropped = wav[start:end]

    dst_wav.parent.mkdir(parents=True, exist_ok=True)

    if normalize_headroom_db is None:
        wavfile.write(dst_wav, sr, cropped)
        return

    if cropped.ndim == 1:
        # Mono
        audio_segment = pydub.AudioSegment(
            cropped.tobytes(),
            frame_rate=sr,
            sample_width=cropped.dtype.itemsize,
            channels=1,
        )
        normalized = pydub.effects.normalize(audio_segment, headroom=float(normalize_headroom_db))
        normalized.export(dst_wav, format="wav")
        return

    if cropped.shape[1] != 2:
        raise ValueError("Expected stereo WAV for per-channel normalization in this skeleton.")

    ch0 = pydub.AudioSegment(
        cropped[:, 0].tobytes(),
        frame_rate=sr,
        sample_width=cropped.dtype.itemsize,
        channels=1,
    )
    ch1 = pydub.AudioSegment(
        cropped[:, 1].tobytes(),
        frame_rate=sr,
        sample_width=cropped.dtype.itemsize,
        channels=1,
    )

    norm0 = pydub.effects.normalize(ch0, headroom=float(normalize_headroom_db))
    norm1 = pydub.effects.normalize(ch1, headroom=float(normalize_headroom_db))
    stereo = pydub.AudioSegment.from_mono_audiosegments(norm0, norm1)
    stereo.export(dst_wav, format="wav")
