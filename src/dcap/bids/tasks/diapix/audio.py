# dcap/bids/tasks/diapix/audio.py
# =============================================================================
#                               DIAPIX AUDIO
# =============================================================================

from pathlib import Path
from typing import Final

import pandas as pd
import pydub
from scipy.io import wavfile


CONVERSATION_DURATION_S: Final[float] = 4.0 * 60.0
NORMALIZE_HEADROOM_DB: Final[float] = 10.0


def crop_and_normalize_audio(
    *,
    src_wav: Path,
    dst_wav: Path,
    audio_onsets_tsv: Path,
    subject_bids: str,
    run: str,
) -> None:
    """
    Crop Diapix WAV using a manually curated onset table and normalize each channel.

    Parameters
    ----------
    src_wav
        Original WAV file (uncropped).
    dst_wav
        Output WAV file.
    audio_onsets_tsv
        Path to `audio_onsets.tsv` containing onset seconds per subject/run.
        Expected columns: subject, run, onset
    subject_bids
        BIDS subject label without "sub-".
    run
        Run number as string.

    Usage example
    -------------
        crop_and_normalize_audio(
            src_wav=Path("conversation_1.wav"),
            dst_wav=Path("sub-NicEle_task-diapix_run-1.wav"),
            audio_onsets_tsv=Path("audio_onsets.tsv"),
            subject_bids="NicEle",
            run="1",
        )
    """
    if not src_wav.exists():
        raise FileNotFoundError(f"Missing WAV: {src_wav}")

    audio_df = pd.read_csv(audio_onsets_tsv, sep="\t")
    required_cols = {"subject", "run", "onset"}
    missing = required_cols - set(audio_df.columns)
    if missing:
        raise ValueError(f"{audio_onsets_tsv} missing columns: {sorted(missing)}")

    run_int = int(run)
    row = audio_df[(audio_df["subject"] == subject_bids) & (audio_df["run"] == run_int)]
    if row.empty:
        raise ValueError(f"No onset row for subject={subject_bids}, run={run}")

    onset_s = float(row["onset"].values[0])

    sr, wav = wavfile.read(src_wav)

    start = int(onset_s * sr)
    stop = int((onset_s + CONVERSATION_DURATION_S) * sr)
    wav = wav[start:stop]

    if wav.ndim != 2 or wav.shape[1] != 2:
        raise ValueError(f"Expected stereo WAV (n_samples, 2). Got shape={wav.shape}")

    dst_wav.parent.mkdir(parents=True, exist_ok=True)

    seg_left = pydub.AudioSegment(
        wav[:, 0].tobytes(),
        frame_rate=sr,
        sample_width=wav.dtype.itemsize,
        channels=1,
    )
    seg_right = pydub.AudioSegment(
        wav[:, 1].tobytes(),
        frame_rate=sr,
        sample_width=wav.dtype.itemsize,
        channels=1,
    )

    norm_left = pydub.effects.normalize(seg_left, headroom=NORMALIZE_HEADROOM_DB)
    norm_right = pydub.effects.normalize(seg_right, headroom=NORMALIZE_HEADROOM_DB)

    stereo = pydub.AudioSegment.from_mono_audiosegments(norm_left, norm_right)
    stereo.export(dst_wav, format="wav")
