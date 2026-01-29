# src/dcap/bids/tasks/diapix/audio.py
# =============================================================================
#                       Diapix: Audio cropping + onsets
# =============================================================================

from pathlib import Path
from typing import Optional

import pandas as pd
import pydub
from scipy.io import wavfile


def crop_and_normalize_audio(
    src_wav: Path,
    dst_wav: Path,
    audio_onsets_tsv: Path,
    dcap_id: str,
    run: str,
    duration_s: float,
    normalize_headroom_db: Optional[float] = 10.0,
) -> None:
    """
    Crop audio using onset times indexed by dcap_id and run.

    Expected TSV columns
    --------------------
    - dcap_id
    - run
    - onset   (seconds)   [or onset_s, adjust below]

    Usage example
    -------------
        crop_and_normalize_audio(
            src_wav=Path("conversation_1.wav"),
            dst_wav=Path("bids/sub-001/audio/sub-001_task-diapix_run-1.wav"),
            audio_onsets_tsv=Path("private/audio_onsets.tsv"),
            dcap_id="Nic-Ele",
            run="1",
            duration_s=240.0,
        )
    """
    df = pd.read_csv(audio_onsets_tsv, sep="\t")

    # Normalize keys
    df["dcap_id"] = df["dcap_id"].astype(str).str.strip()
    df["run"] = df["run"].apply(lambda x: str(int(str(x).strip())))
    run_norm = str(int(str(run).strip()))
    dcap_id_norm = str(dcap_id).strip()

    onset_col = "onset" if "onset" in df.columns else "onset_s"
    if onset_col not in df.columns:
        raise ValueError(f"audio_onsets_tsv must contain 'onset' or 'onset_s' column: {audio_onsets_tsv}")

    match = df[(df["dcap_id"] == dcap_id_norm) & (df["run"] == run_norm)]
    if match.shape[0] == 0:
        available_runs = df[df["dcap_id"] == dcap_id_norm]["run"].unique().tolist()
        raise ValueError(
            f"No onset row for dcap_id={dcap_id_norm}, run={run_norm}. "
            f"Available runs for this dcap_id in onset table: {available_runs}"
        )
    if match.shape[0] > 1:
        raise ValueError(f"Multiple onset rows for dcap_id={dcap_id_norm}, run={run_norm}")

    onset_s = float(match.iloc[0][onset_col])

    sr, wav = wavfile.read(src_wav)
    start = int(round(onset_s * sr))
    end = int(round((onset_s + float(duration_s)) * sr))
    cropped = wav[start:end]

    dst_wav.parent.mkdir(parents=True, exist_ok=True)

    if normalize_headroom_db is None:
        wavfile.write(dst_wav, sr, cropped)
        return

    # Normalize per channel using pydub
    if cropped.ndim == 1:
        seg = pydub.AudioSegment(
            cropped.tobytes(),
            frame_rate=sr,
            sample_width=cropped.dtype.itemsize,
            channels=1,
        )
        norm = pydub.effects.normalize(seg, headroom=float(normalize_headroom_db))
        norm.export(dst_wav, format="wav")
        return

    if cropped.shape[1] != 2:
        raise ValueError("Expected mono or stereo WAV for normalization in this implementation.")

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
