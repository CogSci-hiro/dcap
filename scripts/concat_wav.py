# =============================================================================
#                     ########################################
#                     #         CONCAT WAVS FOR PRAAT         #
#                     ########################################
# =============================================================================
"""
Concatenate a directory of WAV files into a single WAV, and emit a manifest CSV
that maps each original file to its time span in the merged audio.

The manifest is later used to split a single annotated TextGrid back into
per-file TextGrids.

Assumptions
-----------
- Input WAVs are mono (or you want to force mono).
- All WAVs share the same sample rate (or you want to resample).
- Files are "about" 6 seconds, but can be slightly different.

Manifest CSV columns
--------------------
- index: integer order in the merged file
- filename: original wav filename (basename)
- start_s: start time in merged audio (seconds)
- end_s: end time in merged audio (seconds)
- duration_s: end_s - start_s (seconds)

Usage example
-------------
    python concat_wavs.py \
        --in_dir /path/to/wavs \
        --out_wav /path/to/merged.wav \
        --out_csv /path/to/manifest.csv \
        --sort natural
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires soundfile. Install with: pip install soundfile"
    ) from exc


# =============================================================================
#                                  CONSTANTS
# =============================================================================
DEFAULT_DTYPE: str = "float32"


# =============================================================================
#                               HELPER FUNCTIONS
# =============================================================================
_natural_key_regex = re.compile(r"(\d+)")


def _natural_sort_key(text: str) -> List[object]:
    """Sort key that treats embedded integers numerically."""
    parts: List[str] = _natural_key_regex.split(text)
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def _list_wavs(in_dir: Path, sort_mode: str) -> List[Path]:
    wavs: List[Path] = sorted(in_dir.glob("*.wav"))
    if sort_mode == "lex":
        return sorted(wavs, key=lambda p: p.name)
    if sort_mode == "natural":
        return sorted(wavs, key=lambda p: _natural_sort_key(p.name))
    raise ValueError(f"Unknown sort mode: {sort_mode!r} (expected 'lex' or 'natural')")


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert audio to mono if needed (mean across channels)."""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1)
    raise ValueError(f"Unsupported audio shape: {audio.shape}")


@dataclass(frozen=True)
class ManifestRow:
    index: int
    filename: str
    start_s: float
    end_s: float
    duration_s: float


# =============================================================================
#                                   MAIN LOGIC
# =============================================================================
def concat_wavs(
    *,
    in_dir: Path,
    out_wav: Path,
    out_csv: Path,
    sort_mode: str = "natural",
    force_mono: bool = True,
) -> None:
    """
    Concatenate WAV files from `in_dir` into one WAV, and write a CSV manifest.

    Parameters
    ----------
    in_dir
        Directory containing input .wav files (flat, no recursion).
    out_wav
        Path to write the merged wav.
    out_csv
        Path to write the manifest CSV.
    sort_mode
        'natural' (recommended) or 'lex'.
    force_mono
        If True, convert multi-channel WAVs to mono by averaging channels.

    Returns
    -------
    None
    """
    in_dir = in_dir.expanduser().resolve()
    out_wav = out_wav.expanduser().resolve()
    out_csv = out_csv.expanduser().resolve()

    wav_paths: List[Path] = _list_wavs(in_dir=in_dir, sort_mode=sort_mode)
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in: {in_dir}")

    # Read all audio, verify sample rate compatibility
    audio_blocks: List[np.ndarray] = []
    sample_rates: List[int] = []

    for p in wav_paths:
        audio, sr = sf.read(p, always_2d=False)
        audio_np = np.asarray(audio, dtype=np.float32)

        if force_mono:
            audio_np = _to_mono(audio_np)

        audio_blocks.append(audio_np)
        sample_rates.append(int(sr))

    sr0 = sample_rates[0]
    if any(sr != sr0 for sr in sample_rates):
        raise ValueError(
            f"Sample rates differ across files: {sorted(set(sample_rates))}. "
            "Resampling is intentionally not implemented here to keep this minimal."
        )

    # Build merged audio + manifest
    manifest: List[ManifestRow] = []
    merged: List[np.ndarray] = []
    cursor_samples: int = 0

    for i, (p, block) in enumerate(zip(wav_paths, audio_blocks)):
        n = int(block.shape[0])
        start_s = cursor_samples / sr0
        end_s = (cursor_samples + n) / sr0

        manifest.append(
            ManifestRow(
                index=i,
                filename=p.name,
                start_s=float(start_s),
                end_s=float(end_s),
                duration_s=float(end_s - start_s),
            )
        )

        merged.append(block)
        cursor_samples += n

    merged_audio = np.concatenate(merged, axis=0).astype(DEFAULT_DTYPE)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    sf.write(out_wav, merged_audio, sr0)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["index", "filename", "start_s", "end_s", "duration_s"]
        )
        writer.writeheader()
        for row in manifest:
            writer.writerow(
                {
                    "index": row.index,
                    "filename": row.filename,
                    "start_s": f"{row.start_s:.9f}",
                    "end_s": f"{row.end_s:.9f}",
                    "duration_s": f"{row.duration_s:.9f}",
                }
            )


# =============================================================================
#                                    CLI
# =============================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Concatenate WAVs for Praat annotation.")
    p.add_argument("--in_dir", type=Path, required=True)
    p.add_argument("--out_wav", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--sort", type=str, default="natural", choices=["natural", "lex"])
    p.add_argument("--no_mono", action="store_true", help="Do not force mono.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    concat_wavs(
        in_dir=args.in_dir,
        out_wav=args.out_wav,
        out_csv=args.out_csv,
        sort_mode=args.sort,
        force_mono=not args.no_mono,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
