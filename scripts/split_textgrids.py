# =============================================================================
#                     ########################################
#                     #       SPLIT PRAAT TEXTGRID BACK       #
#                     ########################################
# =============================================================================
"""
Split a single Praat TextGrid (annotated on a merged WAV) into per-file TextGrids.

Inputs
------
- merged_textgrid: TextGrid made for the concatenated audio (Praat output)
- manifest_csv: CSV produced by concat_wavs.py
- out_dir: directory to write per-file .TextGrid files into

Behavior
--------
For each original file segment [start_s, end_s] in the merged timeline:
- Extract all intervals/points that overlap that window
- Clip them to the window bounds
- Shift times by -start_s so each output TextGrid starts at 0.0 seconds
- Write <original_filename>.TextGrid (or configurable naming)

Important note
--------------
To avoid edge-case losses at segment boundaries (e.g. labels exactly at 6.0),
you can optionally add a tiny epsilon when deciding overlap.

Usage example
-------------
    python split_textgrid.py \
        --merged_tg /path/to/merged.TextGrid \
        --manifest_csv /path/to/manifest.csv \
        --out_dir /path/to/per_file_textgrids
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from praatio import textgrid


# =============================================================================
#                                  CONSTANTS
# =============================================================================
DEFAULT_EPSILON_S: float = 1e-9


# =============================================================================
#                                   MODELS
# =============================================================================
@dataclass(frozen=True)
class Segment:
    index: int
    filename: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return float(self.end_s - self.start_s)


# =============================================================================
#                               IO / PARSING
# =============================================================================
def _read_manifest(csv_path: Path) -> List[Segment]:
    segments: List[Segment] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"index", "filename", "start_s", "end_s"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Manifest CSV missing required columns {sorted(required)}. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            segments.append(
                Segment(
                    index=int(row["index"]),
                    filename=str(row["filename"]),
                    start_s=float(row["start_s"]),
                    end_s=float(row["end_s"]),
                )
            )
    return segments


# =============================================================================
#                          TEXTGRID TRANSFORMATIONS
# =============================================================================
def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float, eps: float) -> bool:
    return (a_end > b_start + eps) and (a_start < b_end - eps)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _slice_interval_tier(
    tier: textgrid.IntervalTier,
    *,
    seg_start: float,
    seg_end: float,
    eps: float,
) -> textgrid.IntervalTier:
    new_entries: List[Tuple[float, float, str]] = []
    for start, end, label in tier.entries:
        if not _overlaps(start, end, seg_start, seg_end, eps):
            continue
        clipped_start = _clip(start, seg_start, seg_end)
        clipped_end = _clip(end, seg_start, seg_end)
        if clipped_end <= clipped_start + eps:
            continue
        new_entries.append((clipped_start - seg_start, clipped_end - seg_start, label))
    return textgrid.IntervalTier(name=tier.name, entries=new_entries, minT=0.0, maxT=seg_end - seg_start)


def _slice_point_tier(
    tier: textgrid.PointTier,
    *,
    seg_start: float,
    seg_end: float,
    eps: float,
) -> textgrid.PointTier:
    new_entries: List[Tuple[float, str]] = []
    for t, label in tier.entries:
        if (t >= seg_start - eps) and (t <= seg_end + eps):
            shifted_t = t - seg_start
            if shifted_t < -eps:
                continue
            new_entries.append((shifted_t, label))
    return textgrid.PointTier(name=tier.name, entries=new_entries, minT=0.0, maxT=seg_end - seg_start)


def split_textgrid(
    *,
    merged_textgrid_path: Path,
    manifest_csv: Path,
    out_dir: Path,
    epsilon_s: float = DEFAULT_EPSILON_S,
    output_suffix: str = ".TextGrid",
) -> None:
    """
    Split an annotated merged TextGrid into per-file TextGrids using a manifest CSV.

    Parameters
    ----------
    merged_textgrid_path
        TextGrid created by annotating the merged WAV in Praat.
    manifest_csv
        CSV produced by concat_wavs.py (maps files to time spans in merged WAV).
    out_dir
        Directory to write per-file TextGrid files into.
    epsilon_s
        Small tolerance when deciding overlaps on boundaries.
    output_suffix
        Output extension/suffix, default ".TextGrid".

    Returns
    -------
    None
    """
    merged_textgrid_path = merged_textgrid_path.expanduser().resolve()
    manifest_csv = manifest_csv.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = _read_manifest(manifest_csv)

    tg = textgrid.openTextgrid(str(merged_textgrid_path), includeEmptyIntervals=True)

    for seg in segments:
        seg_tg = textgrid.Textgrid(minT=0.0, maxT=seg.duration_s)

        for tier_name in tg.tierNames:
            tier = tg.getTier(tier_name)

            if isinstance(tier, textgrid.IntervalTier):
                new_tier = _slice_interval_tier(tier, seg_start=seg.start_s, seg_end=seg.end_s, eps=epsilon_s)
            elif isinstance(tier, textgrid.PointTier):
                new_tier = _slice_point_tier(tier, seg_start=seg.start_s, seg_end=seg.end_s, eps=epsilon_s)
            else:
                raise TypeError(f"Unsupported tier type: {type(tier)}")

            seg_tg.addTier(new_tier)

        out_path = out_dir / f"{Path(seg.filename).stem}{output_suffix}"
        seg_tg.save(str(out_path), format="long_textgrid", includeBlankSpaces=True)


# =============================================================================
#                                    CLI
# =============================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Split merged Praat TextGrid into per-file TextGrids.")
    p.add_argument("--merged_tg", type=Path, required=True)
    p.add_argument("--manifest_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--epsilon_s", type=float, default=DEFAULT_EPSILON_S)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    split_textgrid(
        merged_textgrid_path=args.merged_tg,
        manifest_csv=args.manifest_csv,
        out_dir=args.out_dir,
        epsilon_s=args.epsilon_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
