# =============================================================================
#                         BIDS: Discovery heuristics
# =============================================================================
#
# - Find source recordings (per run)
# - Parse run numbers and pair associated audio/video if present
# - Keep this deterministic and testable
#
# REVIEW
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import re


@dataclass(frozen=True)
class SourceItem:
    """
    A single source recording unit to convert (typically one run).

    Parameters
    ----------
    run
        Run label (string, e.g. "1" or "01").
    raw_vhdr
        Path to the BrainVision header file for the run, if present.
    audio_wav
        Path to the run's WAV file, if present.
    video_asf
        Path to the run's ASF file, if present.

    Usage example
    -------------
        item = SourceItem(
            run="1",
            raw_vhdr=Path("conversation_1.vhdr"),
            audio_wav=Path("conversation_1.wav"),
            video_asf=None,
        )
    """

    run: str
    raw_vhdr: Optional[Path]
    audio_wav: Optional[Path]
    video_asf: Optional[Path]


_RUN_RE = re.compile(r"^conversation_(?P<run>\d+)\.(?P<ext>vhdr|wav|asf)$")


def discover_source_items(source_root: Path) -> Iterable[SourceItem]:
    """
    Discover runs under a subject source directory.

    Notes
    -----
    This assumes a flat structure like:
      conversation_<RUN>.vhdr
      conversation_<RUN>.wav
      conversation_<RUN>.asf

    Parameters
    ----------
    source_root
        Directory containing source files for one subject.

    Yields
    ------
    SourceItem
        One item per run, with optional paired audio/video.

    Usage example
    -------------
        items = list(discover_source_items(Path("sourcedata/Nic-Ele")))
    """
    if not source_root.exists():
        raise FileNotFoundError(f"source_root does not exist: {source_root}")

    runs: dict[str, dict[str, Path]] = {}

    for path in sorted(source_root.iterdir()):
        if not path.is_file():
            continue

        match = _RUN_RE.match(path.name)
        if match is None:
            continue

        run = match.group("run")
        ext = match.group("ext")

        if run not in runs:
            runs[run] = {}
        runs[run][ext] = path

    for run, files in sorted(runs.items(), key=lambda x: int(x[0])):
        yield SourceItem(
            run=run,
            raw_vhdr=files.get("vhdr"),
            audio_wav=files.get("wav"),
            video_asf=files.get("asf"),
        )
