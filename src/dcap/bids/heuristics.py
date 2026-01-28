# =============================================================================
#                         DCAP: Source discovery heuristics
# =============================================================================
# > Put dataset-specific “how do I find files?” logic here.
# > Keep it deterministic and testable.

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from dcap_bids.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class SourceItem:
    """
    A single source recording to convert.

    Parameters
    ----------
    source_path
        Path to the raw source file or folder.
    kind
        A short string describing the source type (e.g. 'edf', 'brainvision', 'nlx', 'mic').
        This is used to route to the appropriate loader.
    subject_hint
        Optional subject identifier extracted from the filename/folder.
    session_hint
        Optional session identifier extracted from the filename/folder.
    run_hint
        Optional run identifier extracted from the filename/folder.

    Usage example
        item = SourceItem(
            source_path=Path("/data/subj001/run1.edf"),
            kind="edf",
            subject_hint="001",
            session_hint="01",
            run_hint="01",
        )
    """

    source_path: Path
    kind: str
    subject_hint: Optional[str]
    session_hint: Optional[str]
    run_hint: Optional[str]


def discover_source_items(source_root: Path) -> Iterable[SourceItem]:
    """
    Discover source items under a source root.

    This is deliberately simplistic: it searches for known file extensions.
    Replace/extend with your clinical acquisition conventions.

    Parameters
    ----------
    source_root
        Root directory to scan.

    Yields
    ------
    SourceItem
        Discovered items.

    Usage example
        items = list(discover_source_items(Path("./source")))
    """
    if not source_root.exists():
        raise FileNotFoundError(f"source_root does not exist: {source_root}")

    # Extend as needed: .vhdr (BrainVision), .edf, .set (EEGLAB), etc.
    patterns = [
        ("edf", "**/*.edf"),
        ("brainvision", "**/*.vhdr"),
        ("fif", "**/*.fif"),
    ]

    for kind, glob_pattern in patterns:
        for path in sorted(source_root.glob(glob_pattern)):
            yield SourceItem(
                source_path=path,
                kind=kind,
                subject_hint=None,
                session_hint=None,
                run_hint=None,
            )
