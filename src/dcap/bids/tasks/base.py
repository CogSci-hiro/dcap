# =============================================================================
#                           BIDS: Task interface
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

import mne
import numpy as np
from mne_bids import BIDSPath


@dataclass(frozen=True)
class RecordingUnit:
    """
    One convertable unit (typically one run).

    Usage example
    -------------
        unit = RecordingUnit(
            subject="sub-001",
            task="diapix",
            run="01",
            eeg_path=Path("run-01/eeg.vhdr"),
            audio_path=Path("run-01/audio.wav"),
            video_path=Path("run-01/video.mp4"),
            annotation_path=None,
            run_dir=Path("run-01"),
        )
    """
    subject: str
    task: str
    run: str
    eeg_path: Path
    audio_path: Optional[Path]
    video_path: Optional[Path]
    annotation_path: Optional[Path]
    run_dir: Path

    @property
    def raw_path(self) -> Path:
        """Backward-compatible alias for task adapters not yet migrated."""
        return self.eeg_path


@dataclass(frozen=True)
class PreparedEvents:
    """
    Events prepared for MNE-BIDS.

    Usage example
    -------------
        prepared = PreparedEvents(events=events, event_id=event_id)
    """
    events: Optional[np.ndarray]
    event_id: Optional[dict[str, int]]


class BidsTask(Protocol):
    """
    Protocol for task-specific conversion logic.

    A task implementation provides discovery, raw loading, and optional hooks.
    """

    name: str  # e.g., "diapix"

    def discover(self, source_root: Path) -> Sequence[RecordingUnit]:
        ...

    def load_raw(self, unit: RecordingUnit, preload: bool) -> mne.io.BaseRaw:
        ...

    def prepare_events(self, raw: mne.io.BaseRaw, unit: RecordingUnit, bids_path: BIDSPath) -> PreparedEvents:
        ...

    def post_write(self, unit: RecordingUnit, bids_path: BIDSPath) -> None:
        ...


SUPPORTED_EEG_SUFFIXES = (".vhdr", ".edf", ".fif")
_AUDIO_SUFFIXES = (".wav",)
_VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".asf", ".mkv")
_ANNOTATION_SUFFIXES = (".tsv", ".csv", ".json", ".txt", ".asf")


def discover_recording_units_from_task_layout(
    *,
    source_root: Path,
    bids_subject: str,
    task_name: str,
    source_subject_id: Optional[str] = None,
) -> list[RecordingUnit]:
    """
    Discover runs from sourcedata/subjects/<source-subject>/task-YYY/run-ZZ layout.

    Raises descriptive errors for structural violations (missing task folder,
    no run folders, missing/ambiguous EEG).
    """
    source_root = Path(source_root).expanduser().resolve()
    subject_label = str(bids_subject).strip()
    task_label = str(task_name).strip()

    source_subject_dir_name = _normalize_source_subject_dir_name(source_subject_id, fallback_bids_subject=subject_label)
    subject_dir = source_root / source_subject_dir_name
    task_dir = subject_dir / _normalize_task_dir_name(task_label)
    run_dirs = _validate_task_layout(task_dir=task_dir, subject=subject_label, task=task_label)

    units: list[RecordingUnit] = []
    for run_dir in run_dirs:
        eeg_path = _select_exactly_one_eeg(run_dir, subject=subject_label, task=task_label)
        run_label = _parse_run_label(run_dir.name)

        audio_candidates = _discover_audio_candidates(run_dir, exclude={eeg_path})
        video_candidates = _discover_video_candidates(run_dir, exclude={eeg_path})
        annotation_candidates = _discover_annotation_candidates(
            run_dir,
            exclude={eeg_path, *audio_candidates, *video_candidates},
        )

        units.append(
            RecordingUnit(
                subject=_normalize_subject_dir_name(subject_label),
                task=task_label,
                run=run_label,
                eeg_path=eeg_path,
                audio_path=_select_optional_single(audio_candidates, kind="audio", run_dir=run_dir),
                video_path=_select_optional_single(video_candidates, kind="video", run_dir=run_dir),
                annotation_path=_select_optional_single(annotation_candidates, kind="annotation", run_dir=run_dir),
                run_dir=run_dir,
            )
        )

    return units


def _normalize_subject_dir_name(subject: str) -> str:
    s = str(subject).strip()
    return s if s.startswith("sub-") else f"sub-{s}"


def _normalize_source_subject_dir_name(source_subject_id: Optional[str], *, fallback_bids_subject: str) -> str:
    if source_subject_id is None or str(source_subject_id).strip() == "":
        return _normalize_subject_dir_name(fallback_bids_subject)
    return str(source_subject_id).strip()


def _normalize_task_dir_name(task: str) -> str:
    t = str(task).strip()
    return t if t.startswith("task-") else f"task-{t}"


def _validate_task_layout(*, task_dir: Path, subject: str, task: str) -> list[Path]:
    if not task_dir.exists():
        raise FileNotFoundError(
            f"Task folder not found for subject={subject!r}, task={task!r}: {task_dir}"
        )
    if not task_dir.is_dir():
        raise NotADirectoryError(f"Task path is not a directory: {task_dir}")

    run_dirs = sorted(
        p for p in task_dir.iterdir()
        if p.is_dir() and p.name.startswith("run-")
    )
    if len(run_dirs) == 0:
        raise FileNotFoundError(
            f"No run-* folders found in task folder for subject={subject!r}, task={task!r}: {task_dir}"
        )

    for run_dir in run_dirs:
        _ = _select_exactly_one_eeg(run_dir, subject=subject, task=task)

    return run_dirs


def _parse_run_label(run_dir_name: str) -> str:
    if not run_dir_name.startswith("run-"):
        raise ValueError(f"Run directory must start with 'run-': {run_dir_name}")
    run_label = run_dir_name[len("run-"):].strip()
    if run_label == "":
        raise ValueError(f"Run directory has empty run label: {run_dir_name}")
    return run_label


def _select_exactly_one_eeg(run_dir: Path, *, subject: str, task: str) -> Path:
    eeg_candidates = sorted(
        p for p in run_dir.iterdir()
        if p.is_file() and not _is_hidden_sidecar(p) and p.suffix.lower() in SUPPORTED_EEG_SUFFIXES
    )
    if len(eeg_candidates) == 0:
        raise FileNotFoundError(
            f"Run folder must contain exactly one EEG file ({', '.join(SUPPORTED_EEG_SUFFIXES)}); "
            f"found none in {run_dir} (subject={subject!r}, task={task!r})"
        )
    if len(eeg_candidates) > 1:
        names = ", ".join(p.name for p in eeg_candidates)
        raise ValueError(
            f"Run folder must contain exactly one EEG file; found {len(eeg_candidates)} in {run_dir}: {names} "
            f"(subject={subject!r}, task={task!r})"
        )
    return eeg_candidates[0]


def _gather_by_suffix(paths: Iterable[Path], suffixes: tuple[str, ...], *, exclude: set[Path]) -> list[Path]:
    return [
        p for p in sorted(paths)
        if p.is_file() and not _is_hidden_sidecar(p) and p not in exclude and p.suffix.lower() in suffixes
    ]


def _discover_audio_candidates(run_dir: Path, *, exclude: set[Path]) -> list[Path]:
    return _gather_by_suffix(run_dir.iterdir(), _AUDIO_SUFFIXES, exclude=exclude)


def _discover_video_candidates(run_dir: Path, *, exclude: set[Path]) -> list[Path]:
    candidates = _gather_by_suffix(run_dir.iterdir(), _VIDEO_SUFFIXES, exclude=exclude)
    return [p for p in candidates if not _looks_like_annotation_file(p)]


def _discover_annotation_candidates(run_dir: Path, *, exclude: set[Path]) -> list[Path]:
    candidates = [
        p for p in sorted(run_dir.iterdir())
        if p.is_file() and not _is_hidden_sidecar(p) and p not in exclude and p.suffix.lower() in _ANNOTATION_SUFFIXES
    ]
    if len(candidates) <= 1:
        return candidates

    explicit = [p for p in candidates if _looks_like_annotation_file(p)]
    if len(explicit) == 1:
        return explicit
    return candidates


def _looks_like_annotation_file(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.startswith("annotation") or stem.startswith("annotations") or stem == "timing"


def _is_hidden_sidecar(path: Path) -> bool:
    name = path.name
    return name.startswith("._") or (name.startswith(".") and name not in {".", ".."})


def _select_optional_single(candidates: list[Path], *, kind: str, run_dir: Path) -> Optional[Path]:
    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return candidates[0]
    names = ", ".join(p.name for p in candidates)
    raise ValueError(f"Expected at most one {kind} file in {run_dir}, found {len(candidates)}: {names}")
