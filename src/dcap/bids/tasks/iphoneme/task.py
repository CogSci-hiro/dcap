from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import TYPE_CHECKING, Optional, Sequence
import csv
import json
import re

import mne

from dcap.bids.tasks.base import BidsTask, PreparedEvents, RecordingUnit
from dcap.seeg.io.sidecars import find_neighbor_sidecar

if TYPE_CHECKING:  # pragma: no cover
    from mne_bids import BIDSPath


_PRIMARY_EXTENSIONS = (".vhdr", ".edf")
_SIDELOAD_EXTENSIONS = (".eeg", ".dat", ".vmrk", ".avg")
_NON_IPHONEM_HINTS = ("rest", "resting")
_BEHAVIOR_FILE_SUFFIXES = (".wav", ".log", ".sce", ".pcl", ".pk", ".txt", ".exp")
_DEFAULT_EVENT_DURATION_S = 0.0
_TRIGGER_EVENT_FIELDS = ("event_name", "frequency_hz", "condition_level", "event_group", "notes")
_CODE_TO_PART = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
_PART_FROM_LOG_RE = re.compile(r"part(?P<part>[A-Za-z0-9]+)", re.IGNORECASE)
_WAV_WITH_PART_RE = re.compile(
    r".*Iphonem_part(?P<part>[A-Za-z0-9]+)_(?P<stimulus>\d+)-\d+-\d{14}$",
    re.IGNORECASE,
)
_WAV_SIMPLE_RE = re.compile(r"(?P<stimulus>\d+)-\d+-\d{14}$")
IPHONEME_EVENTS_TSV_COLUMNS = (
    "onset",
    "duration",
    "trial_type",
    "value",
    "event_type",
    "block",
    "frequency_hz",
    "condition_level",
    "event_group",
    "stimulus_id",
    "response_audio_file",
    "source_log",
    "source_time_ms",
)


@dataclass(frozen=True)
class IphonemeRecordingUnit(RecordingUnit):
    datatype: str
    source_stem: str
    behavior_dir: Optional[Path]
    subject_stimuli_dir: Optional[Path]
    notes_path: Optional[Path]


class IphonemeTask(BidsTask):
    """
    Converter for the current raw iPhoneme subject-folder layout.

    This task is intentionally conservative:
    - it treats each primary recording file in the subject root as one run
    - it prefers BrainVision headers when markers are available
    - it falls back to EDF when the dataset flags missing markers or only EDF is present
    - it writes a fixed, behavior-enriched `events.tsv` when marker and log alignment succeeds
    - it falls back to marker-only events when richer alignment is not available
    """

    name: str = "iphoneme"

    def __init__(
        self,
        *,
        bids_subject: str,
        dcap_id: str,
        session: Optional[str],
    ) -> None:
        self._bids_subject = str(bids_subject).strip()
        self._dcap_id = str(dcap_id).strip()
        self._session = None if session in (None, "") else str(session).strip()
        self._behavior_exports_written = False
        self._events_cache: dict[str, list[dict[str, object]]] = {}

    def discover(self, source_root: Path) -> Sequence[RecordingUnit]:
        source_root = Path(source_root).expanduser().resolve()
        dataset_root = source_root.parent
        overrides = _load_subject_overrides(dataset_root=dataset_root, subject_id=source_root.name)

        corrected_root = source_root / "with_Correct_Electrodes_Labels"
        candidate_roots: list[Path] = []
        if overrides.use_corrected_labels and corrected_root.exists():
            candidate_roots.append(corrected_root)
        candidate_roots.append(source_root)

        primary_files = _discover_primary_recordings(
            candidate_roots=candidate_roots,
            marker_missing=overrides.marker_missing,
        )
        if len(primary_files) == 0:
            raise FileNotFoundError(
                f"No primary iphoneme recording found in {source_root}. "
                "Expected BrainVision .vhdr or EDF exports in the subject root."
            )

        behavior_dir = source_root / "behavior"
        stimuli_dir = source_root / "stimuli"
        notes_path = source_root / "notes.txt"

        units: list[IphonemeRecordingUnit] = []
        for index, raw_path in enumerate(primary_files, start=1):
            units.append(
                IphonemeRecordingUnit(
                    run=f"{index:02d}",
                    raw_path=raw_path,
                    audio_path=None,
                    video_path=None,
                    datatype="eeg",
                    source_stem=_strip_all_suffixes(raw_path.name),
                    behavior_dir=behavior_dir if behavior_dir.exists() else None,
                    subject_stimuli_dir=stimuli_dir if stimuli_dir.exists() else None,
                    notes_path=notes_path if notes_path.exists() else None,
                )
            )

        return units

    def load_raw(self, unit: RecordingUnit, preload: bool) -> mne.io.BaseRaw:
        suffix = unit.raw_path.suffix.lower()
        if suffix == ".vhdr":
            return mne.io.read_raw_brainvision(unit.raw_path, preload=preload, verbose=False)
        if suffix == ".edf":
            return mne.io.read_raw_edf(unit.raw_path, preload=preload, verbose=False)
        raise ValueError(f"Unsupported iphoneme raw format: {unit.raw_path}")

    def prepare_events(self, raw: mne.io.BaseRaw, unit: RecordingUnit, bids_path: "BIDSPath") -> PreparedEvents:
        if isinstance(unit, IphonemeRecordingUnit):
            self._events_cache[str(unit.raw_path)] = _build_iphoneme_event_rows(raw=raw, unit=unit)
        return PreparedEvents(events=None, event_id=None)

    def post_write(self, unit: RecordingUnit, bids_path: "BIDSPath") -> None:
        _copy_neighbor_sidecar(unit.raw_path, bids_path=bids_path, sidecar_suffix="_events.tsv")
        _copy_neighbor_sidecar(unit.raw_path, bids_path=bids_path, sidecar_suffix="_channels.tsv")
        if isinstance(unit, IphonemeRecordingUnit) and not self._behavior_exports_written:
            _export_behavior_artifacts(unit=unit, bids_path=bids_path)
            self._behavior_exports_written = True
        if isinstance(unit, IphonemeRecordingUnit):
            _write_behavior_enriched_events_tsv(
                bids_path=bids_path,
                event_rows=self._events_cache.get(str(unit.raw_path), []),
            )


@dataclass(frozen=True)
class _SubjectOverrides:
    use_corrected_labels: bool = False
    marker_missing: bool = False


@dataclass(frozen=True)
class _MarkerEvent:
    code: int
    sample: int


@dataclass(frozen=True)
class _BehaviorLogRow:
    event_type: str
    code: str
    time_ms: int
    duration_ms: Optional[int]


@dataclass(frozen=True)
class _BehaviorLog:
    path: Path
    part: str
    rows: tuple[_BehaviorLogRow, ...]


def _discover_primary_recordings(*, candidate_roots: Sequence[Path], marker_missing: bool) -> list[Path]:
    by_stem: dict[str, dict[str, Path]] = {}
    seen_paths: set[Path] = set()

    for root in candidate_roots:
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in _PRIMARY_EXTENSIONS + _SIDELOAD_EXTENSIONS:
                continue
            if _looks_non_iphonem(path.name):
                continue

            stem = _strip_all_suffixes(path.name)
            by_stem.setdefault(stem, {})
            by_stem[stem][suffix] = path.resolve()

    selected: list[Path] = []
    for stem, files in sorted(by_stem.items()):
        vhdr_path = files.get(".vhdr")
        vmrk_path = files.get(".vmrk")
        edf_path = files.get(".edf")

        chosen: Optional[Path] = None
        if vhdr_path is not None and (vmrk_path is not None or not marker_missing):
            chosen = vhdr_path
        elif edf_path is not None:
            chosen = edf_path
        elif vhdr_path is not None:
            chosen = vhdr_path

        if chosen is None or chosen in seen_paths:
            continue
        seen_paths.add(chosen)
        selected.append(chosen)

    return selected


def _looks_non_iphonem(filename: str) -> bool:
    name = filename.lower()
    for token in _NON_IPHONEM_HINTS:
        if token in name:
            return True
    return False


def _load_subject_overrides(*, dataset_root: Path, subject_id: str) -> _SubjectOverrides:
    overrides_path = dataset_root / "config" / "subject_overrides.csv"
    if not overrides_path.exists():
        return _SubjectOverrides()

    use_corrected_labels = False
    marker_missing = False

    with overrides_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("subject_id", "")).strip() != subject_id:
                continue
            rule_type = str(row.get("rule_type", "")).strip()
            override_value = str(row.get("override_value", "")).strip().lower()
            if rule_type == "use_corrected_labels" and override_value == "yes":
                use_corrected_labels = True
            if rule_type == "marker_availability" and override_value == "missing":
                marker_missing = True

    return _SubjectOverrides(
        use_corrected_labels=use_corrected_labels,
        marker_missing=marker_missing,
    )


def _build_iphoneme_event_rows(*, raw: mne.io.BaseRaw, unit: IphonemeRecordingUnit) -> list[dict[str, object]]:
    trigger_map = _load_trigger_code_map(dataset_root=_infer_dataset_root(unit))
    markers = _parse_brainvision_marker_events(unit.raw_path)
    if len(markers) == 0:
        return []

    behavior_logs = _load_behavior_logs(unit.behavior_dir)
    response_audio_index = _index_behavior_response_audio(unit.behavior_dir)
    block_starts = _extract_block_start_markers(markers)

    behavior_rows: list[dict[str, object]] = []
    if len(block_starts) > 0 and len(behavior_logs) > 0:
        behavior_rows = _build_behavior_aligned_rows(
            block_starts=block_starts,
            behavior_logs=behavior_logs,
            response_audio_index=response_audio_index,
            sfreq=float(raw.info["sfreq"]),
            trigger_map=trigger_map,
        )

    if len(behavior_rows) > 0:
        return behavior_rows

    return _build_marker_fallback_rows(
        markers=markers,
        sfreq=float(raw.info["sfreq"]),
        trigger_map=trigger_map,
    )


def _infer_dataset_root(unit: IphonemeRecordingUnit) -> Path:
    if unit.behavior_dir is not None:
        return unit.behavior_dir.parent.parent
    if unit.subject_stimuli_dir is not None:
        return unit.subject_stimuli_dir.parent.parent
    return unit.raw_path.parent.parent


def _load_trigger_code_map(*, dataset_root: Path) -> dict[int, dict[str, str]]:
    trigger_map_path = dataset_root / "config" / "trigger_code_map.csv"
    out: dict[int, dict[str, str]] = {}
    if not trigger_map_path.exists():
        return out

    with trigger_map_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                code = int(str(row.get("code", "")).strip())
            except ValueError:
                continue
            out[code] = {field: str(row.get(field, "")).strip() for field in _TRIGGER_EVENT_FIELDS}
    return out


def _parse_brainvision_marker_events(raw_path: Path) -> list[_MarkerEvent]:
    vmrk_path = raw_path if raw_path.suffix.lower() == ".vmrk" else raw_path.with_suffix(".vmrk")
    if not vmrk_path.exists():
        return []

    markers: list[_MarkerEvent] = []
    pattern = re.compile(r"^Mk\d+=(?P<kind>[^,]+),(?P<desc>[^,]*),(?P<sample>\d+),")
    for line in vmrk_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if match is None:
            continue
        code = _extract_marker_code(match.group("desc").strip())
        if code is None:
            continue
        markers.append(_MarkerEvent(code=code, sample=int(match.group("sample"))))
    return markers


def _extract_marker_code(description: str) -> Optional[int]:
    desc = str(description).strip()
    if not desc.startswith("S"):
        return None
    digits = "".join(ch for ch in desc if ch.isdigit())
    if digits == "":
        return None
    return int(digits)


def _extract_block_start_markers(markers: Sequence[_MarkerEvent]) -> list[tuple[str, _MarkerEvent]]:
    out: list[tuple[str, _MarkerEvent]] = []
    for marker in markers:
        part = _CODE_TO_PART.get(marker.code)
        if part is not None:
            out.append((part, marker))
    return out


def _load_behavior_logs(behavior_dir: Optional[Path]) -> list[_BehaviorLog]:
    if behavior_dir is None or not behavior_dir.exists():
        return []

    logs: list[_BehaviorLog] = []
    for log_path in sorted(behavior_dir.glob("*.log")):
        part = _infer_part_from_log_name(log_path.name)
        if part is None:
            continue
        rows = _parse_behavior_log(log_path)
        if len(rows) == 0:
            continue
        logs.append(_BehaviorLog(path=log_path, part=part, rows=tuple(rows)))
    return logs


def _infer_part_from_log_name(name: str) -> Optional[str]:
    match = _PART_FROM_LOG_RE.search(name)
    if match is None:
        return None
    return match.group("part").upper()


def _parse_behavior_log(log_path: Path) -> list[_BehaviorLogRow]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = None
    for idx, line in enumerate(lines):
        if "Event Type" in line and "Code" in line and "Time" in line:
            header_index = idx
            break
    if header_index is None:
        return []

    header = [cell.strip() for cell in lines[header_index].split("\t")]
    rows: list[_BehaviorLogRow] = []
    for line in lines[header_index + 1:]:
        if not line.strip():
            continue
        parts = [cell.strip() for cell in line.split("\t")]
        if len(parts) < len(header):
            parts.extend([""] * (len(header) - len(parts)))
        row = dict(zip(header, parts))
        event_type = row.get("Event Type", "")
        time_raw = row.get("Time", "")
        if event_type == "" or time_raw == "":
            continue
        try:
            time_ms = int(time_raw)
        except ValueError:
            continue
        duration_ms = None
        duration_raw = row.get("Duration", "")
        if duration_raw not in ("", None):
            try:
                duration_ms = int(duration_raw)
            except ValueError:
                duration_ms = None
        rows.append(
            _BehaviorLogRow(
                event_type=event_type,
                code=row.get("Code", ""),
                time_ms=time_ms,
                duration_ms=duration_ms,
            )
        )
    return rows


def _index_behavior_response_audio(behavior_dir: Optional[Path]) -> dict[tuple[str, str], list[Path]]:
    out: dict[tuple[str, str], list[Path]] = {}
    if behavior_dir is None or not behavior_dir.exists():
        return out

    for wav_path in sorted(behavior_dir.glob("*.wav")):
        stem = wav_path.stem
        match = _WAV_WITH_PART_RE.match(stem)
        if match is not None:
            out.setdefault((match.group("part").upper(), match.group("stimulus")), []).append(wav_path)
            continue
        match = _WAV_SIMPLE_RE.search(stem)
        if match is not None:
            out.setdefault(("", match.group("stimulus")), []).append(wav_path)
    return out


def _build_behavior_aligned_rows(
    *,
    block_starts: Sequence[tuple[str, _MarkerEvent]],
    behavior_logs: Sequence[_BehaviorLog],
    response_audio_index: dict[tuple[str, str], list[Path]],
    sfreq: float,
    trigger_map: dict[int, dict[str, str]],
) -> list[dict[str, object]]:
    logs_by_part = {log.part: log for log in behavior_logs}
    rows: list[dict[str, object]] = []

    for part, marker in block_starts:
        log = logs_by_part.get(part)
        if log is None:
            continue
        block_onset_s = float(max(marker.sample - 1, 0)) / sfreq
        for log_row in log.rows:
            event_code, stimulus_id = _parse_behavior_code(log_row.code)
            event_metadata = trigger_map.get(event_code, {}) if event_code is not None else {}
            response_audio_file = ""
            if stimulus_id is not None:
                candidates = response_audio_index.get((part, stimulus_id), [])
                if len(candidates) == 0:
                    candidates = response_audio_index.get(("", stimulus_id), [])
                if len(candidates) == 1:
                    response_audio_file = f"behavior/{candidates[0].name}"

            row: dict[str, object] = {
                "onset": block_onset_s + (float(log_row.time_ms) / 1000.0),
                "duration": _DEFAULT_EVENT_DURATION_S if log_row.duration_ms is None else float(log_row.duration_ms) / 1000.0,
                "trial_type": _behavior_trial_type(log_row.event_type, event_metadata),
                "value": "" if event_code is None else int(event_code),
                "event_type": log_row.event_type,
                "block": part,
                "frequency_hz": event_metadata.get("frequency_hz", ""),
                "condition_level": event_metadata.get("condition_level", ""),
                "event_group": event_metadata.get("event_group", ""),
                "stimulus_id": "" if stimulus_id is None else stimulus_id,
                "response_audio_file": response_audio_file,
                "source_log": log.path.name,
                "source_time_ms": int(log_row.time_ms),
            }
            rows.append(row)

    return rows


def _behavior_trial_type(event_type: str, event_metadata: dict[str, str]) -> str:
    normalized = event_type.strip().lower()
    if normalized == "sound":
        return event_metadata.get("event_name", "stimulus") or "stimulus"
    if normalized == "sound recording":
        return "response_audio"
    if normalized == "response":
        return "button_response"
    if normalized == "picture":
        code_name = event_metadata.get("event_name", "").strip()
        return code_name if code_name else "picture"
    return normalized.replace(" ", "_")


def _parse_behavior_code(code_text: str) -> tuple[Optional[int], Optional[str]]:
    text = str(code_text).strip()
    if text == "":
        return None, None
    if text.startswith("r_"):
        text = text[2:]
    parts = text.split()

    event_code = None
    stimulus_id = None
    if len(parts) >= 1:
        try:
            event_code = int(parts[0])
        except ValueError:
            event_code = None
    if len(parts) >= 2:
        stimulus_id = parts[1]
    return event_code, stimulus_id


def _build_marker_fallback_rows(
    *,
    markers: Sequence[_MarkerEvent],
    sfreq: float,
    trigger_map: dict[int, dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for marker in markers:
        metadata = trigger_map.get(marker.code, {})
        rows.append(
            {
                "onset": float(max(marker.sample - 1, 0)) / sfreq,
                "duration": _DEFAULT_EVENT_DURATION_S,
                "trial_type": metadata.get("event_name", "marker") or "marker",
                "value": int(marker.code),
                "event_type": "marker",
                "block": _CODE_TO_PART.get(marker.code, ""),
                "frequency_hz": metadata.get("frequency_hz", ""),
                "condition_level": metadata.get("condition_level", ""),
                "event_group": metadata.get("event_group", ""),
                "stimulus_id": "",
                "response_audio_file": "",
                "source_log": "",
                "source_time_ms": "",
            }
        )
    return rows


def _write_behavior_enriched_events_tsv(
    *,
    bids_path: "BIDSPath",
    event_rows: Sequence[dict[str, object]],
) -> None:
    if len(event_rows) == 0:
        return
    if bids_path.directory is None or bids_path.basename is None:
        raise ValueError(f"Could not resolve BIDS events destination for {bids_path}")

    events_path = bids_path.directory / f"{bids_path.basename}_events.tsv"
    with events_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=IPHONEME_EVENTS_TSV_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in event_rows:
            writer.writerow({key: row.get(key, "") for key in IPHONEME_EVENTS_TSV_COLUMNS})


def _copy_neighbor_sidecar(raw_path: Path, *, bids_path: "BIDSPath", sidecar_suffix: str) -> None:
    source_sidecar = find_neighbor_sidecar(raw_path, sidecar_suffix=sidecar_suffix)
    if source_sidecar is None:
        return

    if bids_path.directory is None or bids_path.basename is None:
        raise ValueError(f"Could not resolve BIDS sidecar destination for {bids_path}")

    bids_path.directory.mkdir(parents=True, exist_ok=True)
    destination = bids_path.directory / f"{bids_path.basename}{sidecar_suffix}"
    copy2(source_sidecar, destination)


def _export_behavior_artifacts(*, unit: IphonemeRecordingUnit, bids_path: "BIDSPath") -> None:
    if unit.behavior_dir is None or not unit.behavior_dir.exists():
        return

    behavior_files = [
        path for path in sorted(unit.behavior_dir.iterdir())
        if path.is_file() and path.suffix.lower() in _BEHAVIOR_FILE_SUFFIXES
    ]
    if len(behavior_files) == 0:
        return

    deriv_root = bids_path.root / "derivatives" / "dcap" / _normalize_subject_label(bids_path.subject)
    if bids_path.session is not None:
        deriv_root = deriv_root / f"ses-{bids_path.session}"
    behavior_out_dir = deriv_root / "behavior"
    behavior_out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    for source_path in behavior_files:
        destination = behavior_out_dir / source_path.name
        copy2(source_path, destination)
        manifest_rows.append(
            {
                "source_file": str(source_path),
                "copied_file": str(destination.relative_to(deriv_root)),
                "kind": _classify_behavior_file(source_path),
                "run_assignment": "subject_level_unassigned",
            }
        )

    manifest_path = deriv_root / "iphoneme_behavior_manifest.json"
    manifest_payload = {
        "task": "iphoneme",
        "subject": _normalize_subject_label(bids_path.subject),
        "session": None if bids_path.session is None else f"ses-{bids_path.session}",
        "notes": (
            "Behavior files are preserved as subject-level derivatives. "
            "They are not yet aligned to BIDS runs or trials."
        ),
        "files": manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")


def _classify_behavior_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "response_audio"
    if suffix == ".log":
        return "presentation_log"
    if suffix in {".sce", ".pcl", ".exp"}:
        return "presentation_script"
    return "behavior_aux"


def _normalize_subject_label(subject: Optional[str]) -> str:
    if subject is None:
        return "sub-unknown"
    value = str(subject).strip()
    return value if value.startswith("sub-") else f"sub-{value}"


def _strip_all_suffixes(name: str) -> str:
    base = name
    while True:
        suffix = Path(base).suffix
        if suffix == "":
            return base
        base = Path(base).stem
