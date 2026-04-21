import csv
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Iterable, Optional

from dcap.bids.tasks.naming.models import (
    MarkerRow,
    NamingAlignmentSummary,
    NamingAuxEvent,
    NamingTrial,
    PresentationLogRow,
    ResponseAudioFile,
)


_SEQ_DIR_RE = re.compile(r"^seq_(?P<seq>\d+)_")
_WAV_NAME_RE = re.compile(r"^seq_(?P<seq>\d+)_(?P<stimulus>.+)-001-(?P<stamp>\d{14})$")

_MM_DD_YYYY_HH_MM_SS = "%m/%d/%Y %H:%M:%S"
_PRESENTATION_TICKS_PER_SECOND = 10_000.0


def infer_sequence_id_from_log_dir(log_dir: Path) -> str:
    match = _SEQ_DIR_RE.match(log_dir.name)
    if match is None:
        raise ValueError(f"Could not infer Presentation sequence id from log directory: {log_dir}")
    return match.group("seq")


def parse_presentation_log(log_path: Path) -> tuple[datetime, list[PresentationLogRow]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Presentation logfile not found: {log_path}")

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 5:
        raise ValueError(f"Unexpectedly short Presentation logfile: {log_path}")

    written_prefix = "Logfile written - "
    if not lines[1].startswith(written_prefix):
        raise ValueError(f"Could not find logfile timestamp in {log_path}")
    written_at = datetime.strptime(lines[1][len(written_prefix):].strip(), _MM_DD_YYYY_HH_MM_SS)

    rows: list[PresentationLogRow] = []
    for line in lines[4:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        rows.append(
            PresentationLogRow(
                subject=str(parts[0]).strip(),
                trial=int(parts[1]),
                event_type=str(parts[2]).strip(),
                code=str(parts[3]).strip(),
                time_ms=int(parts[4]),
            )
        )
    return written_at, rows


def parse_sequence_files(presentation_root: Path, sequence_id: str) -> tuple[list[str], list[int]]:
    seq_path = presentation_root / "sequences" / f"seq_{sequence_id}.txt"
    trig_path = presentation_root / "sequences" / f"trig_seq_{sequence_id}.txt"

    if not seq_path.exists():
        raise FileNotFoundError(f"Stimulus sequence file not found: {seq_path}")
    if not trig_path.exists():
        raise FileNotFoundError(f"Trigger sequence file not found: {trig_path}")

    stimuli = [line.strip() for line in seq_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    triggers = [int(line.strip()) for line in trig_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if len(stimuli) != len(triggers):
        raise ValueError(
            f"Stimulus/trigger sequence length mismatch for sequence {sequence_id}: "
            f"{len(stimuli)} stimuli vs {len(triggers)} triggers."
        )
    return stimuli, triggers


def build_stimulus_catalog(presentation_root: Path) -> dict[str, int]:
    seq_dir = presentation_root / "sequences"
    names: set[str] = set()
    for seq_path in sorted(seq_dir.glob("seq_*.txt")):
        for line in seq_path.read_text(encoding="utf-8", errors="replace").splitlines():
            cleaned = normalize_stimulus_id(line)
            if cleaned:
                names.add(cleaned)

    if len(names) == 0:
        raise ValueError(f"No stimuli found under {seq_dir}")

    return {stimulus: idx for idx, stimulus in enumerate(sorted(names), start=1)}


def parse_response_audio_files(log_dir: Path, sequence_id: str) -> dict[str, ResponseAudioFile]:
    mapping: dict[str, ResponseAudioFile] = {}
    for wav_path in sorted(log_dir.glob("*.wav")):
        match = _WAV_NAME_RE.match(wav_path.stem)
        if match is None:
            continue
        if match.group("seq") != str(sequence_id):
            continue

        stimulus_id = normalize_stimulus_id(match.group("stimulus"))
        timestamp = datetime.strptime(match.group("stamp"), "%Y%m%d%H%M%S")
        if stimulus_id in mapping:
            raise ValueError(f"Duplicate response WAV for stimulus {stimulus_id!r} in {log_dir}")
        mapping[stimulus_id] = ResponseAudioFile(stimulus_id=stimulus_id, path=wav_path, timestamp=timestamp)
    return mapping


def parse_brainvision_markers(vmrk_path: Path) -> list[MarkerRow]:
    if not vmrk_path.exists():
        raise FileNotFoundError(f"BrainVision marker file not found: {vmrk_path}")

    rows: list[MarkerRow] = []
    for line in vmrk_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("Mk") or "=" not in line:
            continue
        left, right = line.split("=", 1)
        parts = [part.strip() for part in right.split(",")]
        if len(parts) < 5:
            continue

        description = parts[1]
        code = None
        if description.startswith("S "):
            try:
                code = int(description.split()[1])
            except ValueError:
                code = None

        rows.append(
            MarkerRow(
                index=int(left[2:]),
                kind=parts[0],
                description=description,
                sample=int(parts[2]),
                size=int(parts[3]),
                channel=int(parts[4]),
                code=code,
            )
        )
    return rows


def build_naming_alignment(
    *,
    presentation_root: Path,
    log_dir: Path,
    log_path: Path,
    vmrk_path: Path,
    sfreq: float,
) -> tuple[list[NamingTrial], list[NamingAuxEvent], NamingAlignmentSummary]:
    sequence_id = infer_sequence_id_from_log_dir(log_dir)
    _, log_rows = parse_presentation_log(log_path)
    sequence_stimuli, sequence_triggers = parse_sequence_files(presentation_root, sequence_id=sequence_id)
    stimulus_catalog = build_stimulus_catalog(presentation_root)
    response_audio = parse_response_audio_files(log_dir, sequence_id=sequence_id)
    markers = parse_brainvision_markers(vmrk_path)

    log_trials, _ = _extract_log_trials(log_rows)
    marker_trials, marker_aux = _extract_marker_trials(markers)

    if len(log_trials) != len(sequence_stimuli):
        raise ValueError(
            f"Logfile contains {len(log_trials)} trials but Presentation sequence {sequence_id} "
            f"contains {len(sequence_stimuli)} stimuli."
        )
    if len(marker_trials) != len(log_trials):
        raise ValueError(
            f"BrainVision markers contain {len(marker_trials)} naming trials but logfile contains {len(log_trials)}."
        )

    _validate_log_matches_sequence(log_trials, sequence_stimuli)

    trials: list[NamingTrial] = []
    recording_offset_s: list[float] = []
    rec_to_pic_errors_s: list[float] = []
    pic_to_isi_errors_s: list[float] = []
    wav_timestamps: list[datetime] = []
    n_code_mismatches = 0
    n_interpolated_picture_onsets = 0

    for idx, (log_trial, marker_trial, stimulus_file, sequence_trigger) in enumerate(
        zip(log_trials, marker_trials, sequence_stimuli, sequence_triggers),
        start=1,
    ):
        stimulus_id = normalize_stimulus_id(stimulus_file)
        expected_recorded_code = int(sequence_trigger) - 1
        picture_sample = marker_trial["picture_sample"]
        recorded_picture_code = marker_trial["picture_code"]
        if picture_sample is None:
            picture_sample = int(
                round(
                    marker_trial["recording_sample"]
                    + (_ticks_to_seconds(log_trial["picture_onset_ms"] - log_trial["recording_onset_ms"]) * sfreq)
                )
            )
            n_interpolated_picture_onsets += 1

        if recorded_picture_code != expected_recorded_code:
            n_code_mismatches += 1

        wav = response_audio.get(stimulus_id)
        if wav is not None:
            wav_timestamps.append(wav.timestamp)

        recording_s = marker_trial["recording_sample"] / sfreq
        picture_s = picture_sample / sfreq
        isi_s = marker_trial["isi_sample"] / sfreq

        recording_offset_s.append(recording_s - _ticks_to_seconds(log_trial["recording_onset_ms"]))
        rec_to_pic_errors_s.append(
            (picture_s - recording_s)
            - _ticks_to_seconds(log_trial["picture_onset_ms"] - log_trial["recording_onset_ms"])
        )
        pic_to_isi_errors_s.append(
            (isi_s - picture_s)
            - _ticks_to_seconds(log_trial["isi_onset_ms"] - log_trial["picture_onset_ms"])
        )

        trials.append(
            NamingTrial(
                trial_index=idx,
                stimulus_id=stimulus_id,
                stimulus_file=stimulus_file,
                sequence_trigger_code=int(sequence_trigger),
                recorded_picture_code=None if recorded_picture_code is None else int(recorded_picture_code),
                stimulus_catalog_id=int(stimulus_catalog[stimulus_id]),
                recording_onset_ms=int(log_trial["recording_onset_ms"]),
                picture_onset_ms=int(log_trial["picture_onset_ms"]),
                isi_onset_ms=int(log_trial["isi_onset_ms"]),
                recording_sample=int(marker_trial["recording_sample"]),
                picture_sample=int(picture_sample),
                isi_sample=int(marker_trial["isi_sample"]),
                response_audio_path=wav.path if wav is not None else None,
                response_audio_timestamp=wav.timestamp if wav is not None else None,
            )
        )

    summary = NamingAlignmentSummary(
        sequence_id=sequence_id,
        n_trials=len(trials),
        log_marker_offset_s=float(median(recording_offset_s)),
        recording_to_picture_mae_s=_mean_abs(rec_to_pic_errors_s),
        picture_to_isi_mae_s=_mean_abs(pic_to_isi_errors_s),
        wav_matches_all_trials=all(trial.response_audio_path is not None for trial in trials),
        wav_order_is_monotonic=_is_monotonic(wav_timestamps),
        n_pause_events=sum(1 for event in marker_aux if event.event_type == "pause_onset"),
        n_button_presses=sum(1 for event in marker_aux if event.event_type == "button_press"),
        n_recorded_code_mismatches=n_code_mismatches,
        n_interpolated_picture_onsets=n_interpolated_picture_onsets,
    )
    return trials, marker_aux, summary


def make_bids_event_rows(
    *,
    trials: list[NamingTrial],
    aux_events: list[NamingAuxEvent],
    sfreq: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for trial in trials:
        rows.append(
            {
                "onset": _format_seconds(trial.recording_sample / sfreq),
                "duration": _format_seconds(0.0),
                "trial_type": "recording_onset",
                "value": 1,
                "stimulus_id": trial.stimulus_id,
                "stimulus_catalog_id": trial.stimulus_catalog_id,
                "stimulus_file": trial.stimulus_file,
                "sequence_trigger_code": trial.sequence_trigger_code,
                "recorded_marker_code": 78,
                "trial_index": trial.trial_index,
                "response_audio_file": "",
            }
        )
        rows.append(
            {
                "onset": _format_seconds(trial.picture_sample / sfreq),
                "duration": _format_seconds(0.0),
                "trial_type": "picture_onset",
                "value": 2,
                "stimulus_id": trial.stimulus_id,
                "stimulus_catalog_id": trial.stimulus_catalog_id,
                "stimulus_file": trial.stimulus_file,
                "sequence_trigger_code": trial.sequence_trigger_code,
                "recorded_marker_code": "" if trial.recorded_picture_code is None else trial.recorded_picture_code,
                "trial_index": trial.trial_index,
                "response_audio_file": "",
            }
        )
        rows.append(
            {
                "onset": _format_seconds(trial.isi_sample / sfreq),
                "duration": _format_seconds(0.0),
                "trial_type": "isi_onset",
                "value": 3,
                "stimulus_id": trial.stimulus_id,
                "stimulus_catalog_id": trial.stimulus_catalog_id,
                "stimulus_file": trial.stimulus_file,
                "sequence_trigger_code": trial.sequence_trigger_code,
                "recorded_marker_code": 126,
                "trial_index": trial.trial_index,
                "response_audio_file": "",
            }
        )

    for aux in aux_events:
        value = 4 if aux.event_type == "pause_onset" else 5 if aux.event_type == "button_press" else 6
        rows.append(
            {
                "onset": _format_seconds(aux.sample / sfreq),
                "duration": _format_seconds(0.0),
                "trial_type": aux.event_type,
                "value": value,
                "stimulus_id": "",
                "stimulus_catalog_id": "",
                "stimulus_file": "",
                "sequence_trigger_code": "",
                "recorded_marker_code": "" if aux.marker_code is None else aux.marker_code,
                "trial_index": "",
                "response_audio_file": "",
            }
        )

    rows.sort(key=lambda row: float(row["onset"]))
    return rows


def write_events_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "onset",
        "duration",
        "trial_type",
        "value",
        "stimulus_id",
        "stimulus_catalog_id",
        "stimulus_file",
        "sequence_trigger_code",
        "recorded_marker_code",
        "trial_index",
        "response_audio_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_stimulus_lookup_tsv(path: Path, trials: Iterable[NamingTrial]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique_rows: dict[str, dict[str, object]] = {}
    for trial in trials:
        unique_rows.setdefault(
            trial.stimulus_id,
            {
                "stimulus_catalog_id": trial.stimulus_catalog_id,
                "stimulus_id": trial.stimulus_id,
                "example_stimulus_file": trial.stimulus_file,
            },
        )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stimulus_catalog_id", "stimulus_id", "example_stimulus_file"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in sorted(unique_rows.values(), key=lambda item: int(item["stimulus_catalog_id"])):
            writer.writerow(row)


def write_alignment_summary_json(path: Path, summary: NamingAlignmentSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")


def normalize_stimulus_id(name: str) -> str:
    value = str(name).strip()
    if value.endswith(".bmp"):
        value = value[:-4]
    return value


def _extract_log_trials(rows: list[PresentationLogRow]) -> tuple[list[dict[str, object]], list[NamingAuxEvent]]:
    trials: list[dict[str, object]] = []
    aux: list[NamingAuxEvent] = []

    idx = 0
    while idx < len(rows):
        row = rows[idx]
        if row.event_type == "Sound Recording" and row.code == "Mic_Rec":
            if idx + 2 >= len(rows):
                raise ValueError("Incomplete log trial at end of Presentation logfile.")
            picture = rows[idx + 1]
            isi = rows[idx + 2]
            if picture.event_type != "Picture" or not picture.code.endswith(".bmp"):
                raise ValueError(f"Expected picture after Mic_Rec, found {picture.event_type}/{picture.code!r}")
            if isi.event_type != "Picture" or isi.code != "isi":
                raise ValueError(f"Expected isi after picture, found {isi.event_type}/{isi.code!r}")

            trials.append(
                {
                    "recording_onset_ms": row.time_ms,
                    "picture_onset_ms": picture.time_ms,
                    "isi_onset_ms": isi.time_ms,
                    "stimulus_file": picture.code,
                }
            )
            idx += 3
            continue

        if row.event_type == "Picture" and row.code == "pause":
            aux.append(NamingAuxEvent(event_type="pause_onset", onset_ms=row.time_ms, sample=-1, marker_code=100))
        elif row.event_type == "Response":
            aux.append(NamingAuxEvent(event_type="button_press", onset_ms=row.time_ms, sample=-1, marker_code=int(row.code)))
        elif row.event_type == "Picture" and row.code == "start":
            aux.append(NamingAuxEvent(event_type="run_start", onset_ms=row.time_ms, sample=-1, marker_code=None))
        idx += 1

    return trials, aux


def _extract_marker_trials(rows: list[MarkerRow]) -> tuple[list[dict[str, Optional[int]]], list[NamingAuxEvent]]:
    stimulus_rows = [row for row in rows if row.kind == "Stimulus" and row.code is not None]

    trials: list[dict[str, Optional[int]]] = []
    consumed: set[int] = set()
    for idx, row in enumerate(stimulus_rows):
        if row.code != 78:
            continue

        isi_idx = next(
            (j for j in range(idx + 1, len(stimulus_rows)) if stimulus_rows[j].code == 126),
            None,
        )
        if isi_idx is None:
            continue

        picture_candidates = [
            j
            for j in range(idx + 1, isi_idx)
            if stimulus_rows[j].code not in {78, 98, 126}
        ]
        if len(picture_candidates) == 0:
            trials.append(
                {
                    "recording_sample": stimulus_rows[idx].sample,
                    "picture_sample": None,
                    "picture_code": None,
                    "isi_sample": stimulus_rows[isi_idx].sample,
                }
            )
            consumed.update({idx, isi_idx})
            continue

        picture_idx = picture_candidates[-1]
        trials.append(
            {
                "recording_sample": stimulus_rows[idx].sample,
                "picture_sample": stimulus_rows[picture_idx].sample,
                "picture_code": int(stimulus_rows[picture_idx].code),
                "isi_sample": stimulus_rows[isi_idx].sample,
            }
        )
        consumed.update({idx, picture_idx, isi_idx})

    aux: list[NamingAuxEvent] = []
    for idx, row in enumerate(stimulus_rows):
        if idx in consumed:
            continue
        if row.code == 98:
            aux.append(NamingAuxEvent(event_type="button_press", onset_ms=-1, sample=row.sample, marker_code=row.code))
        elif row.code == 100:
            aux.append(NamingAuxEvent(event_type="pause_onset", onset_ms=-1, sample=row.sample, marker_code=row.code))

    return trials, aux


def _validate_log_matches_sequence(log_trials: list[dict[str, object]], sequence_stimuli: list[str]) -> None:
    observed = [str(row["stimulus_file"]) for row in log_trials]
    if observed != sequence_stimuli:
        for idx, (left, right) in enumerate(zip(observed, sequence_stimuli), start=1):
            if left != right:
                raise ValueError(
                    f"Presentation logfile sequence diverges from seq file at item {idx}: "
                    f"log has {left!r}, sequence file has {right!r}."
                )
        raise ValueError("Presentation logfile sequence length mismatch with sequence file.")


def _mean_abs(values: Iterable[float]) -> float:
    values_list = [abs(float(value)) for value in values]
    if len(values_list) == 0:
        return 0.0
    return float(sum(values_list) / len(values_list))


def _is_monotonic(values: list[datetime]) -> bool:
    if len(values) < 2:
        return True
    return all(left <= right for left, right in zip(values, values[1:]))


def _format_seconds(value: float) -> str:
    return f"{float(value):.6f}"


def _ticks_to_seconds(value: int) -> float:
    return float(value) / _PRESENTATION_TICKS_PER_SECOND
