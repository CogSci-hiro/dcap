from __future__ import annotations

from pathlib import Path

import pytest

from dcap.bids.tasks.base import discover_recording_units_from_task_layout
from dcap.bids.tasks.diapix.task import DiapixTask
from dcap.bids.tasks.naming.task import NamingTask
from dcap.bids.tasks.sorciere.task import SorciereTask


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_diapix_discovery_two_runs_uses_dcap_subject_dir(tmp_path: Path) -> None:
    source_root = tmp_path / "sourcedata" / "subjects"
    dcap_id = "Nic-Ele"

    _touch(source_root / dcap_id / "task-diapix" / "run-01" / "eeg.vhdr")
    _touch(source_root / dcap_id / "task-diapix" / "run-01" / "audio.wav")
    _touch(source_root / dcap_id / "task-diapix" / "run-01" / "video.mp4")
    _touch(source_root / dcap_id / "task-diapix" / "run-01" / "annotations.asf")
    _touch(source_root / dcap_id / "task-diapix" / "run-02" / "eeg.edf")
    _touch(source_root / dcap_id / "task-diapix" / "run-02" / "audio.wav")

    assets = tmp_path / "assets"
    task = DiapixTask(
        bids_subject="sub-001",
        dcap_id=dcap_id,
        session=None,
        audio_onsets_tsv=_touch(assets / "audio_onsets.tsv"),
        stim_wav=_touch(assets / "stim.wav"),
        atlas_path=_touch(assets / "elec2atlas.mat"),
    )

    units = list(task.discover(source_root))

    assert [u.run for u in units] == ["01", "02"]
    assert units[0].eeg_path.name == "eeg.vhdr"
    assert units[0].audio_path is not None and units[0].audio_path.name == "audio.wav"
    assert units[0].video_path is not None and units[0].video_path.name == "video.mp4"
    assert units[0].annotation_path is not None and units[0].annotation_path.name == "annotations.asf"
    assert units[1].eeg_path.name == "eeg.edf"


def test_sorciere_discovery_one_run(tmp_path: Path) -> None:
    source_root = tmp_path / "sourcedata" / "subjects"
    dcap_id = "Pat-002"
    _touch(source_root / dcap_id / "task-sorciere" / "run-01" / "eeg.fif")

    task = SorciereTask(
        bids_subject="sub-002",
        dcap_id=dcap_id,
        session=None,
        stim_wav=_touch(tmp_path / "stimuli" / "sorciere.wav"),
    )

    units = list(task.discover(source_root))

    assert len(units) == 1
    assert units[0].run == "01"
    assert units[0].eeg_path.name == "eeg.fif"


def test_naming_discovery_one_run_with_mic_wav(tmp_path: Path) -> None:
    source_root = tmp_path / "sourcedata" / "subjects"
    dcap_id = "Patient-A"
    run_dir = source_root / dcap_id / "task-naming" / "run-01"
    _touch(run_dir / "eeg.vhdr")
    _touch(run_dir / "mic.wav")
    _touch(run_dir / "timing.tsv")

    task = NamingTask(bids_subject="sub-010", dcap_id=dcap_id, session=None)
    units = list(task.discover(source_root))

    assert len(units) == 1
    assert units[0].run == "01"
    assert units[0].audio_path is not None and units[0].audio_path.name == "mic.wav"
    assert units[0].annotation_path is not None and units[0].annotation_path.name == "timing.tsv"


def test_discovery_error_when_run_has_zero_eeg_files(tmp_path: Path) -> None:
    source_root = tmp_path / "sourcedata" / "subjects"
    run_dir = source_root / "Nic-Ele" / "task-diapix" / "run-01"
    _touch(run_dir / "audio.wav")

    with pytest.raises(FileNotFoundError, match="exactly one EEG file"):
        discover_recording_units_from_task_layout(
            source_root=source_root,
            bids_subject="sub-001",
            task_name="diapix",
            source_subject_id="Nic-Ele",
        )


def test_discovery_error_when_run_has_two_eeg_files(tmp_path: Path) -> None:
    source_root = tmp_path / "sourcedata" / "subjects"
    run_dir = source_root / "Nic-Ele" / "task-diapix" / "run-01"
    _touch(run_dir / "eeg.vhdr")
    _touch(run_dir / "eeg.edf")

    with pytest.raises(ValueError, match="exactly one EEG file"):
        discover_recording_units_from_task_layout(
            source_root=source_root,
            bids_subject="sub-001",
            task_name="diapix",
            source_subject_id="Nic-Ele",
        )


def test_discovery_error_when_task_has_no_run_folder(tmp_path: Path) -> None:
    source_root = tmp_path / "sourcedata" / "subjects"
    (source_root / "Nic-Ele" / "task-diapix").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No run-\\* folders"):
        discover_recording_units_from_task_layout(
            source_root=source_root,
            bids_subject="sub-001",
            task_name="diapix",
            source_subject_id="Nic-Ele",
        )
