from pathlib import Path
from types import SimpleNamespace

import mne

from dcap.bids.tasks.iphoneme.task import IPHONEME_EVENTS_TSV_COLUMNS, IphonemeTask


def _write_subject_overrides(dataset_root: Path, rows: str) -> None:
    config_dir = dataset_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "subject_overrides.csv").write_text(
        "subject_id,rule_type,target,override_value,notes\n" + rows,
        encoding="utf-8",
    )


def test_iphoneme_task_discovers_subject_root_brainvision_recordings(tmp_path: Path) -> None:
    dataset_root = tmp_path / "iPhoneme"
    subject_root = dataset_root / "Nak-Ot"
    subject_root.mkdir(parents=True)
    _write_subject_overrides(dataset_root, "")

    (subject_root / "21_NakOt_iphonem.vhdr").write_text("", encoding="utf-8")
    (subject_root / "21_NakOt_iphonem.vmrk").write_text("", encoding="utf-8")
    (subject_root / "21_NakOt_iphonem-Code4.vhdr").write_text("", encoding="utf-8")
    (subject_root / "21_NakOt_iphonem-Code4.vmrk").write_text("", encoding="utf-8")
    (subject_root / "21_NakOt_RestingYO.vhdr").write_text("", encoding="utf-8")
    (subject_root / "21_NakOt_RestingYO.vmrk").write_text("", encoding="utf-8")
    (subject_root / "behavior").mkdir()
    (subject_root / "stimuli").mkdir()

    task = IphonemeTask(bids_subject="sub-001", dcap_id="Nak-Ot", session=None)
    units = task.discover(subject_root)

    assert [unit.raw_path.name for unit in units] == [
        "21_NakOt_iphonem.vhdr",
        "21_NakOt_iphonem-Code4.vhdr",
    ]
    assert [unit.run for unit in units] == ["01", "02"]


def test_iphoneme_task_uses_corrected_labels_when_override_requests_it(tmp_path: Path) -> None:
    dataset_root = tmp_path / "iPhoneme"
    subject_root = dataset_root / "Pru-Da"
    corrected_root = subject_root / "with_Correct_Electrodes_Labels"
    corrected_root.mkdir(parents=True)
    subject_root.mkdir(parents=True, exist_ok=True)
    _write_subject_overrides(
        dataset_root,
        "Pru-Da,use_corrected_labels,electrodes,yes,Use corrected labels\n",
    )

    (subject_root / "Pru_Da_iPhoneme.vhdr").write_text("", encoding="utf-8")
    (subject_root / "Pru_Da_iPhoneme.vmrk").write_text("", encoding="utf-8")
    (corrected_root / "200917B-B-mod.vhdr").write_text("", encoding="utf-8")
    (corrected_root / "200917B-B-mod.vmrk").write_text("", encoding="utf-8")

    task = IphonemeTask(bids_subject="sub-001", dcap_id="Pru-Da", session=None)
    units = task.discover(subject_root)

    assert units[0].raw_path == (corrected_root / "200917B-B-mod.vhdr").resolve()


def test_iphoneme_task_falls_back_to_edf_when_markers_are_missing(tmp_path: Path) -> None:
    dataset_root = tmp_path / "iPhoneme"
    subject_root = dataset_root / "Bal-Ca"
    subject_root.mkdir(parents=True)
    _write_subject_overrides(
        dataset_root,
        "Bal-Ca,marker_availability,vmrk,missing,Missing vmrk sidecar\n",
    )

    (subject_root / "210402C-B_0000.edf").write_text("", encoding="utf-8")
    (subject_root / "210402C-B_0000.EEG").write_text("", encoding="utf-8")

    task = IphonemeTask(bids_subject="sub-001", dcap_id="Bal-Ca", session=None)
    units = task.discover(subject_root)

    assert [unit.raw_path.name for unit in units] == ["210402C-B_0000.edf"]


def test_iphoneme_task_post_write_copies_neighbor_sidecars(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True)
    raw_path = source_dir / "21_DavTo_iPhoneme.vhdr"
    raw_path.write_text("", encoding="utf-8")
    (source_dir / "21_DavTo_iPhoneme_events.tsv").write_text("onset\tduration\ttrial_type\n", encoding="utf-8")
    (source_dir / "21_DavTo_iPhoneme_channels.tsv").write_text("name\tstatus\nCz\tgood\n", encoding="utf-8")

    bids_dir = tmp_path / "bids" / "sub-001" / "eeg"
    bids_dir.mkdir(parents=True)
    bids_path = SimpleNamespace(
        directory=bids_dir,
        basename="sub-001_task-iphoneme_run-01_eeg",
    )

    task = IphonemeTask(bids_subject="sub-001", dcap_id="Dav-To", session=None)
    unit = task.discover(source_dir)[0]
    task.post_write(unit, bids_path)

    assert (bids_dir / "sub-001_task-iphoneme_run-01_eeg_events.tsv").exists()
    assert (bids_dir / "sub-001_task-iphoneme_run-01_eeg_channels.tsv").exists()


def test_iphoneme_task_post_write_exports_behavior_audio_and_logs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "iPhoneme"
    subject_root = dataset_root / "Dav-To"
    behavior_dir = subject_root / "behavior"
    behavior_dir.mkdir(parents=True)
    _write_subject_overrides(dataset_root, "")

    (subject_root / "21_DavTo_iPhoneme.vhdr").write_text("", encoding="utf-8")
    (subject_root / "21_DavTo_iPhoneme.vmrk").write_text("", encoding="utf-8")
    (behavior_dir / "s_8_210413110623_Iphonem_partA.log").write_text("log", encoding="utf-8")
    (behavior_dir / "s_8_Iphonem_partA_266186-001-20210413111025.wav").write_text("wav", encoding="utf-8")

    bids_dir = tmp_path / "bids" / "sub-001" / "eeg"
    bids_dir.mkdir(parents=True)
    bids_path = SimpleNamespace(
        root=tmp_path / "bids",
        subject="001",
        session=None,
        directory=bids_dir,
        basename="sub-001_task-iphoneme_run-01_eeg",
    )

    task = IphonemeTask(bids_subject="sub-001", dcap_id="Dav-To", session=None)
    unit = task.discover(subject_root)[0]
    task.post_write(unit, bids_path)

    behavior_out_dir = tmp_path / "bids" / "derivatives" / "dcap" / "sub-001" / "behavior"
    assert (behavior_out_dir / "s_8_210413110623_Iphonem_partA.log").exists()
    assert (behavior_out_dir / "s_8_Iphonem_partA_266186-001-20210413111025.wav").exists()

    manifest_path = tmp_path / "bids" / "derivatives" / "dcap" / "sub-001" / "iphoneme_behavior_manifest.json"
    assert manifest_path.exists()
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "subject_level_unassigned" in manifest_text
    assert "response_audio" in manifest_text


def test_iphoneme_task_writes_events_tsv_from_markers_and_logs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "iPhoneme"
    subject_root = dataset_root / "Dav-To"
    behavior_dir = subject_root / "behavior"
    behavior_dir.mkdir(parents=True)
    _write_subject_overrides(dataset_root, "")
    (dataset_root / "config").mkdir(parents=True, exist_ok=True)
    (dataset_root / "config" / "trigger_code_map.csv").write_text(
        "\n".join(
            [
                "code,event_name,frequency_hz,condition_level,event_group,notes",
                "31,stimulus,3,low,stimulus,",
                "77,recording_prompt,,,control,",
                "99,button_response,,,response,",
                "111,isi_start,,,control,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (subject_root / "21_DavTo_iPhoneme.vhdr").write_text("", encoding="utf-8")
    (subject_root / "21_DavTo_iPhoneme.vmrk").write_text(
        "\n".join(
            [
                "BrainVision Data Exchange Marker File Version 1.0",
                "[Common Infos]",
                "DataFile=21_DavTo_iPhoneme.eeg",
                "[Marker Infos]",
                "Mk1=Stimulus,S  1,1000,1,0",
                "Mk2=Stimulus,S 99,2000,1,0",
                "Mk3=Stimulus,S111,2500,1,0",
                "Mk4=Stimulus,S 31,3000,1,0",
                "Mk5=Stimulus,S 77,4000,1,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (behavior_dir / "s_8_210413110623_Iphonem_partA.log").write_text(
        "\n".join(
            [
                "Scenario - Iphonem_partA",
                "Logfile written - 04/13/2021 11:15:26",
                "",
                "Trial\tEvent Type\tCode\tTime\tTTime\tUncertainty\tDuration\tUncertainty\tReqTime\tReqDur\tStim Type\tPair Index",
                "1\tResponse\t99\t1000\t0\t1",
                "2\tPicture\tisi\t1500\t0\t1\t500\t1\t0\t5000\tother\t0",
                "3\tSound\t31 174183\t2000\t0\t1",
                "4\tSound Recording\tr_31 174183\t3000\t0\t1\t35000\t1\t0\t35000\tother\t0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (behavior_dir / "s_8_Iphonem_partA_174183-001-20210413111025.wav").write_text("wav", encoding="utf-8")

    task = IphonemeTask(bids_subject="sub-001", dcap_id="Dav-To", session=None)
    unit = task.discover(subject_root)[0]
    raw = mne.io.RawArray([[0.0] * 10000], mne.create_info(["EEG001"], sfreq=1000.0, ch_types=["eeg"]), verbose=False)
    task.prepare_events(raw, unit, SimpleNamespace())

    bids_dir = tmp_path / "bids" / "sub-001" / "eeg"
    bids_dir.mkdir(parents=True)
    bids_path = SimpleNamespace(
        root=tmp_path / "bids",
        subject="001",
        session=None,
        directory=bids_dir,
        basename="sub-001_task-iphoneme_run-01_eeg",
    )
    task.post_write(unit, bids_path)

    events_path = bids_dir / "sub-001_task-iphoneme_run-01_eeg_events.tsv"
    assert events_path.exists()
    text = events_path.read_text(encoding="utf-8")
    header = text.splitlines()[0].split("\t")
    assert header == list(IPHONEME_EVENTS_TSV_COLUMNS)
    assert "trial_type" in text
    assert "button_response" in text
    assert "stimulus" in text
    assert "response_audio" in text
    assert "behavior/s_8_Iphonem_partA_174183-001-20210413111025.wav" in text
