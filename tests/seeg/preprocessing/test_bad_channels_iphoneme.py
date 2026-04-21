from pathlib import Path

import mne

from dcap.seeg.preprocessing.blocks.bad_channels import _infer_bids_channels_tsv


def test_infer_bids_channels_tsv_for_iphoneme_eeg_run(tmp_path: Path) -> None:
    eeg_dir = tmp_path / "sub-001" / "eeg"
    eeg_dir.mkdir(parents=True)

    raw_path = eeg_dir / "sub-001_task-iphoneme_run-01_eeg.edf"
    raw_path.write_text("", encoding="utf-8")
    channels_path = eeg_dir / "sub-001_task-iphoneme_run-01_channels.tsv"
    channels_path.write_text("name\tstatus\nEEG001\tgood\n", encoding="utf-8")

    raw = mne.io.RawArray([[0.0, 0.0]], mne.create_info(["EEG001"], sfreq=100.0, ch_types=["eeg"]), verbose=False)
    raw._filenames = [str(raw_path)]

    assert _infer_bids_channels_tsv(raw) == channels_path
