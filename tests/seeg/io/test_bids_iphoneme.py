from pathlib import Path
from types import SimpleNamespace

import mne
import pandas as pd

from dcap.seeg.io.bids import _discover_runs_with_extensions, _load_events_df_from_bids


def test_discover_runs_with_extensions_supports_eeg_iphoneme(tmp_path: Path) -> None:
    eeg_dir = tmp_path / "sub-001" / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-001_ses-01_task-iphoneme_run-01_eeg.vhdr").write_text("", encoding="utf-8")

    discovered = _discover_runs_with_extensions(
        bids_root=tmp_path,
        subject="sub-001",
        session="ses-01",
        task="iphoneme",
    )

    assert discovered == {"01": ("eeg", ".vhdr")}


def test_load_events_df_from_bids_finds_neighboring_iphoneme_events(tmp_path: Path) -> None:
    eeg_dir = tmp_path / "sub-001" / "eeg"
    eeg_dir.mkdir(parents=True)
    recording_path = eeg_dir / "sub-001_task-iphoneme_run-01_raw.fif"
    recording_path.write_text("", encoding="utf-8")

    events_path = eeg_dir / "sub-001_task-iphoneme_run-01_events.tsv"
    pd.DataFrame(
        {"onset": [0.0, 1.0], "duration": [0.1, 0.2], "trial_type": ["phoneme_a", "phoneme_b"]}
    ).to_csv(events_path, sep="\t", index=False)

    raw = mne.io.RawArray([[0.0, 0.0]], mne.create_info(["EEG001"], sfreq=100.0, ch_types=["eeg"]), verbose=False)
    df = _load_events_df_from_bids(
        bids_path=SimpleNamespace(fpath=recording_path),
        fallback_raw=raw,
    )

    assert list(df.columns) == ["onset_sec", "duration_sec", "event_type"]
    assert df["event_type"].tolist() == ["phoneme_a", "phoneme_b"]
