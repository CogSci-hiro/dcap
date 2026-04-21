from pathlib import Path

from dcap.cli.commands.preprocess import _iter_runs_from_cfg, _resolve_out_dir
from dcap.seeg.preprocessing.pipelines.standard import StandardPipelineConfig


def _make_cfg(bids_root: Path, *, tasks: list[str] | None = None) -> StandardPipelineConfig:
    return StandardPipelineConfig(
        raw={
            "selection": {
                "scope": "dataset",
                "bids_root": str(bids_root),
                "include": {"tasks": tasks, "subjects": None, "sessions": None, "runs": None},
                "exclude": {},
            },
            "io": {"derivatives_name": "preproc"},
        }
    )


def test_iter_runs_from_cfg_discovers_iphoneme_under_eeg(tmp_path: Path) -> None:
    eeg_dir = tmp_path / "sub-001" / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    eeg_path = eeg_dir / "sub-001_ses-01_task-iphoneme_run-01_eeg.edf"
    eeg_path.write_text("", encoding="utf-8")

    runs = list(_iter_runs_from_cfg(_make_cfg(tmp_path)))

    assert len(runs) == 1
    run = runs[0]
    assert run.subject == "001"
    assert run.session == "01"
    assert run.task == "iphoneme"
    assert run.run == "01"
    assert run.datatype == "eeg"
    assert run.in_path == eeg_path


def test_iter_runs_from_cfg_can_filter_to_iphoneme(tmp_path: Path) -> None:
    eeg_dir = tmp_path / "sub-001" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-001_task-iphoneme_run-01_eeg.edf").write_text("", encoding="utf-8")
    (eeg_dir / "sub-001_task-conversation_run-01_eeg.edf").write_text("", encoding="utf-8")

    runs = list(_iter_runs_from_cfg(_make_cfg(tmp_path, tasks=["iphoneme"])))

    assert [run.task for run in runs] == ["iphoneme"]


def test_resolve_out_dir_preserves_eeg_datatype(tmp_path: Path) -> None:
    eeg_dir = tmp_path / "sub-001" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-001_task-iphoneme_run-01_eeg.edf").write_text("", encoding="utf-8")

    cfg = _make_cfg(tmp_path)
    run = list(_iter_runs_from_cfg(cfg))[0]

    out_dir = _resolve_out_dir(cfg=cfg, run_spec=run)

    assert out_dir == tmp_path / "derivatives" / "preproc" / "sub-001" / "eeg"
