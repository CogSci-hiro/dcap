from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_naming_modules():
    repo_root = Path(__file__).resolve().parents[3]
    root = repo_root / "src" / "dcap" / "bids" / "tasks" / "naming"

    for name in ["dcap", "dcap.bids", "dcap.bids.tasks", "dcap.bids.tasks.naming"]:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

    loaded = {}
    for mod_name in ["models", "presentation"]:
        qualified = f"dcap.bids.tasks.naming.{mod_name}"
        spec = importlib.util.spec_from_file_location(qualified, root / f"{mod_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        loaded[mod_name] = module
    return loaded["models"], loaded["presentation"]


def test_extract_marker_trials_handles_missing_and_multi_markers() -> None:
    models, presentation = _load_naming_modules()

    rows = [
        models.MarkerRow(index=1, kind="Stimulus", description="S 78", sample=100, size=1, channel=0, code=78),
        models.MarkerRow(index=2, kind="Stimulus", description="S 100", sample=110, size=1, channel=0, code=100),
        models.MarkerRow(index=3, kind="Stimulus", description="S 126", sample=120, size=1, channel=0, code=126),
        models.MarkerRow(index=4, kind="Stimulus", description="S 78", sample=200, size=1, channel=0, code=78),
        models.MarkerRow(index=5, kind="Stimulus", description="S 126", sample=220, size=1, channel=0, code=126),
        models.MarkerRow(index=6, kind="Stimulus", description="S 78", sample=300, size=1, channel=0, code=78),
        models.MarkerRow(index=7, kind="Stimulus", description="S 254", sample=305, size=1, channel=0, code=254),
        models.MarkerRow(index=8, kind="Stimulus", description="S 30", sample=310, size=1, channel=0, code=30),
        models.MarkerRow(index=9, kind="Stimulus", description="S 126", sample=320, size=1, channel=0, code=126),
        models.MarkerRow(index=10, kind="Stimulus", description="S 98", sample=400, size=1, channel=0, code=98),
        models.MarkerRow(index=11, kind="Stimulus", description="S 100", sample=500, size=1, channel=0, code=100),
    ]

    trials, aux = presentation._extract_marker_trials(rows)

    assert len(trials) == 3
    assert trials[0]["picture_code"] == 100
    assert trials[1]["picture_code"] is None
    assert trials[2]["picture_code"] == 30
    assert len(aux) == 2
    assert [event.event_type for event in aux] == ["button_press", "pause_onset"]


def test_extract_log_trials_parses_mic_picture_isi_triplets() -> None:
    models, presentation = _load_naming_modules()

    rows = [
        models.PresentationLogRow(subject="14", trial=1, event_type="Picture", code="start", time_ms=1000),
        models.PresentationLogRow(subject="14", trial=2, event_type="Sound Recording", code="Mic_Rec", time_ms=51840),
        models.PresentationLogRow(subject="14", trial=3, event_type="Picture", code="miroir.bmp", time_ms=53004),
        models.PresentationLogRow(subject="14", trial=103, event_type="Picture", code="isi", time_ms=69671),
        models.PresentationLogRow(subject="14", trial=104, event_type="Picture", code="pause", time_ms=70000),
        models.PresentationLogRow(subject="14", trial=105, event_type="Response", code="98", time_ms=71000),
    ]

    trials, aux = presentation._extract_log_trials(rows)

    assert len(trials) == 1
    assert trials[0]["stimulus_file"] == "miroir.bmp"
    assert trials[0]["recording_onset_ms"] == 51840
    assert trials[0]["picture_onset_ms"] == 53004
    assert trials[0]["isi_onset_ms"] == 69671
    assert [event.event_type for event in aux] == ["run_start", "pause_onset", "button_press"]
