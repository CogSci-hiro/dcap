import numpy as np
from mne import Annotations
from mne.io import RawArray
from mne import create_info

from dcap.bids.tasks.registry import list_tasks
from dcap.bids.tasks.sorciere.alignment import (
    _detect_trigger_onsets,
    estimate_alignment_from_candidates,
    extract_raw_trigger_candidates,
)
from dcap.bids.tasks.sorciere.models import RawTriggerCandidate


def test_sorciere_alignment_applies_annotation_origin_offset() -> None:
    reference_onsets_s = np.array([4.0, 7.5, 11.0, 16.0, 22.0, 29.0])
    reference_intervals_s = np.diff(reference_onsets_s)
    expected_stimulus_start_s = 15.86
    delay_s = expected_stimulus_start_s - 3.0

    raw_candidates = [
        RawTriggerCandidate(
            description="Stimulus/S 11",
            event_code=11,
            onset_samples=np.rint((reference_onsets_s + delay_s) * 1000.0).astype(int),
        ),
        RawTriggerCandidate(
            description="Stimulus/S 99",
            event_code=99,
            onset_samples=np.rint(np.array([2.0, 5.0, 9.0, 14.0, 20.0, 27.0]) * 1000.0).astype(int),
        ),
    ]

    result = estimate_alignment_from_candidates(
        reference_onsets_s=reference_onsets_s[:-1],
        reference_intervals_s=reference_intervals_s,
        raw_candidates=raw_candidates,
        sfreq=1000.0,
        annotation_origin_in_reference_s=3.0,
        reference_duration_s=42.0,
    )

    assert result.selected_event_code == 11
    assert result.selected_description == "Stimulus/S 11"
    assert result.matched_hits >= 5
    assert result.stimulus_start_s == expected_stimulus_start_s


def test_extract_raw_trigger_candidates_groups_annotation_descriptions() -> None:
    info = create_info(ch_names=["SEEG1"], sfreq=1000.0, ch_types=["seeg"])
    raw = RawArray(np.zeros((1, 5000)), info, verbose=False)
    raw.set_annotations(
        Annotations(
            onset=[0.500, 1.500, 2.500, 3.500, 4.500, 0.250],
            duration=[0.0] * 6,
            description=[
                "Stimulus/S 11",
                "Stimulus/S 11",
                "Stimulus/S 11",
                "Stimulus/S 11",
                "Stimulus/S 11",
                "Response/R 1",
            ],
        )
    )

    candidates = extract_raw_trigger_candidates(raw)

    descriptions = {candidate.description for candidate in candidates}
    assert "Stimulus/S 11" in descriptions
    assert "Response/R 1" not in descriptions


def test_task_registry_lists_sorciere() -> None:
    assert "sorciere" in list_tasks()


def test_detect_trigger_onsets_collapses_mp3_chatter() -> None:
    sfreq = 1000.0
    trigger = np.zeros(1000, dtype=float)
    trigger[100:105] = 1.0
    trigger[108:112] = 1.0
    trigger[400:405] = 1.0

    onsets = _detect_trigger_onsets(
        trigger=trigger,
        sfreq=sfreq,
        threshold=0.5,
        min_trigger_gap_s=0.1,
    )

    assert np.allclose(onsets, np.array([0.1, 0.4]))
