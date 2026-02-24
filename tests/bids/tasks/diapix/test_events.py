from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest
from scipy.io import wavfile

from dcap.bids.tasks.diapix import events as diapix_events


def _write_trigger_wav(
    path: Path,
    *,
    onsets_s: list[float],
    sr: int = 1000,
    duration_s: float = 20.0,
    amplitude: int = 20000,
    pulse_len_samples: int = 5,
) -> None:
    n_samples = int(sr * duration_s)
    wav = np.zeros((n_samples, 2), dtype=np.int16)
    for onset_s in onsets_s:
        idx = int(round(onset_s * sr))
        wav[idx: idx + pulse_len_samples, 1] = amplitude
    wavfile.write(path, sr, wav)


def _make_minimal_raw(*, sfreq: float = 1000.0, n_times: int = 20_000) -> mne.io.BaseRaw:
    data = np.zeros((1, n_times), dtype=float)
    info = mne.create_info(ch_names=["SEEG1"], sfreq=sfreq, ch_types=["seeg"])
    return mne.io.RawArray(data, info, verbose=False)


def test_compute_delay_seconds_matches_known_constant_offset(tmp_path: Path) -> None:
    stim_wav = tmp_path / "stim.wav"
    wav_onsets = [0.5, 1.3, 2.0, 3.1, 4.7, 6.0, 7.6]
    _write_trigger_wav(stim_wav, onsets_s=wav_onsets, duration_s=10.0)

    delay_s = 0.237
    raw_samples = np.array([int((t + delay_s) * 1000) for t in wav_onsets], dtype=int)
    raw_trigger_events = np.column_stack(
        [raw_samples, np.zeros_like(raw_samples), np.full_like(raw_samples, 10005)]
    )

    got_delay, got_wav_onsets, got_raw_onsets = diapix_events._compute_delay_seconds(
        raw_trigger_events=raw_trigger_events,
        sfreq=1000.0,
        stim_wav=stim_wav,
    )

    assert got_delay == pytest.approx(delay_s, abs=1e-3)
    assert got_wav_onsets[:5] == pytest.approx(np.asarray(wav_onsets[:-1])[:5], abs=1e-3)
    assert got_raw_onsets[:5] == pytest.approx((np.asarray(wav_onsets) + delay_s)[:-1][:5], abs=1e-3)


def test_compute_delay_seconds_uses_onset_fallback_when_interval_match_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stim_wav = tmp_path / "stim.wav"
    _write_trigger_wav(stim_wav, onsets_s=[0.5, 1.2, 2.0, 3.4, 5.0, 7.2], duration_s=9.0)

    raw_samples = np.array([900, 1600, 2400, 3800, 5400, 7600], dtype=int)
    raw_trigger_events = np.column_stack(
        [raw_samples, np.zeros_like(raw_samples), np.full_like(raw_samples, 10005)]
    )

    def _boom(**_: object) -> float:
        raise ValueError("force fallback")

    monkeypatch.setattr(diapix_events, "_match_intervals_delay_mad", _boom)

    fallback_called = {"value": False}
    real_fallback = diapix_events._estimate_delay_by_onset_hits

    def _fallback_wrapper(**kwargs: object) -> float:
        fallback_called["value"] = True
        return real_fallback(**kwargs)

    monkeypatch.setattr(diapix_events, "_estimate_delay_by_onset_hits", _fallback_wrapper)

    got_delay, _, _ = diapix_events._compute_delay_seconds(
        raw_trigger_events=raw_trigger_events,
        sfreq=1000.0,
        stim_wav=stim_wav,
    )

    assert fallback_called["value"] is True
    assert isinstance(got_delay, float)


def test_prepare_diapix_events_clamps_negative_start_and_reports_padding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = _make_minimal_raw(sfreq=1000.0, n_times=3_000)
    stim_wav = tmp_path / "stim.wav"
    _write_trigger_wav(stim_wav, onsets_s=[0.1, 0.4, 0.9, 1.7, 2.9, 4.4], duration_s=6.0)

    fake_events = np.array([[100, 0, 10005], [300, 0, 10005], [700, 0, 99999]], dtype=int)
    monkeypatch.setattr(
        diapix_events.mne,
        "events_from_annotations",
        lambda *_args, **_kwargs: (fake_events, {"dummy": 1}),
    )
    monkeypatch.setattr(
        diapix_events,
        "_compute_delay_seconds",
        lambda **_kwargs: (-10.0, np.array([0.1, 0.5]), np.array([0.2, 0.6])),
    )

    prepared, alignment = diapix_events.prepare_diapix_events(
        raw=raw,
        subject_bids="001",
        run="1",
        stim_wav=stim_wav,
        trigger_id=10005,
    )

    assert prepared.event_id == {"conversation_start": 1, "conversation_end": 2}
    assert prepared.events is not None
    assert prepared.events.shape == (2, 3)
    assert prepared.events[0, 0] == 0  # clamped from negative start
    assert prepared.events[0, 2] == 1
    assert prepared.events[1, 0] == raw.n_times - 1  # clipped to file end
    assert prepared.events[1, 2] == 2

    assert alignment["delay_s"] == -10.0
    assert alignment["conversation_start_s"] == pytest.approx(-6.0)
    assert alignment["pad_required_s"] == pytest.approx(6.0)
    assert alignment["conversation_window_is_full"] is False


def test_prepare_diapix_events_errors_when_trigger_missing(tmp_path: Path) -> None:
    raw = _make_minimal_raw(sfreq=1000.0, n_times=5_000)
    raw.set_annotations(
        mne.Annotations(onset=[0.5, 1.0], duration=[0.0, 0.0], description=["111", "222"])
    )
    stim_wav = tmp_path / "stim.wav"
    _write_trigger_wav(stim_wav, onsets_s=[0.2, 0.9, 1.8, 2.7, 3.5, 4.4], duration_s=6.0)

    with pytest.raises(ValueError, match="No triggers found"):
        diapix_events.prepare_diapix_events(
            raw=raw,
            subject_bids="001",
            run="1",
            stim_wav=stim_wav,
            trigger_id=10005,
        )
