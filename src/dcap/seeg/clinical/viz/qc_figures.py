# =============================================================================
# =============================================================================
#                     ########################################
#                     #        CLINICAL QC FIGURES           #
#                     ########################################
# =============================================================================
# =============================================================================

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import mne


_PSD_FMIN_HZ: float = 0.5
_PSD_FMAX_HZ: float = 250.0
_PSD_N_FFT: int = 4096

_TIMESERIES_DURATION_SEC: float = 30.0
_TIMESERIES_N_CHANNELS: int = 12
_TIMESERIES_DOWNSAMPLE_MAX_HZ: float = 300.0


def make_qc_figures(
    *,
    out_dir: Path,
    raw_original: mne.io.BaseRaw,
    raw_analysis: mne.io.BaseRaw,
    analysis_view_name: str,
    subject_id: str,
    session_id: Optional[str],
    run_id: Optional[str],
) -> Dict[str, str]:
    """
    Save a minimal set of clinician-friendly QC figures (PNG).

    Figures
    -------
    - PSD: original vs analysis view
    - Time-series: original vs analysis view (a handful of channels)

    Returns
    -------
    fig_paths
        Mapping figure_key -> file path (as str).

    Usage example
    -------------
        fig_paths = make_qc_figures(
            out_dir=Path("./out"),
            raw_original=views["original"],
            raw_analysis=views["car"],
            analysis_view_name="car",
            subject_id="sub-001",
            session_id="ses-01",
            run_id="run-1",
        )
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _make_base_name(subject_id=subject_id, session_id=session_id, run_id=run_id)

    psd_path = out_dir / f"{base}_qc_psd_original-vs-{analysis_view_name}.png"
    ts_path = out_dir / f"{base}_qc_timeseries_original-vs-{analysis_view_name}.png"

    _save_psd_comparison(
        path=psd_path,
        raw_a=raw_original,
        label_a="original",
        raw_b=raw_analysis,
        label_b=analysis_view_name,
    )
    _save_timeseries_comparison(
        path=ts_path,
        raw_a=raw_original,
        label_a="original",
        raw_b=raw_analysis,
        label_b=analysis_view_name,
    )

    return {
        "psd_original_vs_analysis": str(psd_path),
        "timeseries_original_vs_analysis": str(ts_path),
    }


def _make_base_name(*, subject_id: str, session_id: Optional[str], run_id: Optional[str]) -> str:
    parts = [subject_id]
    if session_id:
        parts.append(session_id)
    if run_id:
        parts.append(run_id)
    return "_".join(parts)


def _pick_seeg_like_channels(raw: mne.io.BaseRaw) -> np.ndarray:
    """
    Pick channels for QC plots. Prefer sEEG/ECoG, exclude ECG/stim/misc.
    """
    picks = mne.pick_types(
        raw.info,
        seeg=True,
        ecog=True,
        eeg=False,
        meg=False,
        stim=False,
        misc=False,
        eog=False,
        ecg=False,
    )
    return picks


def _save_psd_comparison(
    *,
    path: Path,
    raw_a: mne.io.BaseRaw,
    label_a: str,
    raw_b: mne.io.BaseRaw,
    label_b: str,
) -> None:
    picks_a = _pick_seeg_like_channels(raw_a)
    picks_b = _pick_seeg_like_channels(raw_b)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for raw, label, picks in [(raw_a, label_a, picks_a), (raw_b, label_b, picks_b)]:
        if picks.size == 0:
            ax.text(0.5, 0.5, f"No sEEG/ECoG channels for {label}", ha="center", va="center")
            continue

        data = raw.get_data(picks=picks)
        sfreq = float(raw.info["sfreq"])

        psd, freqs = mne.time_frequency.psd_array_welch(
            data,
            sfreq=sfreq,
            fmin=_PSD_FMIN_HZ,
            fmax=_PSD_FMAX_HZ,
            n_fft=_PSD_N_FFT,
            average="mean",
            verbose=False,
        )
        # Mean across channels
        psd_mean = np.mean(psd, axis=0)

        ax.plot(freqs, 10.0 * np.log10(psd_mean + 1e-30), label=label)

    ax.set_title("Power spectral density (mean across channels)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_xlim(_PSD_FMIN_HZ, _PSD_FMAX_HZ)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_timeseries_comparison(
    *,
    path: Path,
    raw_a: mne.io.BaseRaw,
    label_a: str,
    raw_b: mne.io.BaseRaw,
    label_b: str,
) -> None:
    """
    Save a stacked time-series plot for a subset of channels.
    """
    # Use channel selection from A, but apply to both (intersection)
    picks_a = _pick_seeg_like_channels(raw_a)
    if picks_a.size == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No sEEG/ECoG channels available for time-series QC.", ha="center", va="center")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    ch_names = [raw_a.ch_names[int(i)] for i in picks_a]
    common = [ch for ch in ch_names if ch in set(raw_b.ch_names)]
    if len(common) == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No common sEEG/ECoG channels between original and analysis view.", ha="center", va="center")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    chosen = _choose_channels_for_display(raw_a, common, n=_TIMESERIES_N_CHANNELS)

    # Window: first 30 seconds (or full length if shorter)
    duration = min(_TIMESERIES_DURATION_SEC, raw_a.times[-1] if raw_a.n_times > 1 else 0.0)
    tmin, tmax = 0.0, float(duration)

    raw_a_seg = raw_a.copy().pick(chosen).crop(tmin=tmin, tmax=tmax)
    raw_b_seg = raw_b.copy().pick(chosen).crop(tmin=tmin, tmax=tmax)

    raw_a_seg = _maybe_downsample_for_plot(raw_a_seg)
    raw_b_seg = _maybe_downsample_for_plot(raw_b_seg)

    data_a = raw_a_seg.get_data()
    data_b = raw_b_seg.get_data()
    times = raw_a_seg.times

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    # Normalize per-channel for readability: z-score within channel
    data_a = _zscore_rows(data_a)
    data_b = _zscore_rows(data_b)

    offset = 4.0  # spacing between traces in z units
    for i, ch in enumerate(chosen):
        ax.plot(times, data_a[i] + i * offset, linewidth=0.8)
    # Overlay analysis view with dashed lines
    for i, ch in enumerate(chosen):
        ax.plot(times, data_b[i] + i * offset, linewidth=0.8, linestyle="--")

    ax.set_title(f"Time-series QC ({tmin:.0f}–{tmax:.0f}s): {label_a} (solid) vs {label_b} (dashed)")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([i * offset for i in range(len(chosen))])
    ax.set_yticklabels(chosen)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _choose_channels_for_display(raw: mne.io.BaseRaw, candidates: Sequence[str], n: int) -> list[str]:
    """
    Choose channels by variance (highest first) for visibility.
    """
    n = min(int(n), len(candidates))
    data = raw.copy().pick(list(candidates)).get_data()
    variances = np.var(data, axis=1)
    order = np.argsort(variances)[::-1]
    chosen = [candidates[int(i)] for i in order[:n]]
    return chosen


def _maybe_downsample_for_plot(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    sfreq = float(raw.info["sfreq"])
    if sfreq <= _TIMESERIES_DOWNSAMPLE_MAX_HZ:
        return raw
    return raw.copy().resample(_TIMESERIES_DOWNSAMPLE_MAX_HZ, npad="auto", verbose=False)


def _zscore_rows(data: np.ndarray) -> np.ndarray:
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True) + 1e-12
    return (data - mean) / std
