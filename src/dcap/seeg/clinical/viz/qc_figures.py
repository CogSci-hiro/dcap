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
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _make_base_name(subject_id=subject_id, session_id=session_id, run_id=run_id)

    # -------------------------------------------------------------------------
    # Resolve an *effective* analysis label so we don't end up with nonsense like
    # "original vs original" when the view name is mis-propagated upstream.
    # -------------------------------------------------------------------------
    effective_view_name = analysis_view_name

    if analysis_view_name == "original":
        # If it's literally the same object, fine.
        if raw_analysis is raw_original:
            effective_view_name = "original"
        else:
            # Heuristic: bipolar reref often changes channel names (e.g., "A1-A2")
            orig_set = set(raw_original.ch_names)
            ana_set = set(raw_analysis.ch_names)

            # If analysis introduces many new channels with "-" it's very likely bipolar.
            new_ch = [ch for ch in raw_analysis.ch_names if ch not in orig_set]
            looks_bipolar = any("-" in ch for ch in new_ch) or any("-" in ch for ch in raw_analysis.ch_names)

            if looks_bipolar:
                effective_view_name = "bipolar"
            else:
                # Generic fallback: it's not original, even if upstream label said so.
                effective_view_name = "analysis"

    psd_path = out_dir / f"{base}_qc_psd_original-vs-{effective_view_name}.png"
    ts_path = out_dir / f"{base}_qc_timeseries_original-vs-{effective_view_name}.png"

    _save_psd_comparison(
        path=psd_path,
        raw_a=raw_original,
        label_a="original",
        raw_b=raw_analysis,
        label_b=effective_view_name,
    )
    _save_timeseries_comparison(
        path=ts_path,
        raw_a=raw_original,
        label_a="original",
        raw_b=raw_analysis,
        label_b=effective_view_name,
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
    picks_a_all = _pick_seeg_like_channels(raw_a)
    if picks_a_all.size == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No sEEG/ECoG channels available for time-series QC.", ha="center", va="center")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    ch_names_a = [raw_a.ch_names[int(i)] for i in picks_a_all]
    ch_set_b = set(raw_b.ch_names)
    common = [ch for ch in ch_names_a if ch in ch_set_b]
    if len(common) == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No common sEEG/ECoG channels between original and analysis view.", ha="center", va="center")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    chosen = _choose_channels_for_display(raw_a, common, n=_TIMESERIES_N_CHANNELS)

    duration = min(_TIMESERIES_DURATION_SEC, float(raw_a.times[-1]) if raw_a.n_times > 1 else 0.0)
    tmin, tmax = 0.0, float(duration)

    # -------------------------------------------------------------------------
    # Robust extraction: index-based mapping + sample-range slicing
    # -------------------------------------------------------------------------
    sfreq_a = float(raw_a.info["sfreq"])
    sfreq_b = float(raw_b.info["sfreq"])

    start_a, stop_a = _time_range_to_samples(raw_a, tmin=tmin, tmax=tmax)
    start_b, stop_b = _time_range_to_samples(raw_b, tmin=tmin, tmax=tmax)

    picks_a = np.array([raw_a.ch_names.index(ch) for ch in chosen], dtype=int)
    picks_b = np.array([raw_b.ch_names.index(ch) for ch in chosen], dtype=int)

    data_a = raw_a.get_data(picks=picks_a, start=start_a, stop=stop_a)
    data_b = raw_b.get_data(picks=picks_b, start=start_b, stop=stop_b)

    # Plot-time downsampling (decimate) to avoid heavy resample logic + avoid surprises
    data_a, times = _downsample_for_plot_array(data_a, sfreq=sfreq_a)
    data_b, _ = _downsample_for_plot_array(data_b, sfreq=sfreq_b)

    data_a = _zscore_rows(data_a)
    data_b = _zscore_rows(data_b)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    palette = (
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    )

    offset = 4.0
    offsets = np.arange(len(chosen)) * offset

    # Original
    for i in range(len(chosen)):
        ax.plot(
            times,
            data_a[i] + offsets[i],
            color="#B0B0B0",
            linewidth=0.9,
            alpha=0.9,
            zorder=1,
        )

    # Analysis
    for i in range(len(chosen)):
        ax.plot(
            times,
            data_b[i] + offsets[i],
            color=palette[i % len(palette)],
            linewidth=1.6,
            alpha=0.95,
            zorder=2,
        )

    # -------------------------------------------------------------------------
    # Sanity check: are all analysis channels identical (up to numerical tol)?
    # If yes, the figure is misleading; flag it loudly.
    # -------------------------------------------------------------------------
    if _rows_all_close(data_b):
        ax.text(
            0.01,
            0.99,
            "WARNING: analysis traces are identical across channels.\n"
            "This usually indicates a CAR/transform bug (signal broadcast) or a data-view aliasing issue.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

    ax.set_title(f"Time-series QC ({tmin:.0f}–{tmax:.0f}s): {label_a} vs {label_b}")
    ax.set_xlabel("Time (s)")
    ax.set_yticks(offsets)
    ax.set_yticklabels(chosen)
    ax.grid(True, alpha=0.2)

    ax.plot([], [], color="#B0B0B0", linewidth=0.9, label=label_a)
    ax.plot([], [], color=palette[0], linewidth=1.6, label=label_b)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _time_range_to_samples(raw: mne.io.BaseRaw, *, tmin: float, tmax: float) -> tuple[int, int]:
    sfreq = float(raw.info["sfreq"])
    start = int(round(tmin * sfreq))
    stop = int(round(tmax * sfreq))
    start = max(0, min(start, raw.n_times))
    stop = max(start, min(stop, raw.n_times))
    return start, stop


def _downsample_for_plot_array(data: np.ndarray, *, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Cheap deterministic downsampling for plotting (decimation).
    Returns (data_ds, times_ds).
    """
    if sfreq <= _TIMESERIES_DOWNSAMPLE_MAX_HZ:
        times = np.arange(data.shape[1], dtype=float) / sfreq
        return data, times

    decim = int(np.floor(sfreq / _TIMESERIES_DOWNSAMPLE_MAX_HZ))
    decim = max(1, decim)

    data_ds = data[:, ::decim]
    sfreq_ds = sfreq / decim
    times_ds = np.arange(data_ds.shape[1], dtype=float) / sfreq_ds
    return data_ds, times_ds


def _rows_all_close(data: np.ndarray, *, rtol: float = 1e-6, atol: float = 1e-8) -> bool:
    """
    True if all rows are (nearly) identical to the first row.
    """
    if data.shape[0] <= 1:
        return False
    ref = data[0:1, :]
    return bool(np.allclose(data, ref, rtol=rtol, atol=atol))


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
