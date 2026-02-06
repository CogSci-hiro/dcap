# =============================================================================
# =============================================================================
#                     ########################################
#                     #    BLOCK 3: LINE-NOISE REMOVAL       #
#                     ########################################
# =============================================================================
# =============================================================================
"""
Line-noise removal utilities (50/60 Hz + harmonics).

Two APIs
--------
1) Analysis-friendly API (library style)
   - remove_line_noise(raw, method="notch" | "zapline", ...) -> Raw
   - MNE-like: arguments have sensible defaults, no config object required.

2) Clinical/report wrapper
   - remove_line_noise_view(raw, cfg, ctx) -> (Raw, BlockArtifact)
   - reads parameters from LineNoiseConfig + records provenance in ctx
   - returns BlockArtifact (warnings/metrics) for report/QC

Methods
-------
- "notch":
    Uses MNE's notch_filter with harmonic frequencies computed from:
      freq_base, max_harmonic_hz

- "zapline":
    Uses meegkit.dss.dss_line (chunked, memory-safe) via apply_zapline_chunked().
    Intended as a cheap, robust alternative in settings where notch filtering
    is undesirable.

Notes
-----
- ZapLine requires `meegkit`. If it is not installed, we raise ImportError.
- For zapline we compute a simple "line power proxy" metric (before/after)
  to confirm the operation had an effect (useful for QC).
"""

from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np
import mne

from dcap.seeg.preprocessing.configs import LineNoiseConfig
from dcap.seeg.preprocessing.types import BlockArtifact, PreprocContext


# =============================================================================
#                     ########################################
#                     #              Public API              #
#                     ########################################
# =============================================================================
def _compute_line_freqs(freq_base: float, max_freq: float) -> list[float]:
    """
    Compute harmonic frequencies: [f, 2f, 3f, ...] up to max_freq.
    """
    n_harmonics = int(max_freq // freq_base)
    return [freq_base * k for k in range(1, n_harmonics + 1)]


def remove_line_noise(
    raw: mne.io.BaseRaw,
    method: str = "notch",
    *,
    freq_base: float = 50.0,
    max_harmonic_hz: float = 250.0,
    picks: Optional[np.ndarray] = None,
    # Notch-specific options (kept explicit to mirror MNE feel)
    notch_method: str = "spectrum_fit",
    phase: str = "zero",
    # ZapLine-specific options
    chunk_sec: float = 60.0,
    nremove: int = 1,
    use_float32: bool = True,
    # Safety / consistency
    copy: bool = True,
) -> mne.io.BaseRaw:
    """
    Remove line noise from Raw using notch filtering or ZapLine.

    Parameters
    ----------
    raw
        Input MNE Raw.
    method
        "notch" or "zapline".
    freq_base
        Base line frequency (typically 50.0 or 60.0).
    max_harmonic_hz
        Max frequency for harmonic notches (only used for notch).
    picks
        Channel picks (indices). If None, defaults to sEEG+ECoG.
        (Stim/ECG/EOG/etc are excluded by default.)
    notch_method
        Passed to `raw.notch_filter(method=...)`. Default "spectrum_fit".
    phase
        Passed to `raw.notch_filter(phase=...)`. Default "zero".
    chunk_sec, nremove, use_float32
        ZapLine parameters (used only if method="zapline").
    copy
        If True, operate on a copy and return it. If False, modify the input Raw.

    Returns
    -------
    raw_out
        Raw with line-noise removal applied.

    Usage example
    -------------
        # Notch (50 Hz + harmonics)
        raw_clean = remove_line_noise(raw, method="notch", freq_base=50.0)

        # ZapLine (meegkit required)
        raw_clean = remove_line_noise(raw, method="zapline", freq_base=50.0, chunk_sec=60.0, nremove=1)
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("remove_line_noise expects an mne.io.BaseRaw.")

    raw_out = raw.copy() if copy else raw

    # Default picks = iEEG only (safe default; avoids messing with stim/ECG/etc)
    if picks is None:
        picks = mne.pick_types(
            raw_out.info,
            seeg=True,
            ecog=True,
            eeg=False,
            meg=False,
            stim=False,
            misc=False,
            eog=False,
            ecg=False,
        )
    picks = np.asarray(picks, dtype=int)

    if method == "notch":
        freqs = _compute_line_freqs(freq_base, max_harmonic_hz)
        # MNE applies in-place to raw_out
        raw_out.notch_filter(
            freqs=freqs,
            picks=picks,
            method=notch_method,
            phase=phase,
            verbose=False,
        )
        return raw_out

    if method == "zapline":
        # ZapLine requires data access; chunked implementation requires preload.
        if not bool(raw_out.preload):
            raw_out.load_data()

        raw_out, _metrics = apply_zapline_chunked(
            raw_out,
            freq_base=freq_base,
            chunk_sec=chunk_sec,
            nremove=nremove,
            picks=picks,
            use_float32=use_float32,
        )
        return raw_out

    raise ValueError(f"Unknown line-noise method: {method!r}")


# =============================================================================
#                     ########################################
#                     #         Clinical wrapper API         #
#                     ########################################
# =============================================================================
def remove_line_noise_view(
    raw: mne.io.BaseRaw,
    cfg: LineNoiseConfig,
    ctx: PreprocContext,
) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    """
    Clinical/report wrapper for line-noise removal.

    - Records provenance in ctx
    - Runs the requested method
    - Returns BlockArtifact with warnings + summary metrics

    Usage example
    -------------
        raw_out, artifact = remove_line_noise_view(raw, cfg, ctx)
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("remove_line_noise_view expects an mne.io.BaseRaw.")

    ctx.add_record("line_noise", asdict(cfg))

    warnings: list[str] = []

    # Precompute freqs for artifact (meaningful for notch; still useful to log for zapline)
    freqs = _compute_line_freqs(cfg.freq_base, cfg.max_harmonic_hz)

    # Run via analysis API, but keep config-driven knobs
    metrics: Dict[str, float] = {}

    if cfg.method == "notch":
        raw_out = remove_line_noise(
            raw,
            method="notch",
            freq_base=float(cfg.freq_base),
            max_harmonic_hz=float(cfg.max_harmonic_hz),
            picks=cfg.picks,
            notch_method="spectrum_fit",
            phase="zero",
            copy=True,
        )

    elif cfg.method == "zapline":
        # Ensure meegkit present early so the failure is explicit.
        try:
            import meegkit  # noqa: F401
        except ImportError as exc:
            raise ImportError("Zapline requested but `meegkit` is not installed (pip install meegkit).") from exc

        raw_pre = raw.copy()
        if not bool(raw_pre.preload):
            raw_pre.load_data()

        raw_out, metrics = apply_zapline_chunked(
            raw_pre,
            freq_base=float(cfg.freq_base),
            chunk_sec=float(cfg.chunk_sec),
            nremove=int(cfg.nremove),
            picks=cfg.picks,
            use_float32=True,
        )
        warnings.append("ZapLine applied using meegkit.dss.dss_line (chunked).")

    else:
        raise ValueError(f"Unknown line-noise method: {cfg.method!r}")

    artifact = BlockArtifact(
        name="line_noise",
        parameters={**asdict(cfg), "freqs_applied": freqs},
        summary_metrics={
            "n_freqs": float(len(freqs)),
            # ZapLine metrics (if any)
            **metrics,
        },
        warnings=warnings,
        figures=[],
    )
    return raw_out, artifact


# Backward-compat alias (optional):
# If existing code calls remove_line_noise(raw, cfg, ctx), keep this for now.
def remove_line_noise_block(
    raw: mne.io.BaseRaw,
    cfg: LineNoiseConfig,
    ctx: PreprocContext,
) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    """
    Deprecated compat alias for the old block entrypoint name.

    Prefer:
        remove_line_noise_view(raw, cfg, ctx)
    """
    return remove_line_noise_view(raw, cfg, ctx)


# =============================================================================
#                           ZAPLINE (CHEAP, CHUNKED)
# =============================================================================
_ZAPLINE_DEFAULT_CHUNK_SEC: float = 60.0
_ZAPLINE_DEFAULT_NREMOVE: int = 1


def apply_zapline_chunked(
    raw: mne.io.BaseRaw,
    *,
    freq_base: float,
    chunk_sec: float = _ZAPLINE_DEFAULT_CHUNK_SEC,
    nremove: int = _ZAPLINE_DEFAULT_NREMOVE,
    picks: np.ndarray | None = None,
    use_float32: bool = True,
) -> tuple[mne.io.BaseRaw, dict[str, float]]:
    """
    Apply memory-safe (chunked) ZapLine line-noise removal using meegkit.dss.dss_line.

    Parameters
    ----------
    raw
        Input Raw. Must be preloaded (raw.preload == True). (We enforce this.)
    freq_base
        Base line frequency (50 or 60).
    chunk_sec
        Chunk duration in seconds. Larger chunks = better frequency resolution, more RAM.
    nremove
        Number of DSS components to remove (1 is usually enough).
    picks
        Channel picks. If None, defaults to iEEG (sEEG+ECoG).
    use_float32
        Convert chunks to float32 to reduce memory.

    Returns
    -------
    raw_out
        Cleaned copy of raw (only picked channels modified).
    metrics
        Cheap line-power proxy before/after (useful for QC).

    Usage example
    -------------
        raw_clean, metrics = apply_zapline_chunked(raw, freq_base=50.0, chunk_sec=60.0, nremove=1)
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("apply_zapline_chunked expects an mne.io.BaseRaw.")
    if not bool(raw.preload):
        raise ValueError("apply_zapline_chunked requires raw.preload == True (call raw.load_data()).")
    if chunk_sec <= 0:
        raise ValueError("chunk_sec must be > 0.")
    if nremove <= 0:
        raise ValueError("nremove must be >= 1.")

    try:
        from meegkit import dss
    except Exception as exc:  # pragma: no cover
        raise ImportError("ZapLine requires `meegkit` (pip install meegkit).") from exc

    if picks is None:
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

    picks = np.asarray(picks, dtype=int)
    if picks.size == 0:
        return raw.copy(), {"line_power_before": 0.0, "line_power_after": 0.0}

    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)

    chunk_len = int(round(chunk_sec * sfreq))
    chunk_len = max(chunk_len, int(sfreq))  # at least ~1 second

    data_in = raw.get_data(picks=picks)  # (n_ch, n_times)
    data_out = data_in.copy()

    # Cheap QC metric: energy at exactly freq_base via sin/cos projection
    line_before = _line_power_proxy(data_in, sfreq=sfreq, fline=float(freq_base))
    line_after_accum = 0.0

    start = 0
    while start < n_times:
        stop = min(start + chunk_len, n_times)

        # meegkit expects (n_samples, n_chans)
        chunk_ch_time = data_out[:, start:stop]
        chunk_time_ch = chunk_ch_time.T

        if use_float32 and chunk_time_ch.dtype != np.float32:
            chunk_time_ch = chunk_time_ch.astype(np.float32, copy=False)

        result = dss.dss_line(chunk_time_ch, fline=float(freq_base), sfreq=sfreq, nremove=int(nremove))
        clean_time_ch = result[0] if isinstance(result, tuple) else result

        clean_ch_time = np.asarray(clean_time_ch).T
        data_out[:, start:stop] = clean_ch_time

        line_after_accum += _line_power_proxy(clean_ch_time, sfreq=sfreq, fline=float(freq_base))
        start = stop

    raw_out = raw.copy()
    raw_out._data[picks, :] = data_out

    metrics = {
        "line_power_before": float(line_before),
        "line_power_after": float(line_after_accum),
        "sfreq_hz": sfreq,
        "fline_hz": float(freq_base),
        "chunk_sec": float(chunk_sec),
        "nremove": float(nremove),
        "n_channels_cleaned": float(picks.size),
    }
    return raw_out, metrics


def _line_power_proxy(data_ch_time: np.ndarray, *, sfreq: float, fline: float) -> float:
    """
    Cheap proxy for energy at fline: projection onto sin/cos at fline.

    Parameters
    ----------
    data_ch_time
        Array of shape (n_ch, n_times).
    sfreq
        Sampling frequency (Hz).
    fline
        Line frequency (Hz).

    Returns
    -------
    power
        Scalar proxy (arbitrary units). Comparable before vs after.
    """
    n_times = int(data_ch_time.shape[1])
    t = np.arange(n_times, dtype=float) / float(sfreq)
    sin = np.sin(2.0 * np.pi * float(fline) * t)[None, :]
    cos = np.cos(2.0 * np.pi * float(fline) * t)[None, :]

    a = np.sum(data_ch_time * sin, axis=1)
    b = np.sum(data_ch_time * cos, axis=1)

    power = np.mean(a * a + b * b) / max(n_times, 1)
    return float(power)
