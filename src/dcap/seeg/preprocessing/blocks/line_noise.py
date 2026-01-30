# =============================================================================
# =============================================================================
#                     ########################################
#                     #    BLOCK 3: LINE-NOISE REMOVAL       #
#                     ########################################
# =============================================================================
# =============================================================================

from dataclasses import asdict
from typing import Tuple

import numpy as np
import mne

from dcap.seeg.preprocessing.configs import LineNoiseConfig
from dcap.seeg.preprocessing.types import BlockArtifact, PreprocContext


def _compute_line_freqs(freq_base: float, max_freq: float) -> list[float]:
    n_harmonics = int(max_freq // freq_base)
    return [freq_base * k for k in range(1, n_harmonics + 1)]


def remove_line_noise(
    raw: mne.io.BaseRaw,
    cfg: LineNoiseConfig,
    ctx: PreprocContext,
) -> Tuple[mne.io.BaseRaw, BlockArtifact]:
    """
    Remove line noise using notch or zapline.

    Parameters
    ----------
    raw
        MNE Raw object.
    cfg
        Line-noise configuration.
    ctx
        Preprocessing context.

    Returns
    -------
    raw_out
        Raw object (copy) with line-noise removal applied.
    artifact
        Block artifact.

    Usage example
    -------------
        ctx = PreprocContext()
        raw_out, artifact = remove_line_noise(
            raw=raw,
            cfg=LineNoiseConfig(method="notch", freq_base=50.0, max_harmonic_hz=250.0),
            ctx=ctx,
        )
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("remove_line_noise expects an mne.io.BaseRaw.")

    ctx.add_record("line_noise", asdict(cfg))

    freqs = _compute_line_freqs(cfg.freq_base, cfg.max_harmonic_hz)
    raw_out = raw.copy()

    warnings: list[str] = []

    if cfg.method == "notch":
        raw_out.notch_filter(
            freqs=freqs,
            picks=cfg.picks,
            method="spectrum_fit",
            phase="zero",
            verbose=False,
        )
    elif cfg.method == "zapline":
        try:
            from meegkit import dss
        except ImportError as exc:
            raise ImportError("Zapline requested but meegkit is not installed.") from exc

        data = raw_out.get_data(picks=cfg.picks)
        print("ZapLine input shape:", data.shape, "dtype:", data.dtype)
        cleaned, _ = apply_zapline_chunked(
            raw,
            freq_base=cfg.freq_base,
            chunk_sec=cfg.chunk_sec,
            nremove=cfg.nremove,
        )

        warnings.append("Zapline applied using meegkit.dss.line (nremove=1).")
    else:
        raise ValueError(f"Unknown line-noise method: {cfg.method!r}")

    artifact = BlockArtifact(
        name="line_noise",
        parameters={**asdict(cfg), "freqs_applied": freqs},
        summary_metrics={"n_freqs": float(len(freqs))},
        warnings=warnings,
        figures=[],
    )
    return raw_out, artifact


# =============================================================================
#                           ZAPLINE (CHEAP, CHUNKED)
# =============================================================================

_ZAPLINE_DEFAULT_CHUNK_SEC: float = 60.0
_ZAPLINE_DEFAULT_NREMOVE: int = 1
_ZAPLINE_EPS: float = 1e-30


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
    Apply a memory-safe (chunked) ZapLine line-noise removal using meegkit.

    This wrapper is intentionally "cheap":
    - uses `meegkit.dss.dss_line` (single-pass) instead of the heavier iterative variant
    - processes data in time chunks to prevent OOM
    - applies only to picked channels (default: sEEG/ECoG; excludes ECG/stim/misc)

    Parameters
    ----------
    raw
        Input MNE Raw. Must be preloaded (raw.preload == True).
    freq_base
        Line frequency base (typically 50.0 or 60.0).
    chunk_sec
        Chunk duration in seconds. Larger chunks improve frequency resolution but use more RAM.
        60 seconds is a good default for clinical runs.
    nremove
        Number of DSS components to remove. Usually 1 is enough.
    picks
        Optional channel picks (indices). If None, picks sEEG/ECoG channels.
    use_float32
        If True, converts chunk data to float32 before calling meegkit (half the memory).

    Returns
    -------
    raw_out
        Copy of raw with cleaned data (only for the picked channels).
    metrics
        Basic metrics (currently: mean line-band power before/after, in arbitrary units).

    Usage example
    -------------
        raw_out, metrics = apply_zapline_chunked(
            raw,
            freq_base=50.0,
            chunk_sec=60.0,
            nremove=1,
        )
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
        # Nothing to do
        return raw.copy(), {"line_power_before": 0.0, "line_power_after": 0.0}

    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    chunk_len = int(round(chunk_sec * sfreq))
    chunk_len = max(chunk_len, int(sfreq))  # at least ~1 second

    data_in = raw.get_data(picks=picks)  # shape: (n_ch, n_times)
    data_out = data_in.copy()

    # Very rough “line power” proxy: energy at exactly freq_base using a sinusoid projection.
    # Cheap + fast, useful just to confirm it did something.
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

        # DSS line removal (single-pass). Returns (clean, artifact) in many versions.
        result = dss.dss_line(chunk_time_ch, fline=float(freq_base), sfreq=sfreq, nremove=int(nremove))
        if isinstance(result, tuple):
            clean_time_ch = result[0]
        else:
            clean_time_ch = result

        clean_ch_time = np.asarray(clean_time_ch).T
        data_out[:, start:stop] = clean_ch_time

        # accumulate quick after-metric on the cleaned chunk
        line_after_accum += _line_power_proxy(clean_ch_time, sfreq=sfreq, fline=float(freq_base))

        start = stop

    raw_out = raw.copy()
    # write back only picked channels
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
        Sampling frequency.
    fline
        Line frequency.

    Returns
    -------
    power
        Scalar proxy (arbitrary units). Comparable before vs after.
    """
    n_times = int(data_ch_time.shape[1])
    t = np.arange(n_times, dtype=float) / float(sfreq)
    sin = np.sin(2.0 * np.pi * float(fline) * t)[None, :]
    cos = np.cos(2.0 * np.pi * float(fline) * t)[None, :]

    # projections per channel
    a = np.sum(data_ch_time * sin, axis=1)
    b = np.sum(data_ch_time * cos, axis=1)

    # energy proxy
    power = np.mean(a * a + b * b) / max(n_times, 1)
    return float(power)

