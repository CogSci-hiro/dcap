# src/dcap/seeg/clinical/qc.py
# =============================================================================
# =============================================================================
#                     ########################################
#                     #         CLINICAL QC SUMMARY          #
#                     ########################################
# =============================================================================
# =============================================================================

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import mne


# =============================================================================
# Constants
# =============================================================================

_FLAT_EPS: float = 1e-12
_IQR_OUTLIER_K: float = 6.0


@dataclass(frozen=True)
class ClinicalQcSummary:
    """
    Minimal QC summary intended for clinician-facing reports.

    Attributes
    ----------
    recording
        Recording-level summary metrics.
    views
        Per-view summary (n_channels, duration, sfreq).
    channel_qc
        Optional per-channel QC table (variance, flat flag, outlier flag).

    Usage example
    -------------
        qc = compute_clinical_qc(raw_views={"original": raw})
    """

    recording: Mapping[str, float]
    views: pd.DataFrame
    channel_qc: Optional[pd.DataFrame] = None


def compute_clinical_qc(
    *,
    raw_views: Mapping[str, mne.io.BaseRaw],
    include_channel_table: bool = True,
) -> ClinicalQcSummary:
    """
    Compute a minimal QC summary from preprocessed views.

    Parameters
    ----------
    raw_views
        Mapping view_name -> Raw. Must include "original".
    include_channel_table
        If True, compute a per-channel QC table for the "original" view.

    Returns
    -------
    qc
        ClinicalQcSummary.

    Usage example
    -------------
        qc = compute_clinical_qc(raw_views={"original": raw})
    """
    if "original" not in raw_views:
        raise ValueError('raw_views must include an "original" view.')

    original = raw_views["original"]
    duration_sec = float(original.n_times) / float(original.info["sfreq"])
    n_channels = float(len(original.ch_names))
    sfreq = float(original.info["sfreq"])

    recording = {
        "duration_sec": duration_sec,
        "sfreq_hz": sfreq,
        "n_channels": n_channels,
    }

    views_rows = []
    for view_name, raw in raw_views.items():
        views_rows.append(
            {
                "view": str(view_name),
                "n_channels": int(len(raw.ch_names)),
                "sfreq_hz": float(raw.info["sfreq"]),
                "duration_sec": float(raw.n_times) / float(raw.info["sfreq"]),
            }
        )
    views_df = pd.DataFrame(views_rows, columns=["view", "n_channels", "sfreq_hz", "duration_sec"])

    channel_df: Optional[pd.DataFrame] = None
    if include_channel_table:
        channel_df = _compute_channel_qc_table(original)

    return ClinicalQcSummary(recording=recording, views=views_df, channel_qc=channel_df)


def _compute_channel_qc_table(raw: mne.io.BaseRaw) -> pd.DataFrame:
    """
    Compute per-channel QC metrics from the provided Raw.

    Notes
    -----
    This is intentionally simple and robust:
    - variance-based flat detection
    - robust outlier detection via IQR on log-variance

    Returns
    -------
    df
        DataFrame with columns: channel, variance, log_variance, is_flat, is_outlier

    Usage example
    -------------
        df = _compute_channel_qc_table(raw)
    """
    data = raw.get_data()
    variances = np.var(data, axis=1).astype(float)
    log_var = np.log10(np.maximum(variances, _FLAT_EPS))

    is_flat = variances <= _FLAT_EPS
    is_outlier = _robust_outlier_mask(log_var, k=_IQR_OUTLIER_K)

    df = pd.DataFrame(
        {
            "channel": list(raw.ch_names),
            "variance": variances,
            "log10_variance": log_var,
            "is_flat": is_flat,
            "is_outlier": is_outlier,
        }
    )

    return df


def _robust_outlier_mask(values: np.ndarray, k: float) -> np.ndarray:
    """
    Robust outlier mask using IQR rule: [Q1 - k*IQR, Q3 + k*IQR].

    Parameters
    ----------
    values
        1D array.
    k
        Outlier multiplier.

    Returns
    -------
    mask
        Boolean mask of same shape as values.

    Usage example
    -------------
        mask = _robust_outlier_mask(values, k=6.0)
    """
    q1 = float(np.nanpercentile(values, 25))
    q3 = float(np.nanpercentile(values, 75))
    iqr = max(q3 - q1, 1e-12)
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return (values < lo) | (values > hi)
