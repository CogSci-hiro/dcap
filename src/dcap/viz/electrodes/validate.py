# =============================================================================
#                     ########################################
#                     #        INPUT VALIDATION / CLEANING    #
#                     ########################################
# =============================================================================
"""Validation and cleaning helpers for electrode plotting inputs."""

from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
#                     ########################################
#                     #               CONSTANTS               #
#                     ########################################
# =============================================================================
REQUIRED_COLUMNS = ("name", "x", "y", "z")


def validate_and_clean_electrodes_df(
    electrodes_df: pd.DataFrame,
    *,
    values_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Validate and clean the electrode table.

    Parameters
    ----------
    electrodes_df
        Canonical electrode table with at least columns:
        - name (str)
        - x, y, z (numeric)
    values_col
        Optional column name for per-electrode numeric values.

    Returns
    -------
    cleaned_df
        DataFrame filtered to rows with finite x/y/z. Preserves columns.

    Notes
    -----
    Example expected input format:

    +------+-------+-------+------+-------+--------+
    | name | x     | y     | z    | space | score  |
    +------+-------+-------+------+-------+--------+
    | LA1  | -34.2 | -12.0 | 18.5 | MNI   | 0.12   |
    | LA2  | -33.7 | -10.9 | 16.9 | MNI   | 0.08   |
    | RA1  |  29.1 |  -8.4 | 21.2 | MNI   | 0.31   |
    +------+-------+-------+------+-------+--------+
    """
    if electrodes_df.empty:
        raise ValueError("electrodes_df is empty.")

    missing = [col for col in REQUIRED_COLUMNS if col not in electrodes_df.columns]
    if missing:
        raise ValueError(f"electrodes_df missing required columns: {missing}")

    if values_col is not None and values_col not in electrodes_df.columns:
        raise ValueError(f"values_col='{values_col}' not found in electrodes_df columns.")

    # Coerce to numeric where possible; invalid parses become NaN.
    x = pd.to_numeric(electrodes_df["x"], errors="coerce")
    y = pd.to_numeric(electrodes_df["y"], errors="coerce")
    z = pd.to_numeric(electrodes_df["z"], errors="coerce")

    finite_mask = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy()) & np.isfinite(z.to_numpy())
    cleaned_df = electrodes_df.loc[finite_mask].copy()

    if cleaned_df.empty:
        raise ValueError("All electrode rows were dropped due to missing/non-finite coordinates.")

    return cleaned_df
