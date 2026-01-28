# src/dcap/registry/schema/private.py
# =============================================================================
#                       Registry schema: private overlay
# =============================================================================
from dataclasses import dataclass
from typing import Literal, Sequence

import pandas as pd


QcStatus = Literal["pass", "fail", "review", "unknown"]


@dataclass(frozen=True, slots=True)
class PrivateRegistrySchema:
    """
    Schema for the private (never-committed) registry overlay.

    Required columns
    ----------------
    - record_id

    Recommended columns
    -------------------
    - qc_status: pass|fail|review|unknown
    - exclude: bool
    - exclude_reason: str
    - notes: str
    - tags: str (comma-separated) or JSON-ish; keep simple initially
    - reviewer: str
    - review_date: ISO date string

    Usage example
    -------------
        schema = PrivateRegistrySchema()
        schema.validate(df_private)
    """

    required_columns: tuple[str, ...] = ("record_id",)

    allowed_columns: tuple[str, ...] = (
        "record_id",
        "qc_status",
        "exclude",
        "exclude_reason",
        "notes",
        "tags",
        "reviewer",
        "review_date",
        "clock_drift_ms",
        "bad_channels",
        "private_doc_ref",
    )

    allowed_qc_values: tuple[str, ...] = ("pass", "fail", "review", "unknown")

    def validate(self, df: pd.DataFrame) -> None:
        """
        Validate private registry overlay columns and allowed qc_status values.

        Parameters
        ----------
        df
            Private registry DataFrame.

        Raises
        ------
        ValueError
            If required columns are missing, or unexpected columns exist,
            or qc_status values are invalid.

        Usage example
        -------------
            schema = PrivateRegistrySchema()
            schema.validate(df_private)
        """
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Private registry missing required columns: {missing}")

        unexpected = [c for c in df.columns if c not in self.allowed_columns]
        if unexpected:
            raise ValueError(f"Private registry has unexpected columns (strict mode): {unexpected}")

        if "qc_status" in df.columns:
            bad = sorted(set(df["qc_status"].dropna().unique()) - set(self.allowed_qc_values))
            if bad:
                raise ValueError(f"Private registry has invalid qc_status values: {bad}")
