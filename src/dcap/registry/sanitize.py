"""
Utilities for generating sanitized (shareable) metadata from private sources.

Important
---------
This module exists to help create *public* artifacts from private metadata
without leaking identifiers. It uses strict allowlists and explicit mapping
inputs.

Nothing here should ever auto-detect or infer identifiers.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from dcap.registry.schema import PUBLIC_REQUIRED_COLUMNS, QC_STATUS_ALLOWED


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def make_public_registry(
    *,
    private_registry_path: Path,
    output_path: Path,
    bids_root: str,
    default_qc_status: str = "unknown",
    subject_map_path: Optional[Path] = None,
    subject_key_column: str = "subject_key",
    subject_column: str = "subject",
) -> pd.DataFrame:
    """
    Create a **public/shareable** registry table from a private registry.

    Parameters
    ----------
    private_registry_path
        Path to the private registry (CSV/Parquet). Must contain join keys.
    output_path
        Output path for the public registry (CSV/Parquet).
    bids_root
        BIDS root path to write into the public registry `bids_root` column.
        This is often a dataset root, not a per-run file.
    default_qc_status
        QC status to use if none exists in the private registry.
        Must be one of: pass/fail/review/unknown.
    subject_map_path
        Optional path to a mapping table with columns:
        - subject_key
        - subject
        Use this if the private registry identifies subjects via `subject_key`.
    subject_key_column
        Column name for subject keys in the private registry and mapping file.
    subject_column
        Column name for the anonymized subject label in the output.

    Returns
    -------
    public_registry
        Public registry DataFrame that was written to disk.

    Notes
    -----
    Expected output DataFrame format (example):

    | subject | session | task | run | bids_root | qc_status |
    |---|---|---|---:|---|---|
    | sub-001 | ses-01 | conversation | 1 | /data/bids | unknown |

    Usage example
    ------------
        from pathlib import Path
        from dcap.registry.sanitize import make_public_registry

        make_public_registry(
            private_registry_path=Path("~/.dcap_private/registry_private.csv").expanduser(),
            output_path=Path("registry_public.csv"),
            bids_root="/data/bids/conversation",
            subject_map_path=Path("subject_map.csv"),
        )
    """
    if default_qc_status not in QC_STATUS_ALLOWED:
        raise ValueError(f"default_qc_status must be one of {list(QC_STATUS_ALLOWED)}; got {default_qc_status!r}")

    private_df = _read_table(private_registry_path)

    required_join = ["session", "task", "run"]
    missing = [c for c in required_join if c not in private_df.columns]
    if missing:
        raise ValueError(f"Private registry missing required columns: {missing}")

    df = private_df.copy()

    # Resolve subject label
    if subject_column in df.columns:
        pass
    else:
        if subject_map_path is None:
            raise ValueError(
                f"Private registry does not contain '{subject_column}'. Provide --subject-map to map "
                f"'{subject_key_column}' -> '{subject_column}'."
            )
        mapping = _read_table(subject_map_path)
        if subject_key_column not in mapping.columns or subject_column not in mapping.columns:
            raise ValueError(
                f"Subject map must contain columns '{subject_key_column}' and '{subject_column}'. "
                f"Got columns={list(mapping.columns)}"
            )
        if subject_key_column not in df.columns:
            raise ValueError(
                f"Private registry missing subject key column '{subject_key_column}' required for mapping."
            )
        df = df.merge(mapping[[subject_key_column, subject_column]], how="left", on=subject_key_column, validate="many_to_one")
        if df[subject_column].isna().any():
            n_missing = int(df[subject_column].isna().sum())
            raise ValueError(f"Subject mapping incomplete: {n_missing} rows missing '{subject_column}'.")

    # qc_status
    if "qc_status" not in df.columns:
        df["qc_status"] = default_qc_status
    else:
        bad = sorted(set(df["qc_status"].dropna().astype(str)) - set(QC_STATUS_ALLOWED))
        if bad:
            raise ValueError(f"Invalid qc_status values in private registry: {bad}. Allowed={list(QC_STATUS_ALLOWED)}")

    # Compose public table with strict allowlist
    public_cols = ["subject", "session", "task", "run", "qc_status"]
    public = df[public_cols].copy()
    public["bids_root"] = str(bids_root)

    # Enforce required column order (and presence)
    for col in PUBLIC_REQUIRED_COLUMNS:
        if col not in public.columns:
            raise RuntimeError(f"Internal error: missing required public column {col!r}")

    public = public[list(PUBLIC_REQUIRED_COLUMNS)].copy()
    _write_table(public, output_path)
    return public
