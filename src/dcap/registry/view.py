# src/dcap/registry/view.py
# =============================================================================
#                     Registry view: public + private join
# =============================================================================
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True, slots=True)
class RegistryMergePolicy:
    """
    Policy controlling how private metadata overlays public manifest.

    Rules
    -----
    - Private may not redefine BIDS entities or dataset identity.
    - Private may add/override annotation fields such as qc_status, exclude, notes.

    Usage example
    -------------
        policy = RegistryMergePolicy()
        df_view, warnings = build_registry_view(df_public, df_private, policy)
    """

    protected_columns: tuple[str, ...] = (
        "dataset_id",
        "bids_root",
        "subject",
        "session",
        "task",
        "run",
        "datatype",
        "record_id",
    )

    overlay_columns: tuple[str, ...] = (
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


def build_registry_view(
    df_public: pd.DataFrame,
    df_private: pd.DataFrame,
    policy: Optional[RegistryMergePolicy] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Join public registry with optional private overlay and compute derived fields.

    Parameters
    ----------
    df_public
        Public manifest DataFrame.
    df_private
        Private overlay DataFrame (possibly empty).
    policy
        Merge policy controlling allowed overlays.

    Returns
    -------
    df_view : pandas.DataFrame
        Joined view with derived fields:
        - effective_qc_status
        - is_excluded
        - is_usable
    warnings : list[str]
        Human-readable warnings (e.g., orphan private rows).

    View format (example)
    ---------------------
    | record_id | subject | task | run | qc_status | exclude | effective_qc_status | is_usable |
    |----------|---------|------|-----|----------:|--------:|---------------------|----------:|
    | ...      | sub-001 | conv | 1   | pass      | False   | pass                | True      |

    Usage example
    -------------
        df_view, warnings = build_registry_view(df_public, df_private)
        usable = df_view[df_view["is_usable"]]
    """
    if policy is None:
        policy = RegistryMergePolicy()

    warnings: list[str] = []

    if "record_id" not in df_public.columns:
        raise ValueError("df_public must include 'record_id' column.")
    if "record_id" not in df_private.columns:
        # allow empty overlay created by loader
        df_private = pd.DataFrame({"record_id": pd.Series(dtype="string")})

    # Detect forbidden columns in private that collide with protected columns
    forbidden = [c for c in df_private.columns if c in policy.protected_columns and c != "record_id"]
    if forbidden:
        raise ValueError(
            "Private registry must not define protected columns. "
            f"Found forbidden columns in private: {forbidden}"
        )

    # Orphan private rows (record_id not in public)
    if len(df_private) > 0:
        public_ids = set(df_public["record_id"].astype(str).tolist())
        private_ids = set(df_private["record_id"].astype(str).tolist())
        orphan_ids = sorted(private_ids - public_ids)
        if orphan_ids:
            warnings.append(
                f"Private registry contains {len(orphan_ids)} orphan record_id(s) not present in public manifest. "
                "They will be ignored by the join."
            )
            df_private = df_private[df_private["record_id"].astype(str).isin(public_ids)].copy()

    # Reduce private to overlay columns (keep record_id)
    keep_private_cols = ["record_id", *[c for c in policy.overlay_columns if c in df_private.columns]]
    df_private_reduced = df_private.loc[:, keep_private_cols].copy()

    df_view = df_public.merge(df_private_reduced, how="left", on="record_id", validate="one_to_one")

    # Derived fields
    if "qc_status" not in df_view.columns:
        df_view["qc_status"] = pd.Series([pd.NA] * len(df_view), dtype="string")

    if "exclude" not in df_view.columns:
        df_view["exclude"] = False

    df_view["effective_qc_status"] = df_view["qc_status"].fillna("unknown")
    df_view["is_excluded"] = df_view["exclude"].fillna(False).astype(bool)
    df_view["is_usable"] = (df_view["effective_qc_status"] == "pass") & (~df_view["is_excluded"])

    return df_view, warnings
