# dcap/registry/view.py
# =============================================================================
#                         Registry: runtime view
# =============================================================================
#
# Build an in-memory "registry view" by joining:
# - registry_public.tsv (shareable structural index)
# - registry_private.tsv (optional, local-only decisions)
#
# Join key: record_id
#
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import csv


# =============================================================================
# Constants
# =============================================================================

PRIVATE_DECISIONS_EXPECTED_COLUMNS: Tuple[str, ...] = (
    "record_id",
    "exclude_reason",
    "review_date",
    "notes",
)

VIEW_EXTRA_COLUMNS: Tuple[str, ...] = (
    "excluded",
    "exclude_reason",
    "review_date",
    "notes",
)


# =============================================================================
# Public API
# =============================================================================

def build_registry_view(
    *,
    public_registry: Path,
    private_registry: Optional[Path],
) -> List[Dict[str, Any]]:
    """
    Build the runtime registry view (public + optional private overlays).

    Parameters
    ----------
    public_registry
        Path to registry_public.tsv.
    private_registry
        Optional path to registry_private.tsv. If None or missing, the view is
        built from public only (no exclusions / notes).

    Returns
    -------
    list[dict[str, Any]]
        List of row dicts. Each row includes all public columns plus:
        - excluded: bool
        - exclude_reason: str
        - review_date: str
        - notes: str

    Notes
    -----
    This is an in-memory join on record_id.
    The resulting view is *not* intended for version control.

    Output format example
    ---------------------
    Registry view rows look like:

    | dataset_id | subject  | session | acquisition_id | protocol_id | task        | sex    | age_years | record_id                                 | excluded | exclude_reason | review_date  |
    |-----------|----------|---------|----------------|------------|-------------|--------|----------:|-------------------------------------------|---------:|----------------|------------|
    | Timone2025| sub-001  | ses-01  | acq-01         | prot-01    | conversation| male   | 34        | Timone2025|sub-001|ses-01|acq-01|prot-01 | False    |                |            |

    Usage example
    -------------
        from pathlib import Path
        from dcap.registry.view import build_registry_view

        rows = build_registry_view(
            public_registry=Path("registry_public.tsv"),
            private_registry=Path("/secure/DCAP_PRIVATE_ROOT/registry_private.tsv"),
        )

        # Optional: convert to pandas if you want
        # import pandas as pd
        # df = pd.DataFrame(rows)
    """
    public_rows, public_header = _read_tsv(public_registry)

    private_index: Dict[str, Dict[str, str]] = {}
    if private_registry is not None and private_registry.exists():
        private_rows, private_header = _read_tsv(private_registry)
        private_index = _index_private_decisions(private_rows, private_header)

    view_rows: List[Dict[str, Any]] = []
    for public_row in public_rows:
        record_id = str(public_row.get("record_id", "")).strip()

        merged: Dict[str, Any] = dict(public_row)

        private = private_index.get(record_id)
        if private is None:
            merged.update(
                {
                    "excluded": False,
                    "exclude_reason": "",
                    "review_date": "",
                    "notes": "",
                }
            )
        else:
            exclude_reason = str(private.get("exclude_reason", "")).strip()
            merged.update(
                {
                    "excluded": exclude_reason != "",
                    "exclude_reason": exclude_reason,
                    "review_date": str(private.get("review_date", "")).strip(),
                    "notes": str(private.get("notes", "")).strip(),
                }
            )

        view_rows.append(merged)

    return view_rows


# =============================================================================
# I/O helpers
# =============================================================================

def _read_tsv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"TSV file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = list(reader.fieldnames) if reader.fieldnames is not None else []
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
        return rows, header


def _index_private_decisions(
    rows: Sequence[Dict[str, str]],
    header: Sequence[str],
) -> Dict[str, Dict[str, str]]:
    """
    Index registry_private.tsv by record_id.

    This expects at least the columns in PRIVATE_DECISIONS_EXPECTED_COLUMNS.
    Extra columns are ignored.

    Usage example
    -------------
        index = _index_private_decisions(rows, header)
    """
    header_set = set(header)
    required_missing = [c for c in PRIVATE_DECISIONS_EXPECTED_COLUMNS if c not in header_set]
    if required_missing:
        raise ValueError(
            f"registry_private.tsv missing required columns: {required_missing}. "
            f"Found: {list(header)}"
        )

    index: Dict[str, Dict[str, str]] = {}
    for row in rows:
        record_id = str(row.get("record_id", "")).strip()
        if not record_id:
            continue
        index[record_id] = {
            "exclude_reason": str(row.get("exclude_reason", "") or ""),
            "review_date": str(row.get("review_date", "") or ""),
            "notes": str(row.get("notes", "") or ""),
        }
    return index
