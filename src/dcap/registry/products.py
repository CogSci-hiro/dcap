# dcap/registry/products.py
# =============================================================================
#                      Registry: sanitized products
# =============================================================================
#
# Build shareable indices (no sensitive fields) from the runtime registry view.
#
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import csv


# =============================================================================
# Constants
# =============================================================================

AVAILABILITY_COLUMNS: Tuple[str, ...] = (
    "dataset_id",
    "task",
    "protocol_id",
    "total_records",
    "usable_records",
    "excluded_records",
    "usable_fraction",
)


# =============================================================================
# Public API
# =============================================================================

def write_availability_index_by_task(
    *,
    registry_view_rows: Sequence[Dict[str, Any]],
    out_tsv: Path,
) -> Path:
    """
    Write a sanitized availability index by dataset_id × task × protocol_id.

    Parameters
    ----------
    registry_view_rows
        Output of build_registry_view(...).
    out_tsv
        Output TSV path.

    Returns
    -------
    Path
        Written TSV path.

    Notes
    -----
    This product is explicitly sanitized:
    - No subject identifiers beyond dataset/task/protocol aggregation
    - No notes
    - No record_ids

    Output format example
    ---------------------
    | dataset_id | task         | protocol_id | total_records | usable_records | excluded_records | usable_fraction |
    |-----------|--------------|-------------|--------------:|---------------:|-----------------:|----------------:|
    | Timone2025| conversation | prot-01     | 12            | 10             | 2                | 0.8333          |

    Usage example
    -------------
        from pathlib import Path
        from dcap.registry.view import build_registry_view
        from dcap.registry.products import write_availability_index_by_task

        view_rows = build_registry_view(
            public_registry=Path("registry_public.tsv"),
            private_registry=Path("/secure/DCAP_PRIVATE_ROOT/registry_private.tsv"),
        )

        write_availability_index_by_task(
            registry_view_rows=view_rows,
            out_tsv=Path("availability_by_task.tsv"),
        )
    """
    grouped: Dict[Tuple[str, str, str], Dict[str, int]] = {}

    for row in registry_view_rows:
        dataset_id = str(row.get("dataset_id", "")).strip()
        task = str(row.get("task", "")).strip()
        protocol_id = str(row.get("protocol_id", "")).strip()

        key = (dataset_id, task, protocol_id)
        if key not in grouped:
            grouped[key] = {"total": 0, "excluded": 0}

        grouped[key]["total"] += 1
        if bool(row.get("excluded", False)):
            grouped[key]["excluded"] += 1

    output_rows: List[Dict[str, str]] = []
    for (dataset_id, task, protocol_id), counts in sorted(grouped.items()):
        total = counts["total"]
        excluded = counts["excluded"]
        usable = total - excluded
        usable_fraction = (usable / total) if total > 0 else 0.0

        output_rows.append(
            {
                "dataset_id": dataset_id,
                "task": task,
                "protocol_id": protocol_id,
                "total_records": str(total),
                "usable_records": str(usable),
                "excluded_records": str(excluded),
                "usable_fraction": f"{usable_fraction:.4f}",
            }
        )

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(AVAILABILITY_COLUMNS), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    return out_tsv
