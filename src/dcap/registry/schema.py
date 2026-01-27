"""
Registry schema definitions.

The registry exists in two layers:
- public (shareable; version-controlled)
- private (local-only; NEVER committed)

The merged registry is produced at runtime by joining on the required keys.

Required join keys
------------------
- subject
- session
- task
- run

Usage example
------------
    from dcap.registry.schema import PUBLIC_REQUIRED_COLUMNS, JOIN_KEYS

    assert set(JOIN_KEYS).issubset(PUBLIC_REQUIRED_COLUMNS)
"""
from dataclasses import dataclass
from typing import Final, Sequence

# NOTE: User preference is to avoid `from __future__ import annotations`.

JOIN_KEYS: Final[tuple[str, ...]] = ("subject", "session", "task", "run")

QC_STATUS_ALLOWED: Final[tuple[str, ...]] = ("pass", "fail", "review", "unknown")

PUBLIC_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "subject",
    "session",
    "task",
    "run",
    "bids_root",
    "qc_status",
)

PUBLIC_OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "exclude_reason",
    "dataset",
    "notes_public",
)

PRIVATE_REQUIRED_COLUMNS: Final[tuple[str, ...]] = JOIN_KEYS


@dataclass(frozen=True, slots=True)
class SchemaValidationReport:
    """
    Summary of registry schema validation.

    Parameters
    ----------
    ok
        Whether validation succeeded.
    errors
        List of human-readable error messages.

    Usage example
    ------------
        from dcap.registry.schema import SchemaValidationReport

        report = SchemaValidationReport(ok=True, errors=[])
        assert report.ok
    """
    ok: bool
    errors: list[str]
