"""
BIDS and signal-level common (placeholder).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True, slots=True)
class QcResult:
    """
    Result of a QC/common run.

    Parameters
    ----------
    ok
        Whether the dataset passed checks.
    report_path
        Optional path to a human-readable report (e.g., HTML/PDF).
    summary_path
        Optional path to a machine-readable summary (e.g., JSON/CSV/Parquet).

    Usage example
    ------------
        from pathlib import Path
        from dcap.qc.validate import run_qc

        result = run_qc(Path("/data/bids/conversation_bids"))
        assert isinstance(result.ok, bool)
    """
    ok: bool
    report_path: Optional[Path] = None
    summary_path: Optional[Path] = None


def run_qc(bids_root: Path) -> QcResult:
    """
    Run dataset QC/common (skeleton).

    Parameters
    ----------
    bids_root
        Path to a BIDS dataset root.

    Returns
    -------
    result
        QC result placeholder.

    Usage example
    ------------
        from pathlib import Path
        from dcap.qc.validate import run_qc

        result = run_qc(Path("/data/bids/conversation_bids"))
        print(result.ok)
    """
    _ = bids_root
    return QcResult(ok=False)
