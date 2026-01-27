"""
BIDS conversion for resting-state recordings (placeholder).
"""
from pathlib import Path
from typing import Optional


def convert_rest_to_bids(
    raw_root: Path,
    bids_root: Path,
    *,
    subject: str,
    session: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Convert resting-state raw data into a BIDS dataset (skeleton).

    Parameters
    ----------
    raw_root
        Root directory of raw resting-state data.
    bids_root
        Destination BIDS root directory.
    subject
        Anonymized BIDS subject label (e.g., "sub-001").
    session
        Optional BIDS session label.
    overwrite
        If True, allows overwriting existing outputs.

    Usage example
    ------------
        from pathlib import Path
        from dcap.bids.rest import convert_rest_to_bids

        convert_rest_to_bids(
            raw_root=Path("/data/raw/rest"),
            bids_root=Path("/data/bids/rest_bids"),
            subject="sub-002",
            session=None,
            overwrite=False,
        )
    """
    _ = (raw_root, bids_root, subject, session, overwrite)
    raise NotImplementedError("BIDS conversion not implemented yet.")
