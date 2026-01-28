"""
BIDS conversion for conversational recordings (placeholder).
"""
from pathlib import Path
from typing import Optional


def convert_conversation_to_bids(
    raw_root: Path,
    bids_root: Path,
    *,
    subject: str,
    session: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Convert conversational raw data into a BIDS dataset (skeleton).

    Parameters
    ----------
    raw_root
        Root directory of raw conversational data.
    bids_root
        Destination BIDS root directory.
    subject
        Anonymized BIDS subject label (e.g., "sub-001").
    session
        Optional BIDS session label.
    overwrite
        If True, allows overwriting existing outputs.

    Notes
    -----
    This function is intentionally a stub. Implementation must:
    - write BIDS-compliant filenames
    - produce required sidecars (e.g. channels.tsv)
    - handle irregular clinical acquisition gracefully

    Usage example
    ------------
        from pathlib import Path
        from dcap.bids.conversation import convert_conversation_to_bids

        convert_conversation_to_bids(
            raw_root=Path("/data/raw/conversation"),
            bids_root=Path("/data/bids/conversation_bids"),
            subject="sub-001",
            session="ses-01",
            overwrite=False,
        )
    """
    _ = (raw_root, bids_root, subject, session, overwrite)
    raise NotImplementedError("BIDS conversion not implemented yet.")
