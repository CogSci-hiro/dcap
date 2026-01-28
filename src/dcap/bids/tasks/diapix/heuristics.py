# dcap/bids/tasks/diapix/heuristics.py
# =============================================================================
#                             DIAPIX HEURISTICS
# =============================================================================

from pathlib import Path
from typing import Final
import codecs


_MICRO_BYTE: Final[bytes] = b"\xb5"  # 'µ' in latin-1-ish encodings


def ensure_vhdr_utf8(vhdr_path: Path) -> None:
    """
    BrainVision headers sometimes include 'µ' which breaks UTF-8 parsing in MNE.

    This converts the offending byte to 'u' in-place.

    Parameters
    ----------
    vhdr_path
        Path to a `.vhdr` file.

    Usage example
    -------------
        ensure_vhdr_utf8(Path("conversation_1.vhdr"))
    """
    if not vhdr_path.exists():
        raise FileNotFoundError(f"Missing vhdr: {vhdr_path}")

    with codecs.open(vhdr_path, "rb") as f:
        raw_bytes = f.read()

    if _MICRO_BYTE not in raw_bytes:
        return

    patched = raw_bytes.replace(_MICRO_BYTE, b"u")
    with open(vhdr_path, "wb") as f:
        f.write(patched)
