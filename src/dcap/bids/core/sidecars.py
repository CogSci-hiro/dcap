# =============================================================================
#                           BIDS: Sidecar writing
# =============================================================================
#
# Minimal optional sidecars beyond what MNE-BIDS writes.
#
# REVIEW
# =============================================================================

import json
from dataclasses import dataclass
from typing import Any, Dict

from mne_bids import BIDSPath


@dataclass(frozen=True)
class SidecarBundle:
    """
    Container for optional sidecar JSON content.

    Usage example
    -------------
        sidecars = SidecarBundle(modality_json={"TaskName": "diapix"})
        sidecars.write(bids_path)
    """

    modality_json: Dict[str, Any]

    def write(self, bids_path: BIDSPath) -> None:
        """
        Write sidecars next to the recording.

        Usage example
        -------------
            sidecars.write(bids_path)
        """
        bids_path.mkdir()

        json_path = bids_path.copy().update(extension=".json")
        with open(json_path.fpath, "w", encoding="utf-8") as f:
            json.dump(self.modality_json, f, indent=2, ensure_ascii=False)
