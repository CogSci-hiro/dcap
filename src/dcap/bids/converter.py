# =============================================================================
#                          BIDS: Converter entry
# =============================================================================
#
# Library entry points for source -> BIDS conversion.
# No CLI / argparse here.
#
# REVIEW
# =============================================================================

from pathlib import Path
from typing import List

from dcap.bids.config import BidsConvertConfig
from dcap.bids.heuristics import SourceItem, discover_source_items
from dcap.bids.pipeline import convert_one_run, convert_subject_extras


def convert_subject_to_bids(cfg: BidsConvertConfig) -> None:
    """
    Convert a subject worth of source recordings to a BIDS dataset.

    Parameters
    ----------
    cfg
        Conversion config.

    Usage example
    -------------
        convert_subject_to_bids(cfg)
    """
    cfg.bids_root.mkdir(parents=True, exist_ok=True)

    items: List[SourceItem] = list(discover_source_items(cfg.source_root))
    for item in items:
        convert_one_run(cfg=cfg, item=item)

    convert_subject_extras(cfg=cfg, items=items)
