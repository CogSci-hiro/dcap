# tests/bids/core/test_converter_end_to_end.py
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
def test_bids_convert_end_to_end_creates_expected_outputs(tmp_path: Path) -> None:
    # -------------------------------------------------------------------------
    # Setup: fake sourcedata + output dirs
    # -------------------------------------------------------------------------
    source_root = tmp_path / "sourcedata" / "Nic-Ele"
    bids_root = tmp_path / "bids"
    private_root = tmp_path / "private"

    source_root.mkdir(parents=True, exist_ok=True)
    bids_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)

    # TODO: create the minimum source files your task adapter expects.
    # Example placeholders (replace with what your code actually looks for):
    # (source_root / "sub-raw.edf").write_bytes(b"")  # if code only checks exists
    # (private_root / "subject_keys.yaml").write_text("...")

    # -------------------------------------------------------------------------
    # Act: call conversion entrypoint
    # -------------------------------------------------------------------------
    from dcap.bids.core.config import BidsCoreConfig
    from dcap.bids.core.converter import convert_subject

    cfg = BidsCoreConfig(
        # TODO: fill with the minimum required fields for your config
        source_root=source_root,
        bids_root=bids_root,
        private_root=private_root,
        dataset_id="TESTDATASET",
        line_freq=50,
        overwrite=True,
    )

    convert_subject(
        cfg=cfg,
        bids_subject="sub-001",
        dcap_id="Nic-Ele",
        datatype="ieeg",
        session=None,
        task_name="diapix",
    )

    # -------------------------------------------------------------------------
    # Assert: check existence of key outputs
    # -------------------------------------------------------------------------
    sub_dir = bids_root / "sub-001"
    assert sub_dir.exists()

    # You likely know your expected BIDS paths; start with a couple “anchor” files.
    # Examples (adjust to your naming conventions):
    ieeg_dir = sub_dir / "ieeg"
    assert ieeg_dir.exists()

    # Find at least one sidecar JSON
    jsons = list(ieeg_dir.rglob("*.json"))
    assert len(jsons) >= 1

    # Confirm no private ID leaked
    for p in bids_root.rglob("*"):
        if p.is_file() and p.suffix in {".json", ".tsv"}:
            text = p.read_text(errors="ignore")
            assert "Nic-Ele" not in text  # dcap_id should never appear in public BIDS
