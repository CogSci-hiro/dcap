import pytest

from dcap.bids.tasks.diapix import convert_conversation_to_bids


def test_convert_conversation_to_bids_is_stub() -> None:
    with pytest.raises(NotImplementedError):
        convert_conversation_to_bids(
            raw_root=__import__("pathlib").Path("/tmp/raw"),
            bids_root=__import__("pathlib").Path("/tmp/bids"),
            subject="sub-001",
        )
