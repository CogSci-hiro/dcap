# tests/test_registry_private.py
# =============================================================================
#                      Tests: private registry overlay
# =============================================================================
from pathlib import Path

import pandas as pd
import pytest

from dcap.registry.view import build_registry_view, RegistryMergePolicy


def _make_public_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": ["siteA_2024", "siteA_2024"],
            "bids_root": ["/data/bidsA", "/data/bidsA"],
            "subject": ["sub-001", "sub-001"],
            "session": ["ses-01", "ses-01"],
            "task": ["conversation", "conversation"],
            "run": ["1", "2"],
            "datatype": ["ieeg", "ieeg"],
            "record_id": [
                "siteA_2024|sub-001|ses-01|conversation|1|ieeg",
                "siteA_2024|sub-001|ses-01|conversation|2|ieeg",
            ],
        }
    )


def test_private_orphan_rows_are_ignored_with_warning() -> None:
    df_public = _make_public_df()
    df_private = pd.DataFrame(
        {
            "record_id": [
                "siteA_2024|sub-001|ses-01|conversation|1|ieeg",
                "siteA_2024|sub-999|ses-01|conversation|1|ieeg",  # orphan
            ],
            "qc_status": ["pass", "fail"],
        }
    )

    df_view, warnings = build_registry_view(df_public, df_private)
    assert len(warnings) == 1
    assert "orphan" in warnings[0].lower()
    assert len(df_view) == 2
    assert df_view.loc[df_view["run"] == "1", "effective_qc_status"].iloc[0] == "pass"


def test_private_cannot_override_protected_columns() -> None:
    df_public = _make_public_df()
    df_private = pd.DataFrame(
        {
            "record_id": ["siteA_2024|sub-001|ses-01|conversation|1|ieeg"],
            "subject": ["sub-999"],  # forbidden
            "qc_status": ["pass"],
        }
    )

    with pytest.raises(ValueError, match="protected columns"):
        build_registry_view(df_public, df_private)


def test_no_private_columns_present_if_not_provided() -> None:
    df_public = _make_public_df()
    df_private = pd.DataFrame({"record_id": pd.Series(dtype="string")})

    df_view, warnings = build_registry_view(df_public, df_private)

    assert len(warnings) == 0
    assert "effective_qc_status" in df_view.columns
    assert "is_usable" in df_view.columns
    # qc_status exists but is NA -> effective becomes "unknown"
    assert (df_view["effective_qc_status"] == "unknown").all()


def test_is_usable_logic() -> None:
    df_public = _make_public_df()
    df_private = pd.DataFrame(
        {
            "record_id": [
                "siteA_2024|sub-001|ses-01|conversation|1|ieeg",
                "siteA_2024|sub-001|ses-01|conversation|2|ieeg",
            ],
            "qc_status": ["pass", "pass"],
            "exclude": [False, True],
        }
    )

    df_view, _ = build_registry_view(df_public, df_private)
    usable = df_view.set_index("run")["is_usable"].to_dict()
    assert usable["1"] is True
    assert usable["2"] is False


def test_private_notes_do_not_affect_protected_identity_fields() -> None:
    df_public = _make_public_df()
    df_private = pd.DataFrame(
        {
            "record_id": ["siteA_2024|sub-001|ses-01|conversation|1|ieeg"],
            "notes": ["contains sensitive info"],
            "qc_status": ["review"],
        }
    )
    df_view, _ = build_registry_view(df_public, df_private)
    assert df_view.loc[df_view["run"] == "1", "subject"].iloc[0] == "sub-001"
    assert df_view.loc[df_view["run"] == "1", "effective_qc_status"].iloc[0] == "review"
