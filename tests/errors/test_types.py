from dataclasses import FrozenInstanceError

import pytest

from dcap.errors.types import (
    ArtifactBuildError,
    DataValidationError,
    DcapError,
    ExternalToolError,
    OptionalArtifactError,
)


def test_dcap_error_sets_fields_and_exception_message() -> None:
    err = DcapError(
        message="failed",
        stage="viz.trf",
        artifact="heatmap",
        context={"patient": "P001"},
    )
    assert str(err) == "viz.trf: failed"
    assert err.message == "failed"
    assert err.stage == "viz.trf"
    assert err.artifact == "heatmap"
    assert err.context == {"patient": "P001"}
    assert err.cause is None


def test_dcap_error_is_frozen() -> None:
    err = DcapError(message="failed", stage="registry.validate")
    with pytest.raises(FrozenInstanceError):
        err.stage = "other.stage"  # type: ignore[misc]


@pytest.mark.parametrize(
    "error_cls",
    [DataValidationError, ArtifactBuildError, ExternalToolError, OptionalArtifactError],
)
def test_error_subclasses_are_dcap_errors(error_cls: type[DcapError]) -> None:
    err = error_cls(message="failed", stage="x")
    assert isinstance(err, DcapError)
    assert str(err) == "x: failed"

