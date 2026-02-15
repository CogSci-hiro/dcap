import logging

import pytest

from dcap.errors.policy import ErrorMode, ErrorPolicy, run_with_policy
from dcap.errors.record import ErrorLog
from dcap.errors.types import ArtifactBuildError, DataValidationError, OptionalArtifactError


def _raise_value_error() -> None:
    raise ValueError("boom")


def test_run_with_policy_raise_mode_wraps_and_raises() -> None:
    policy = ErrorPolicy(mode=ErrorMode.RAISE)

    with pytest.raises(ArtifactBuildError) as exc_info:
        run_with_policy(
            _raise_value_error,
            policy=policy,
            stage="viz.trf",
            artifact="kernel_plot",
            context={"patient": "P001"},
        )

    err = exc_info.value
    assert err.message == "boom"
    assert err.stage == "viz.trf"
    assert err.artifact == "kernel_plot"
    assert err.context == {"patient": "P001"}
    assert isinstance(err.cause, ValueError)


def test_run_with_policy_warn_mode_returns_fallback_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    policy = ErrorPolicy(mode=ErrorMode.WARN, logger_name="dcap.errors.tests")

    with caplog.at_level(logging.WARNING, logger=policy.logger_name):
        result = run_with_policy(
            _raise_value_error,
            policy=policy,
            stage="features.audio",
            on_error_return=123,
            optional=True,
        )

    assert result == 123
    assert any("features.audio: boom" in rec.message for rec in caplog.records)


def test_run_with_policy_collect_mode_records_and_can_skip_warning(caplog: pytest.LogCaptureFixture) -> None:
    policy = ErrorPolicy(mode=ErrorMode.COLLECT, logger_name="dcap.errors.tests", warn_on_collect=False)
    log = ErrorLog()

    with caplog.at_level(logging.WARNING, logger=policy.logger_name):
        result = run_with_policy(
            _raise_value_error,
            policy=policy,
            stage="registry.build",
            artifact="public_tsv",
            error_log=log,
            on_error_return=None,
        )

    assert result is None
    assert len(log.records) == 1
    assert log.records[0].stage == "registry.build"
    assert log.records[0].artifact == "public_tsv"
    assert log.records[0].error_type == "ArtifactBuildError"
    assert "registry.build: boom" in log.records[0].message
    assert not caplog.records


def test_run_with_policy_collect_mode_preserves_dcap_error_type() -> None:
    policy = ErrorPolicy(mode=ErrorMode.COLLECT, warn_on_collect=False)
    log = ErrorLog()

    def _raise_dcap_error() -> None:
        raise DataValidationError(message="missing column", stage="registry.validate")

    _ = run_with_policy(
        _raise_dcap_error,
        policy=policy,
        stage="registry.validate",
        error_log=log,
        on_error_return=None,
    )

    assert len(log.records) == 1
    assert log.records[0].error_type == "DataValidationError"
    assert log.records[0].message == "registry.validate: missing column"


def test_run_with_policy_wrap_as_overrides_default_wrapper() -> None:
    policy = ErrorPolicy(mode=ErrorMode.RAISE)

    with pytest.raises(DataValidationError) as exc_info:
        run_with_policy(
            _raise_value_error,
            policy=policy,
            stage="registry.validate",
            wrap_as=DataValidationError,
        )

    assert exc_info.value.message == "boom"
    assert exc_info.value.stage == "registry.validate"


def test_run_with_policy_optional_uses_optional_artifact_error() -> None:
    policy = ErrorPolicy(mode=ErrorMode.RAISE)

    with pytest.raises(OptionalArtifactError):
        run_with_policy(
            _raise_value_error,
            policy=policy,
            stage="viz.reports",
            optional=True,
        )

