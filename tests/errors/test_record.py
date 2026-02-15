import re

from dcap.errors.record import ErrorLog, ErrorRecord


def test_add_exception_creates_structured_record_with_copied_context() -> None:
    context = {"patient": "P001"}
    log = ErrorLog()

    log.add_exception(
        stage="viz.trf",
        artifact="kernel_plot",
        exc=ValueError("bad value"),
        context=context,
        traceback_str="traceback",
    )

    context["patient"] = "P999"
    record = log.records[0]

    assert len(log.records) == 1
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", record.timestamp_utc)
    assert record.stage == "viz.trf"
    assert record.artifact == "kernel_plot"
    assert record.error_type == "ValueError"
    assert record.message == "bad value"
    assert record.context == {"patient": "P001"}
    assert record.traceback_str == "traceback"


def test_error_log_has_errors_and_summary_lines_with_truncation() -> None:
    log = ErrorLog(
        records=[
            ErrorRecord("2026-02-01T00:00:00Z", "stage.a", "a1", "TypeA", "msg-a", None, None),
            ErrorRecord("2026-02-01T00:00:01Z", "stage.b", None, "TypeB", "msg-b", None, None),
            ErrorRecord("2026-02-01T00:00:02Z", "stage.c", "a3", "TypeC", "msg-c", None, None),
        ]
    )

    assert log.has_errors()

    lines = log.summary_lines(max_lines=2)
    assert lines == [
        "2026-02-01T00:00:00Z | stage.a [a1] | TypeA: msg-a",
        "2026-02-01T00:00:01Z | stage.b | TypeB: msg-b",
        "... (1 more)",
    ]


def test_error_log_has_errors_is_false_when_empty() -> None:
    log = ErrorLog()
    assert not log.has_errors()

