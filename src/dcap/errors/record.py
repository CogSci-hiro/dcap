# =============================================================================
# =============================================================================
#                       #####################################
#                       #         ERROR RECORDING           #
#                       #####################################
# =============================================================================
# =============================================================================

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Mapping, Optional


@dataclass(frozen=True)
class ErrorRecord:
    """
    Structured record of a failure that occurred during a run.

    Parameters
    ----------
    timestamp_utc:
        ISO timestamp (UTC) when recorded.
    stage:
        Stage identifier (e.g. "viz.trf").
    artifact:
        Optional artifact identifier.
    error_type:
        Exception class name.
    message:
        Human-readable summary.
    context:
        Optional structured context dict.
    traceback_str:
        Optional traceback string for debugging.

    Usage example
    -------------
        record = ErrorRecord(
            timestamp_utc="2026-02-03T20:12:34Z",
            stage="viz.trf",
            artifact="trf_kernel_heatmap",
            error_type="ValueError",
            message="kernels contains NaN",
            context={"patient": "P001"},
            traceback_str="Traceback ...",
        )
    """
    timestamp_utc: str
    stage: str
    artifact: Optional[str]
    error_type: str
    message: str
    context: Optional[Mapping[str, Any]]
    traceback_str: Optional[str]


@dataclass
class ErrorLog:
    """
    Collects error records across a run.

    Usage example
    -------------
        errors = ErrorLog()
        errors.add_exception(stage="viz.trf", artifact="heatmap", exc=exc, context={"patient": "P001"})
        if errors.has_errors():
            print(errors.summary_lines())
    """
    records: List[ErrorRecord] = field(default_factory=list)

    def add_record(self, record: ErrorRecord) -> None:
        self.records.append(record)

    def add_exception(
        self,
        *,
        stage: str,
        exc: BaseException,
        artifact: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        traceback_str: Optional[str] = None,
    ) -> None:
        timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        record = ErrorRecord(
            timestamp_utc=timestamp_utc,
            stage=stage,
            artifact=artifact,
            error_type=type(exc).__name__,
            message=str(exc),
            context=dict(context) if context is not None else None,
            traceback_str=traceback_str,
        )
        self.add_record(record)

    def has_errors(self) -> bool:
        return len(self.records) > 0

    def summary_lines(self, *, max_lines: int = 50) -> List[str]:
        lines: List[str] = []
        for r in self.records[:max_lines]:
            art = f" [{r.artifact}]" if r.artifact else ""
            lines.append(f"{r.timestamp_utc} | {r.stage}{art} | {r.error_type}: {r.message}")
        if len(self.records) > max_lines:
            lines.append(f"... ({len(self.records) - max_lines} more)")
        return lines
