# =============================================================================
# =============================================================================
#                       #####################################
#                       #           ERROR POLICY            #
#                       #####################################
# =============================================================================
# =============================================================================

import logging
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, TypeVar

from .record import ErrorLog
from .types import ArtifactBuildError, DcapError, OptionalArtifactError

T = TypeVar("T")


class ErrorMode(str, Enum):
    """
    How to handle failures.

    - RAISE: raise immediately
    - WARN: log warning and return fallback
    - COLLECT: record in ErrorLog (and optionally warn) and return fallback
    - SILENT: return fallback only (strongly discouraged)
    """
    RAISE = "raise"
    WARN = "warn"
    COLLECT = "collect"
    SILENT = "silent"


@dataclass(frozen=True)
class ErrorPolicy:
    """
    Error handling policy for a run or subsystem.

    Parameters
    ----------
    mode:
        ErrorMode behavior.
    logger_name:
        Logger to use for warnings/errors.
    warn_on_collect:
        If True, also log warnings when collecting.
    """
    mode: ErrorMode = ErrorMode.COLLECT
    logger_name: str = "dcap"
    warn_on_collect: bool = True

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(self.logger_name)


def run_with_policy(
    fn: Callable[[], T],
    *,
    policy: ErrorPolicy,
    stage: str,
    artifact: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
    error_log: Optional[ErrorLog] = None,
    on_error_return: Optional[T] = None,
    wrap_as: Optional[type[DcapError]] = None,
    optional: bool = False,
) -> Optional[T]:
    """
    Execute `fn()` under an error policy.

    Parameters
    ----------
    fn:
        Callable with no args.
    policy:
        Error policy (raise/warn/collect/silent).
    stage:
        Stage identifier string (e.g., "viz.trf").
    artifact:
        Optional artifact identifier.
    context:
        Optional structured context (patient, run_id, etc.).
    error_log:
        Optional ErrorLog to record failures when mode=COLLECT.
    on_error_return:
        Fallback value returned on non-RAISE modes.
    wrap_as:
        Optional DcapError subclass to wrap unknown exceptions into.
    optional:
        If True, wraps as OptionalArtifactError by default (unless wrap_as is set).

    Returns
    -------
    result or fallback:
        Returns `fn()` result on success, else fallback (unless RAISE).

    Usage example
    -------------
        from dcap.errors import ErrorPolicy, ErrorMode, run_with_policy
        from dcap.errors.record import ErrorLog

        errors = ErrorLog()
        policy = ErrorPolicy(mode=ErrorMode.COLLECT)

        fig = run_with_policy(
            lambda: plot_trf_kernels_heatmap(...),
            policy=policy,
            stage="viz.trf",
            artifact="trf_kernel_heatmap",
            context={"patient": "P001"},
            error_log=errors,
            on_error_return=None,
            optional=True,
        )
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        # Wrap raw exceptions into a DCAP domain error if requested.
        if isinstance(exc, DcapError):
            dcap_exc: DcapError = exc
        else:
            error_cls: type[DcapError]
            if wrap_as is not None:
                error_cls = wrap_as
            else:
                error_cls = OptionalArtifactError if optional else ArtifactBuildError

            dcap_exc = error_cls(
                message=str(exc),
                stage=stage,
                artifact=artifact,
                context=context,
                cause=exc,
            )

        tb_str = traceback.format_exc()

        if policy.mode == ErrorMode.RAISE:
            raise dcap_exc from exc

        if policy.mode == ErrorMode.WARN:
            policy.logger.warning("%s\n%s", str(dcap_exc), tb_str)
            return on_error_return

        if policy.mode == ErrorMode.COLLECT:
            if error_log is not None:
                error_log.add_exception(stage=stage, artifact=artifact, exc=dcap_exc, context=context, traceback_str=tb_str)
            if policy.warn_on_collect:
                policy.logger.warning("%s", str(dcap_exc))
            return on_error_return

        # SILENT (discouraged)
        return on_error_return
