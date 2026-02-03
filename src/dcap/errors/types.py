# =============================================================================
# =============================================================================
#                       #####################################
#                       #           DCAP ERROR TYPES        #
#                       #####################################
# =============================================================================
# =============================================================================

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class DcapError(Exception):
    message: str
    stage: str
    artifact: Optional[str] = None
    context: Optional[Mapping[str, Any]] = None
    cause: Optional[BaseException] = None

    def __init__(
        self,
        *,
        message: str,
        stage: str,
        artifact: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        # Initialize Exception with a readable message
        super().__init__(f"{stage}: {message}")

        # Dataclass fields (need object.__setattr__ because frozen=True)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "artifact", artifact)
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "cause", cause)


class DataValidationError(DcapError):
    """Inputs are missing/invalid (wrong columns, shapes, etc.)."""


class ArtifactBuildError(DcapError):
    """Failed to produce an artifact (figure, html panel, export file)."""


class ExternalToolError(DcapError):
    """Failure in an external dependency (Praat, MNE, subprocess tools)."""


class OptionalArtifactError(DcapError):
    """Artifact is allowed to fail without failing the whole run."""
