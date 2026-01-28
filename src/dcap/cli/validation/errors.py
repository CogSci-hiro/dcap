# =============================================================================
#                           Validation: issues model
# =============================================================================
from dataclasses import dataclass
from typing import Literal


IssueLevel = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """
    A single validation issue.

    Usage example
    -------------
        issue = ValidationIssue(
            level="error",
            location="registry_public.tsv:row=3:subject",
            message="subject must match '^sub-\\d{3}$'",
        )
    """

    level: IssueLevel
    location: str
    message: str
