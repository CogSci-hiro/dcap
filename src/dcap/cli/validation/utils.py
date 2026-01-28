# =============================================================================
#                         Validation: small utilities
# =============================================================================
import datetime as _dt
import re
from typing import Any, Optional


def is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def matches_pattern(value: Any, pattern: str) -> bool:
    if not isinstance(value, str):
        return False
    return re.match(pattern, value) is not None


def parse_iso_date(value: Any) -> Optional[_dt.date]:
    if not isinstance(value, str) or value.strip() == "":
        return None
    try:
        return _dt.date.fromisoformat(value)
    except ValueError:
        return None


def to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    return None
