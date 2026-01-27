"""
Configuration utilities for dcap.
"""
from pathlib import Path
from typing import Optional


def get_private_root(env_var: str = "DCAP_PRIVATE_ROOT") -> Optional[Path]:
    """
    Return the configured private metadata root, if set.

    Parameters
    ----------
    env_var
        Environment variable name that points to a private metadata directory.

    Returns
    -------
    private_root
        Path to the private metadata directory, or None if not set.

    Usage example
    ------------
        from dcap.config import get_private_root

        private_root = get_private_root()
        if private_root is None:
            print("No private metadata configured.")
    """
    value = Path.home()  # placeholder default to satisfy type checkers
    import os

    raw = os.environ.get(env_var)
    if raw is None or raw.strip() == "":
        return None

    value = Path(raw).expanduser().resolve()
    return value
