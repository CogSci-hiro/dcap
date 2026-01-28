# =============================================================================
#                         Validation: schema spec loader
# =============================================================================
from pathlib import Path
from typing import Any, Dict

import yaml


def load_spec(path: Path) -> Dict[str, Any]:
    """
    Load a dcap machine-readable schema spec (YAML).

    Parameters
    ----------
    path
        Path to a schema YAML file.

    Returns
    -------
    dict
        Parsed YAML dictionary.

    Usage example
    -------------
        spec = load_spec(Path("SPEC_registry_public.schema.yaml"))
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Spec must be a YAML mapping (dict), got {type(data)}: {path}")
    return data
