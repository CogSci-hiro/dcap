"""
Registry loaders (skeleton).

Public layer:
- shareable, version-controlled indices

Private layer:
- local-only metadata referenced via DCAP_PRIVATE_ROOT
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from dcap.config import get_private_root


@dataclass(frozen=True, slots=True)
class RegistrySources:
    """
    Locations of registry source files.

    Parameters
    ----------
    public_registry_path
        Path to the public/shareable registry index file.
    private_registry_path
        Path to the private registry file, if available.

    Usage example
    ------------
        from pathlib import Path
        from dcap.registry.loader import RegistrySources

        sources = RegistrySources(
            public_registry_path=Path("registry_public.parquet"),
            private_registry_path=None,
        )
    """
    public_registry_path: Path
    private_registry_path: Optional[Path]


def resolve_registry_sources(
    public_registry_path: Path,
    *,
    private_filename: str = "registry_private.parquet",
) -> RegistrySources:
    """
    Resolve public and private registry sources.

    Parameters
    ----------
    public_registry_path
        Path to the public registry file (shareable).
    private_filename
        Expected filename under DCAP_PRIVATE_ROOT.

    Returns
    -------
    sources
        Resolved registry sources.

    Usage example
    ------------
        from pathlib import Path
        from dcap.registry.loader import resolve_registry_sources

        sources = resolve_registry_sources(Path("registry_public.parquet"))
    """
    private_root = get_private_root()
    private_path = None
    if private_root is not None:
        candidate = (private_root / private_filename).resolve()
        if candidate.exists():
            private_path = candidate

    return RegistrySources(public_registry_path=public_registry_path, private_registry_path=private_path)


def load_registry_table(sources: RegistrySources) -> pd.DataFrame:
    """
    Load and (optionally) merge public + private registry tables.

    Parameters
    ----------
    sources
        Locations of public and private registry sources.

    Returns
    -------
    registry
        Merged registry table.

    Notes
    -----
    This is a skeleton. Merge rules must be explicitly defined
    to avoid accidental leakage of private identifiers.

    Usage example
    ------------
        from pathlib import Path
        from dcap.registry.loader import resolve_registry_sources, load_registry_table

        sources = resolve_registry_sources(Path("registry_public.parquet"))
        registry = load_registry_table(sources)
        print(registry.shape)
    """
    public_path = sources.public_registry_path
    if public_path.suffix.lower() == ".parquet":
        public_df = pd.read_parquet(public_path)
    else:
        public_df = pd.read_csv(public_path)

    if sources.private_registry_path is None:
        return public_df

    private_path = sources.private_registry_path
    if private_path.suffix.lower() == ".parquet":
        private_df = pd.read_parquet(private_path)
    else:
        private_df = pd.read_csv(private_path)

    # Placeholder merge: concat columns by index (unsafe, to be replaced)
    # Implementation MUST define join keys explicitly (e.g., subject/session/task/run).
    merged = public_df.copy()
    for col in private_df.columns:
        if col in merged.columns:
            continue
        merged[col] = private_df[col]

    return merged
