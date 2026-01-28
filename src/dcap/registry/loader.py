# src/dcap/registry/loader.py
# =============================================================================
#                      Registry loader: public + private
# =============================================================================
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from dcap.registry.schema.private import PrivateRegistrySchema
from dcap.registry.schema.public import PublicRegistrySchema


@dataclass(frozen=True, slots=True)
class RegistryPaths:
    """
    Standard filenames under a DCAP private root.

    Usage example
    -------------
        paths = RegistryPaths()
        private_registry_path = paths.private_registry_path(Path("~/.dcap_private").expanduser())
    """

    private_registry_filename: str = "registry_private.parquet"

    def private_registry_path(self, private_root: Path) -> Path:
        """
        Compute private registry path under a private root.

        Usage example
        -------------
            paths = RegistryPaths()
            p = paths.private_registry_path(Path("~/.dcap_private").expanduser())
        """
        return private_root / self.private_registry_filename


def load_public_registry(path: Path) -> pd.DataFrame:
    """
    Load the public registry.

    Parameters
    ----------
    path
        Path to a CSV or Parquet file.

    Returns
    -------
    pandas.DataFrame
        Public registry DataFrame.

    Example format
    --------------
    | dataset_id | bids_root      | subject  | session | task         | run | datatype | record_id                             |
    |-----------:|----------------|----------|---------|--------------|-----|----------|--------------------------------------|
    | siteA_2024 | /data/bidsA    | sub-001  | ses-01  | conversation | 1   | ieeg     | siteA_2024|sub-001|ses-01|...|ieeg |

    Usage example
    -------------
        df_public = load_public_registry(Path("registry_public.parquet"))
    """
    if not path.exists():
        raise FileNotFoundError(f"Public registry not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported public registry extension: {path.suffix}")

    PublicRegistrySchema().validate(df)
    return df


def resolve_private_root(private_root: Optional[str | Path]) -> Optional[Path]:
    """
    Resolve private root path.

    Parameters
    ----------
    private_root
        - None: no private overlay
        - "env": read from DCAP_PRIVATE_ROOT
        - path-like: explicit path

    Returns
    -------
    pathlib.Path | None
        Resolved private root, or None.

    Usage example
    -------------
        root = resolve_private_root("env")
    """
    if private_root is None:
        return None
    if isinstance(private_root, str) and private_root.lower() == "env":
        env_val = os.environ.get("DCAP_PRIVATE_ROOT", "").strip()
        if not env_val:
            return None
        return Path(env_val).expanduser().resolve()
    return Path(private_root).expanduser().resolve()


def load_private_registry(private_root: Optional[str | Path]) -> pd.DataFrame:
    """
    Load the private registry overlay.

    If private_root is None or missing/unset, returns an empty DataFrame.

    Parameters
    ----------
    private_root
        See resolve_private_root.

    Returns
    -------
    pandas.DataFrame
        Private registry overlay (possibly empty).

    Example format
    --------------
    | record_id                             | qc_status | exclude | exclude_reason | notes            |
    |--------------------------------------|----------:|--------:|----------------|------------------|
    | siteA_2024|sub-001|ses-01|...|ieeg    | pass      | False   |                | "noisy but ok"   |

    Usage example
    -------------
        df_private = load_private_registry("env")
    """
    root = resolve_private_root(private_root)
    if root is None:
        return pd.DataFrame({"record_id": pd.Series(dtype="string")})

    paths = RegistryPaths()
    reg_path = paths.private_registry_path(root)
    if not reg_path.exists():
        # Empty overlay is acceptable; user might not have started QC yet.
        return pd.DataFrame({"record_id": pd.Series(dtype="string")})

    df = pd.read_parquet(reg_path)
    PrivateRegistrySchema().validate(df)
    return df
