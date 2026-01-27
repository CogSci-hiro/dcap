"""
Registry loaders and validators.

Public layer:
- shareable, version-controlled indices

Private layer:
- local-only metadata referenced via DCAP_PRIVATE_ROOT

Important
---------
Private metadata must NEVER overwrite public columns. The merge is a left join
from public -> private, using JOIN_KEYS.

"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from dcap.config import get_private_root
from dcap.registry.schema import (
    JOIN_KEYS,
    PRIVATE_REQUIRED_COLUMNS,
    PUBLIC_REQUIRED_COLUMNS,
    QC_STATUS_ALLOWED,
    SchemaValidationReport,
)


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


def _read_table(path: Path) -> pd.DataFrame:
    """
    Read a registry table from CSV or Parquet.

    Parameters
    ----------
    path
        File path.

    Returns
    -------
    table
        Loaded table.

    Usage example
    ------------
        from pathlib import Path
        from dcap.registry.loader import _read_table

        df = _read_table(Path("registry_public.csv"))
    """
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def validate_public_registry(public_df: pd.DataFrame) -> SchemaValidationReport:
    """
    Validate the public registry schema.

    Parameters
    ----------
    public_df
        Public registry table.

    Returns
    -------
    report
        Validation report.

    Notes
    -----
    Expected DataFrame format (example):

    | subject | session | task | run | bids_root | qc_status |
    |---|---|---|---:|---|---|
    | sub-001 | ses-01 | conversation | 1 | /data/bids/conversation | pass |

    Usage example
    ------------
        import pandas as pd
        from dcap.registry.loader import validate_public_registry

        df = pd.DataFrame(
            [{"subject": "sub-001", "session": "ses-01", "task": "conversation", "run": 1,
              "bids_root": "/data/bids", "qc_status": "pass"}]
        )
        report = validate_public_registry(df)
        assert report.ok
    """
    errors: list[str] = []

    missing = [c for c in PUBLIC_REQUIRED_COLUMNS if c not in public_df.columns]
    if missing:
        errors.append(f"Public registry missing required columns: {missing}")

    if not errors:
        dup_mask = public_df.duplicated(list(JOIN_KEYS), keep=False)
        if bool(dup_mask.any()):
            dups = public_df.loc[dup_mask, list(JOIN_KEYS)].drop_duplicates().to_dict("records")
            errors.append(f"Public registry contains duplicate JOIN_KEYS rows: {dups[:10]}")

        if "qc_status" in public_df.columns:
            bad = sorted(set(public_df["qc_status"].dropna().astype(str)) - set(QC_STATUS_ALLOWED))
            if bad:
                errors.append(f"Public registry has invalid qc_status values: {bad}. Allowed={list(QC_STATUS_ALLOWED)}")

    return SchemaValidationReport(ok=(len(errors) == 0), errors=errors)


def validate_private_registry(private_df: pd.DataFrame) -> SchemaValidationReport:
    """
    Validate the private registry schema (only structural checks).

    Parameters
    ----------
    private_df
        Private registry table.

    Returns
    -------
    report
        Validation report.

    Usage example
    ------------
        import pandas as pd
        from dcap.registry.loader import validate_private_registry

        df = pd.DataFrame(
            [{"subject": "sub-001", "session": "ses-01", "task": "conversation", "run": 1,
              "subject_key": "HOSP123"}]
        )
        report = validate_private_registry(df)
        assert report.ok
    """
    errors: list[str] = []

    missing = [c for c in PRIVATE_REQUIRED_COLUMNS if c not in private_df.columns]
    if missing:
        errors.append(f"Private registry missing required columns: {missing}")

    if not errors:
        dup_mask = private_df.duplicated(list(JOIN_KEYS), keep=False)
        if bool(dup_mask.any()):
            dups = private_df.loc[dup_mask, list(JOIN_KEYS)].drop_duplicates().to_dict("records")
            errors.append(f"Private registry contains duplicate JOIN_KEYS rows: {dups[:10]}")

    return SchemaValidationReport(ok=(len(errors) == 0), errors=errors)


def load_registry_table(
    sources: RegistrySources,
    *,
    private_prefix: str = "private__",
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load and safely merge public + (optional) private registry tables.

    Parameters
    ----------
    sources
        Locations of public and private registry sources.
    private_prefix
        Prefix applied to all private-only columns during merge.
    validate
        If True, run schema validations and raise ValueError on failure.

    Returns
    -------
    registry
        Merged registry table.

    Notes
    -----
    Merge rules:
    - left join from public -> private on JOIN_KEYS
    - private columns never overwrite public columns
    - private-only columns are prefixed

    Usage example
    ------------
        from pathlib import Path
        from dcap.registry.loader import resolve_registry_sources, load_registry_table

        sources = resolve_registry_sources(Path("registry_public.parquet"))
        registry = load_registry_table(sources, private_prefix="private__")
        print(registry.columns)
    """
    public_df = _read_table(sources.public_registry_path)

    if validate:
        report = validate_public_registry(public_df)
        if not report.ok:
            raise ValueError("Invalid public registry:\n" + "\n".join(report.errors))

    if sources.private_registry_path is None:
        return public_df.reset_index(drop=True)

    private_df = _read_table(sources.private_registry_path)

    if validate:
        report = validate_private_registry(private_df)
        if not report.ok:
            raise ValueError("Invalid private registry:\n" + "\n".join(report.errors))

    # Determine private-only columns and prefix them
    private_only_cols = [c for c in private_df.columns if c not in public_df.columns and c not in JOIN_KEYS]
    private_renamed = private_df[list(JOIN_KEYS) + private_only_cols].copy()
    private_renamed = private_renamed.rename(columns={c: f"{private_prefix}{c}" for c in private_only_cols})

    merged = public_df.merge(private_renamed, how="left", on=list(JOIN_KEYS), validate="one_to_one")
    return merged.reset_index(drop=True)
