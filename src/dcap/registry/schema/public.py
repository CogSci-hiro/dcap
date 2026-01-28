# src/dcap/registry/schema/public.py
# =============================================================================
#                       Registry schema: public manifest
# =============================================================================
from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True, slots=True)
class PublicRegistrySchema:
    """
    Schema for the public/shareable registry manifest.

    Required columns
    ----------------
    - dataset_id
    - bids_root
    - subject
    - session
    - task
    - run
    - datatype
    - record_id

    Optional columns
    ----------------
    - bids_relpath
    - exists

    Usage example
    -------------
        schema = PublicRegistrySchema()
        schema.validate(df_public)
    """

    required_columns: tuple[str, ...] = (
        "dataset_id",
        "bids_root",
        "subject",
        "session",
        "task",
        "run",
        "datatype",
        "record_id",
    )

    optional_columns: tuple[str, ...] = ("bids_relpath", "exists")

    def validate(self, df: pd.DataFrame) -> None:
        """
        Validate public registry columns.

        Parameters
        ----------
        df
            Public registry DataFrame.

        Raises
        ------
        ValueError
            If required columns are missing.

        Usage example
        -------------
            schema = PublicRegistrySchema()
            schema.validate(df_public)
        """
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Public registry missing required columns: {missing}")

        unexpected = [c for c in df.columns if c not in (*self.required_columns, *self.optional_columns)]
        if unexpected:
            # Strict by default: you can loosen later if you want.
            raise ValueError(f"Public registry has unexpected columns (strict mode): {unexpected}")
