# src/dcap/registry/queries.py
# =============================================================================
#                        Registry queries: convenience API
# =============================================================================
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from dcap.registry.loader import load_private_registry, load_public_registry
from dcap.registry.view import build_registry_view, RegistryMergePolicy


@dataclass(frozen=True, slots=True)
class Registry:
    """
    Registry access object: loads public + private and exposes query helpers.

    Usage example
    -------------
        reg = Registry.from_paths(
            public_path=Path("registry_public.parquet"),
            private_root="env",
        )
        df_runs = reg.list_runs(task="conversation", qc="pass")
    """

    df_view: pd.DataFrame
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_paths(
        cls,
        public_path: Path,
        private_root: Optional[str | Path] = None,
        policy: Optional[RegistryMergePolicy] = None,
    ) -> "Registry":
        """
        Build a Registry from on-disk public registry and optional private overlay.

        Parameters
        ----------
        public_path
            Path to public registry CSV/Parquet.
        private_root
            None, "env", or a path.
        policy
            Merge policy.

        Returns
        -------
        Registry
            Loaded registry.

        Usage example
        -------------
            reg = Registry.from_paths(Path("registry_public.parquet"), private_root="env")
        """
        df_public = load_public_registry(public_path)
        df_private = load_private_registry(private_root)
        df_view, warnings = build_registry_view(df_public, df_private, policy=policy)
        return cls(df_view=df_view, warnings=tuple(warnings))

    def list_runs(
        self,
        task: Optional[str] = None,
        qc: Optional[str] = None,
        usable_only: bool = False,
    ) -> pd.DataFrame:
        """
        List registry rows filtered by task and qc.

        Parameters
        ----------
        task
            Task filter (BIDS task name).
        qc
            QC filter applied to effective_qc_status (e.g., "pass", "fail", "review", "unknown").
        usable_only
            If True, keep only rows where is_usable is True.

        Returns
        -------
        pandas.DataFrame
            Filtered view.

        Example output format
        ---------------------
        | subject | session | task         | run | datatype | effective_qc_status | is_usable |
        |---------|---------|--------------|-----|----------|---------------------|----------|
        | sub-001 | ses-01  | conversation | 1   | ieeg     | pass                | True     |

        Usage example
        -------------
            df = reg.list_runs(task="conversation", qc="pass", usable_only=True)
        """
        df = self.df_view
        if task is not None:
            df = df[df["task"] == task]
        if qc is not None:
            df = df[df["effective_qc_status"] == qc]
        if usable_only:
            df = df[df["is_usable"]]
        return df.reset_index(drop=True)

    def available_tasks(self, subject: str) -> list[str]:
        """
        List tasks available for a given subject (in public manifest).

        Usage example
        -------------
            tasks = reg.available_tasks("sub-001")
        """
        df = self.df_view
        tasks = sorted(df.loc[df["subject"] == subject, "task"].dropna().unique().tolist())
        return tasks

    def subjects_with_tasks(self, tasks: Iterable[str], usable_only: bool = False) -> list[str]:
        """
        List subjects that have all tasks in `tasks`.

        Parameters
        ----------
        tasks
            Iterable of task names.
        usable_only
            If True, requires rows to be is_usable.

        Usage example
        -------------
            subs = reg.subjects_with_tasks(["conversation", "localizer"], usable_only=True)
        """
        df = self.df_view
        if usable_only:
            df = df[df["is_usable"]]

        required = set(tasks)
        subjects = []
        for subject, sub_df in df.groupby("subject"):
            have = set(sub_df["task"].dropna().unique().tolist())
            if required.issubset(have):
                subjects.append(subject)
        return sorted(subjects)
