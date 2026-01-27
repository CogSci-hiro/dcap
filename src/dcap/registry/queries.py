"""
Registry query helpers (skeleton).
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import pandas as pd


@dataclass(frozen=True, slots=True)
class RunSelector:
    """
    A simple selector used to filter a registry table.

    Parameters
    ----------
    task
        BIDS task label.
    qc_status
        QC status filter (e.g. "pass").
    subject
        Optional subject filter.

    Usage example
    ------------
        from dcap.registry.queries import RunSelector

        selector = RunSelector(task="conversation", qc_status="pass")
    """
    task: Optional[str] = None
    qc_status: Optional[str] = None
    subject: Optional[str] = None


def list_runs(registry: pd.DataFrame, selector: RunSelector) -> pd.DataFrame:
    """
    List runs matching a selector.

    Parameters
    ----------
    registry
        Registry table.
    selector
        Run selection criteria.

    Returns
    -------
    runs
        Filtered registry table.

    Usage example
    ------------
        import pandas as pd
        from dcap.registry.queries import list_runs, RunSelector

        registry = pd.DataFrame(
            [
                {"subject": "sub-001", "task": "conversation", "qc_status": "pass"},
                {"subject": "sub-002", "task": "rest", "qc_status": "fail"},
            ]
        )
        runs = list_runs(registry, RunSelector(task="conversation", qc_status="pass"))
        assert len(runs) == 1
    """
    df = registry.copy()
    if selector.task is not None:
        df = df[df["task"] == selector.task]
    if selector.qc_status is not None:
        df = df[df["qc_status"] == selector.qc_status]
    if selector.subject is not None:
        df = df[df["subject"] == selector.subject]
    return df.reset_index(drop=True)


def subjects_with_tasks(registry: pd.DataFrame, tasks: Sequence[str]) -> list[str]:
    """
    Return subjects that have all requested tasks.

    Parameters
    ----------
    registry
        Registry table.
    tasks
        List of required tasks.

    Returns
    -------
    subjects
        Subjects that contain all tasks.

    Usage example
    ------------
        import pandas as pd
        from dcap.registry.queries import subjects_with_tasks

        registry = pd.DataFrame(
            [
                {"subject": "sub-001", "task": "conversation"},
                {"subject": "sub-001", "task": "rest"},
                {"subject": "sub-002", "task": "conversation"},
            ]
        )
        subjects = subjects_with_tasks(registry, ["conversation", "rest"])
        assert subjects == ["sub-001"]
    """
    needed = set(tasks)
    grouped = registry.groupby("subject")["task"].apply(lambda x: set(x))
    subjects = [sub for sub, have in grouped.items() if needed.issubset(have)]
    subjects.sort()
    return subjects
