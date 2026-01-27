import pandas as pd

from dcap.registry.queries import RunSelector, list_runs, subjects_with_tasks


def test_list_runs_filters_by_task_and_qc_status() -> None:
    registry = pd.DataFrame(
        [
            {"subject": "sub-001", "task": "conversation", "qc_status": "pass"},
            {"subject": "sub-001", "task": "rest", "qc_status": "pass"},
            {"subject": "sub-002", "task": "conversation", "qc_status": "fail"},
        ]
    )

    runs = list_runs(registry, RunSelector(task="conversation", qc_status="pass"))
    assert len(runs) == 1
    assert runs.loc[0, "subject"] == "sub-001"


def test_subjects_with_tasks_returns_subjects_having_all_tasks() -> None:
    registry = pd.DataFrame(
        [
            {"subject": "sub-001", "task": "conversation"},
            {"subject": "sub-001", "task": "rest"},
            {"subject": "sub-002", "task": "conversation"},
        ]
    )

    subjects = subjects_with_tasks(registry, ["conversation", "rest"])
    assert subjects == ["sub-001"]
