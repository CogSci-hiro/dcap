# =============================================================================
#                      BIDS Tasks: Registry / factories
# =============================================================================
#
# Central place to map task names -> task factories.
# Keeps CLI generic and prevents core from importing tasks.
#
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional

from dcap.bids.tasks.base import BidsTask


@dataclass(frozen=True, slots=True)
class TaskFactoryContext:
    """
    Context passed to task factories. This is *task-layer*, not core.

    Usage example
    -------------
        ctx = TaskFactoryContext(
            subject="NicEle",
            session=None,
            task_assets_dir=Path("assets/diapix"),
        )
    """

    subject: str
    session: Optional[str]
    task_assets_dir: Optional[Path]


TaskFactory = Callable[[TaskFactoryContext], BidsTask]


def get_task_factories() -> Mapping[str, TaskFactory]:
    """
    Return mapping task_name -> factory.

    Notes
    -----
    Import tasks lazily inside the function to avoid import side-effects and
    to keep startup fast.

    Usage example
    -------------
        factories = get_task_factories()
        task = factories["diapix"](ctx)
    """
    from dcap.bids.tasks.diapix.task import DiapixTask  # local import by design

    def make_diapix(ctx: TaskFactoryContext) -> BidsTask:
        if ctx.task_assets_dir is None:
            raise ValueError("diapix requires --task-assets-dir (directory with audio_onsets.tsv, stim_wav, atlas).")

        audio_onsets_tsv = ctx.task_assets_dir / "audio_onsets.tsv"
        stim_wav = ctx.task_assets_dir / "beeps_pre-task-1-sec_post-task-4-sec.wav"
        atlas_path = ctx.task_assets_dir / "elec2atlas.mat"

        return DiapixTask(
            subject_bids=ctx.subject,
            audio_onsets_tsv=audio_onsets_tsv,
            stim_wav=stim_wav,
            atlas_path=atlas_path,
        )

    return {
        "diapix": make_diapix,
        # "conversation": make_conversation,
        # "rest": make_rest,
    }


def resolve_task(task_name: str, ctx: TaskFactoryContext) -> BidsTask:
    factories = get_task_factories()
    key = str(task_name).strip().lower()

    if key not in factories:
        available = ", ".join(sorted(factories.keys()))
        raise ValueError(f"Unknown task {task_name!r}. Available tasks: {available}")

    return factories[key](ctx)
