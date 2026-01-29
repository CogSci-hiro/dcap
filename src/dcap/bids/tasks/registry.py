# =============================================================================
#                      BIDS Tasks: Registry / factories
# =============================================================================
#
# Central place to map task names -> task factories.
# Keeps CLI generic and prevents core from importing tasks.
#
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

from dcap.bids.tasks.base import BidsTask
from dcap.bids.tasks.subject_map import load_subject_reid_map


@dataclass(frozen=True, slots=True)
class TaskFactoryContext:
    """
    Context passed to task factories.

    Notes
    -----
    - `bids_subject` is BIDS-facing (may be "001" or "sub-001").
    - `dcap_id` is resolved from private YAML; never written to BIDS.

    Usage example
    -------------
        ctx = TaskFactoryContext(
            task_name="diapix",
            dataset_id="Timone2025",
            bids_subject="sub-001",
            session=None,
            private_root=Path("/private"),
            subject_map_yaml=Path("/private/subject_reid_map.yml"),
            task_assets_dir=Path("/private/assets/diapix"),
        )
    """

    task_name: str
    dataset_id: str
    bids_subject: str
    session: Optional[str]
    private_root: Optional[Path]
    subject_map_yaml: Optional[Path]
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

        task = resolve_task(
            TaskFactoryContext(
                task_name="diapix",
                dataset_id=ctx.dataset_id,
                bids_subject=ctx.bids_subject,
                session=ctx.session,
                private_root=ctx.private_root,
                subject_map_yaml=ctx.subject_map_yaml,
                task_assets_dir=ctx.task_assets_dir,
            )
        )

        return task

    return {
        "diapix": make_diapix,
        # "conversation": make_conversation,
        # "rest": make_rest,
    }


def resolve_task(ctx: TaskFactoryContext) -> BidsTask:
    """
    Resolve a task instance from a task name + context.

    Usage example
    -------------
        task = resolve_task(ctx)
    """
    factories = _get_task_factories()
    name = str(ctx.task_name).strip().lower()

    if name not in factories:
        available = ", ".join(sorted(factories.keys()))
        raise ValueError(f"Unknown task {ctx.task_name!r}. Available tasks: {available}")

    return factories[name](ctx)


# =============================================================================
# Internal helpers
# =============================================================================

def _get_task_factories() -> Mapping[str, TaskFactory]:
    """
    Map task name -> factory.

    Note: imports are intentionally local to avoid import side effects.
    """
    from dcap.bids.tasks.diapix.task import DiapixTask

    def make_diapix(ctx: TaskFactoryContext) -> BidsTask:
        dcap_id = _resolve_dcap_id(
            dataset_id=ctx.dataset_id,
            bids_subject=ctx.bids_subject,
            private_root=ctx.private_root,
            subject_map_yaml=ctx.subject_map_yaml,
        )

        if ctx.task_assets_dir is None:
            raise ValueError(
                "diapix requires --task-assets-dir (directory containing audio_onsets.tsv, stim wav, atlas, etc.)."
            )

        task_assets_dir = ctx.task_assets_dir.expanduser().resolve()
        audio_onsets_tsv = task_assets_dir / "audio_onsets.tsv"
        stim_wav = task_assets_dir / "beeps_pre-task-1-sec_post-task-4-sec.wav"
        atlas_path = task_assets_dir / "elec2atlas.mat"

        task = resolve_task(
            TaskFactoryContext(
                task_name="diapix",
                dataset_id=ctx.dataset_id,
                bids_subject=ctx.bids_subject,
                session=ctx.session,
                private_root=ctx.private_root,
                subject_map_yaml=ctx.subject_map_yaml,
                task_assets_dir=ctx.task_assets_dir,
            )
        )

        return task

    return {
        "diapix": make_diapix,
        # "conversation": make_conversation,
        # "rest": make_rest,
    }


def _resolve_dcap_id(
    *,
    dataset_id: str,
    bids_subject: str,
    private_root: Optional[Path],
    subject_map_yaml: Optional[Path],
) -> str:
    """
    Resolve the private clinical identifier (dcap_id) for (dataset_id, bids_subject).

    Priority:
    1) subject_map_yaml (explicit)
    2) private_root / "subject_reid_map.yml"
    """
    if subject_map_yaml is not None:
        yaml_path = Path(subject_map_yaml).expanduser().resolve()
    else:
        if private_root is None:
            raise ValueError(
                "Cannot resolve dcap_id without --private-root (or $DCAP_PRIVATE_ROOT) or --subject-map-yaml."
            )
        yaml_path = Path(private_root).expanduser().resolve() / "subject_reid_map.yml"

    mapping = load_subject_reid_map(yaml_path)
    return mapping.resolve_dcap_id(dataset_id=str(dataset_id).strip(), bids_subject=str(bids_subject).strip())

