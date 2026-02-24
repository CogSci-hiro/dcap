# =============================================================================
#                      BIDS Tasks: Registry / factories
# =============================================================================
#
# Central place to map task names -> task factories.
# Keeps CLI generic and prevents core from importing tasks.
#
# IMPORTANT:
# - Factories MUST construct and return a task directly.
# - Factories MUST NOT call resolve_task() (would recurse forever).
#
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

from dcap.bids.tasks.base import BidsTask
from dcap.bids.tasks.subject_map import load_subject_reid_map


# =============================================================================
# Context + factory types
# =============================================================================

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
            subject_map_yaml=Path("/private/subject_keys.yaml"),
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


# =============================================================================
# Public API
# =============================================================================

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


def list_tasks() -> list[str]:
    """
    List available task names.

    Usage example
    -------------
        print(list_tasks())
    """
    return sorted(_get_task_factories().keys())


# =============================================================================
# Internal: factory mapping
# =============================================================================

def _get_task_factories() -> Mapping[str, TaskFactory]:
    """
    Map task name -> factory.

    Note: imports are intentionally local to avoid import side effects.
    """
    from dcap.bids.tasks.diapix.task import DiapixTask
    from dcap.bids.tasks.naming.task import NamingTask
    from dcap.bids.tasks.sorciere.task import SorciereTask

    def make_diapix(ctx: TaskFactoryContext) -> BidsTask:
        dcap_id = _resolve_dcap_id(
            dataset_id=ctx.dataset_id,
            bids_subject=ctx.bids_subject,
            private_root=ctx.private_root,
            subject_map_yaml=ctx.subject_map_yaml,
        )

        if ctx.task_assets_dir is None:
            raise ValueError(
                "diapix requires --task-assets-dir "
                "(flat directory containing audio_onsets.tsv and the shared stimulus wav)."
            )

        task_assets_dir = Path(ctx.task_assets_dir).expanduser().resolve()

        audio_onsets_tsv = task_assets_dir / "audio_onsets.tsv"
        stim_wav = task_assets_dir / "beeps_pre-task-1-sec_post-task-4-sec.wav"
        # Optional: fail fast with helpful errors
        _require_file(audio_onsets_tsv, "audio_onsets.tsv")
        _require_file(stim_wav, "stim wav")

        return DiapixTask(
            bids_subject=ctx.bids_subject,
            dcap_id=dcap_id,
            session=ctx.session,
            audio_onsets_tsv=audio_onsets_tsv,
            stim_wav=stim_wav,
            atlas_path=None,
        )

    def make_sorciere(ctx: TaskFactoryContext) -> BidsTask:
        dcap_id = _resolve_dcap_id(
            dataset_id=ctx.dataset_id,
            bids_subject=ctx.bids_subject,
            private_root=ctx.private_root,
            subject_map_yaml=ctx.subject_map_yaml,
        )

        if ctx.task_assets_dir is None:
            raise ValueError(
                "sorciere requires --task-assets-dir "
                "(directory containing the shared Sorciere stimulus WAV)."
            )

        task_assets_dir = Path(ctx.task_assets_dir).expanduser().resolve()
        stim_wav = _resolve_single_stim_wav(task_assets_dir, preferred_tokens=("sorciere", "ispeech", "passive"))

        return SorciereTask(
            bids_subject=ctx.bids_subject,
            dcap_id=dcap_id,
            session=ctx.session,
            stim_wav=stim_wav,
            trigger_id=10004,
        )

    def make_naming(ctx: TaskFactoryContext) -> BidsTask:
        # Naming currently derives events from raw annotations only and does not
        # require private subject mapping or task assets.
        return NamingTask(
            bids_subject=ctx.bids_subject,
            dcap_id=None,
            session=ctx.session,
        )

    return {
        "diapix": make_diapix,
        "naming": make_naming,
        "sorciere": make_sorciere,
        # "conversation": make_conversation,
        # "rest": make_rest,
    }


# =============================================================================
# Internal: helpers
# =============================================================================

def _resolve_dcap_id(
    *,
    dataset_id: str,
    bids_subject: str,
    private_root: Optional[Path],
    subject_map_yaml: Optional[Path],
) -> str:
    """
    Resolve the private clinical identifier (dcap_id) for (dataset_id, bids_subject).

    Priority
    --------
    1) subject_map_yaml (explicit)
    2) private_root / "subject_keys.yaml"
    """
    if subject_map_yaml is not None:
        yaml_path = Path(subject_map_yaml).expanduser().resolve()
    else:
        if private_root is None:
            raise ValueError("Missing private_root and subject_map_yaml; cannot resolve dcap_id.")
        yaml_path = Path(private_root).expanduser().resolve() / "subject_keys.yaml"

    mapping = load_subject_reid_map(yaml_path)
    return mapping.resolve_dcap_id(
        dataset_id=str(dataset_id).strip(),
        bids_subject=str(bids_subject).strip(),
    )


def _require_file(path: Path, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Required {label} not found: {Path(path)}")


def _resolve_single_stim_wav(task_assets_dir: Path, preferred_tokens: tuple[str, ...]) -> Path:
    wavs = sorted(p for p in task_assets_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    if len(wavs) == 0:
        raise FileNotFoundError(f"No WAV stimulus file found in task assets dir: {task_assets_dir}")
    if len(wavs) == 1:
        return wavs[0]

    scored: list[tuple[int, str, Path]] = []
    for wav in wavs:
        stem = wav.stem.lower()
        rank = len(preferred_tokens) + 1
        for idx, token in enumerate(preferred_tokens):
            if token in stem:
                rank = idx
                break
        scored.append((rank, wav.name, wav))

    scored.sort(key=lambda x: (x[0], x[1]))
    if len(scored) >= 2 and scored[0][0] == scored[1][0]:
        names = ", ".join(w.name for w in wavs)
        raise ValueError(
            f"Ambiguous stimulus WAV selection in {task_assets_dir}; "
            f"multiple candidates share the same priority: {names}"
        )
    return scored[0][2]
