# =============================================================================
# =============================================================================
#                           #################################
#                           #          BIDS I/O (sEEG)       #
#                           #################################
# =============================================================================
# =============================================================================
#
# I/O layer only:
# - locate BIDS files
# - load MNE Raw
# - load events table into a DataFrame
# - (optionally) load electrodes table into a mapping for coordinate attachment
#
# No preprocessing or scientific logic belongs here.
#
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Sequence, Tuple

import mne
import pandas as pd

from dcap.seeg.io.sidecars import find_neighbor_sidecar

if TYPE_CHECKING:  # pragma: no cover
    from mne_bids import BIDSPath


@dataclass(frozen=True)
class BidsRunSpec:
    """
    Identifies a single BIDS run.

    Attributes
    ----------
    bids_root
        BIDS dataset root.
    subject_id
        BIDS subject identifier (e.g., "sub-001").
    session_id
        Optional BIDS session identifier (e.g., "ses-01").
    task
        BIDS task label (without "task-").
    run_id
        Optional run identifier (e.g., "1" corresponds to "run-1").

    Usage example
    -------------
        spec = BidsRunSpec(
            bids_root=Path("/data/bids"),
            subject_id="sub-001",
            session_id="ses-01",
            task="conversation",
            run_id="1",
        )
    """

    bids_root: Path
    subject_id: str
    session_id: Optional[str]
    task: str
    run_id: Optional[str]


def _strip_prefix(value: Optional[str], prefix: str) -> Optional[str]:
    if value is None:
        return None
    return value[len(prefix):] if value.startswith(prefix) else value


def _normalize_bids_ids(
    subject_id: str,
    session_id: Optional[str],
    task: str,
    run_id: Optional[str],
) -> Tuple[str, Optional[str], str, Optional[str]]:
    subject_id_norm = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    session_id_norm = None
    if session_id is not None:
        session_id_norm = session_id if session_id.startswith("ses-") else f"ses-{session_id}"

    task_norm = _strip_prefix(task, "task-") or task

    run_id_norm = None
    if run_id is not None:
        run_str = _strip_prefix(run_id, "run-") or run_id
        run_id_norm = run_str

    return subject_id_norm, session_id_norm, task_norm, run_id_norm


def _discover_runs_with_extensions(
    *,
    bids_root: Path,
    subject: str,
    session: Optional[str],
    task: str,
) -> dict[str, tuple[str, str]]:
    """
    Discover available runs and pick a preferred loadable extension per run.

    We prefer "entrypoint" files that MNE can read directly:
    - BrainVision: .vhdr (preferred), not .eeg/.vmrk
    - EDF: .edf
    - BDF: .bdf
    - EEGLAB: .set
    - FIF: .fif
    """
    subj_dir = bids_root / subject
    # Priority order: pick the first one that exists for each run
    preferred_exts = [".vhdr", ".edf", ".bdf", ".set", ".fif"]
    preferred_datatypes = ["ieeg", "eeg"]

    run_to_choices: dict[str, dict[str, set[str]]] = {}
    for datatype in preferred_datatypes:
        data_dir = (subj_dir / datatype) if session is None else (subj_dir / session / datatype)
        if not data_dir.exists():
            continue

        pattern = f"{subject}*task-{task}_run-*_{datatype}*"
        candidates = [p for p in data_dir.glob(pattern) if p.is_file()]
        for fp in candidates:
            m = re.search(rf"_run-(?P<run>[^_]+)_{datatype}", fp.name)
            if m is None:
                continue
            run = m.group("run")
            ext = fp.suffix.lower()

            if ext in {".json", ".tsv", ".gz"}:
                continue

            run_to_choices.setdefault(run, {}).setdefault(datatype, set()).add(ext)

    # Choose preferred extension per run
    run_to_pref: dict[str, tuple[str, str]] = {}
    for run, datatype_to_exts in run_to_choices.items():
        chosen_datatype = None
        chosen_ext = None

        for datatype in preferred_datatypes:
            exts = datatype_to_exts.get(datatype)
            if not exts:
                continue
            chosen_datatype = datatype
            for ext in preferred_exts:
                if ext in exts:
                    chosen_ext = ext
                    break
            if chosen_ext is None:
                chosen_ext = sorted(exts)[0]
            break

        if chosen_datatype is None or chosen_ext is None:
            continue

        run_to_pref[run] = (chosen_datatype, chosen_ext)

    return dict(sorted(run_to_pref.items(), key=lambda kv: kv[0]))


def load_bids_run(
    *,
    bids_root: Path,
    subject_id: str,
    session_id: Optional[str],
    task: str,
    run_id: Optional[str],
) -> Tuple["mne.io.BaseRaw", pd.DataFrame, Optional[Mapping[str, Sequence[float]]]]:
    """
    Load a BIDS run as (Raw, events_df, electrodes_table).

    Returns
    -------
    raw
        Loaded MNE Raw (preload=True).
    events_df
        Events DataFrame with (at minimum) columns:
        onset_sec, duration_sec, event_type
    electrodes_table
        Optional mapping: channel_name -> (x, y, z). Units depend on the source file.

    Notes
    -----
    This function requires `mne-bids`.

    Usage example
    -------------
        raw, events_df, electrodes_table = load_bids_run(
            bids_root=Path("/data/bids"),
            subject_id="sub-001",
            session_id="ses-01",
            task="conversation",
            run_id="1",
        )
    """
    subject_id_norm, session_id_norm, task_norm, run_id_norm = _normalize_bids_ids(
        subject_id=subject_id,
        session_id=session_id,
        task=task,
        run_id=run_id,
    )

    run_to_choice = _discover_runs_with_extensions(
        bids_root=Path(bids_root),
        subject=subject_id_norm,
        session=session_id_norm,
        task=task_norm,
    )

    if run_id_norm is None:
        if len(run_to_choice) == 1:
            run_id_norm = next(iter(run_to_choice.keys()))
        elif len(run_to_choice) > 1:
            pretty = [f"run-{r}:{datatype}{ext}" for r, (datatype, ext) in run_to_choice.items()]
            raise ValueError(
                "Multiple runs exist for this subject/task. Please specify --run. "
                f"Available: {pretty}"
            )

    datatype = "ieeg"
    extension = None
    if run_id_norm is not None and run_id_norm in run_to_choice:
        datatype, extension = run_to_choice[run_id_norm]

    try:
        import mne  # noqa: F401
        from mne_bids import BIDSPath, read_raw_bids
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "BIDS loading requires `mne-bids`. Install it (e.g., `pip install mne-bids`)."
        ) from exc

    bids_path = BIDSPath(
        root=Path(bids_root),
        subject=_strip_prefix(subject_id_norm, "sub-"),
        session=_strip_prefix(session_id_norm, "ses-") if session_id_norm else None,
        task=task_norm,
        run=run_id_norm,
        datatype=datatype,
        extension=extension,
    )

    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()

    events_df = _load_events_df_from_bids(bids_path=bids_path, fallback_raw=raw)
    electrodes_table = _load_electrodes_table_from_bids(bids_path=bids_path)

    return raw, events_df, electrodes_table


def _load_events_df_from_bids(*, bids_path: "BIDSPath", fallback_raw: "mne.io.BaseRaw") -> pd.DataFrame:
    """Load events as a DataFrame, preferring neighboring *_events.tsv; else fall back to annotations."""
    import pandas as pd

    recording_path = Path(str(bids_path.fpath))
    events_path = find_neighbor_sidecar(recording_path, sidecar_suffix="_events.tsv")
    if events_path is not None:
        df = pd.read_csv(events_path, sep="\t")
        if "onset" in df.columns and "onset_sec" not in df.columns:
            df["onset_sec"] = df["onset"].astype(float)
        if "duration" in df.columns and "duration_sec" not in df.columns:
            df["duration_sec"] = df["duration"].astype(float)
        if "trial_type" in df.columns and "event_type" not in df.columns:
            df["event_type"] = df["trial_type"].astype(str)

        required = {"onset_sec", "duration_sec", "event_type"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Events TSV is missing required columns: {missing}")

        # Keep the shared table task-agnostic here; task-specific enrichment should
        # happen upstream once a given task defines stronger event semantics.
        return df.loc[:, ["onset_sec", "duration_sec", "event_type"]].copy()

    if fallback_raw.annotations is None or len(fallback_raw.annotations) == 0:
        return pd.DataFrame(columns=["onset_sec", "duration_sec", "event_type"])

    onset_sec = fallback_raw.annotations.onset.astype(float)
    duration_sec = fallback_raw.annotations.duration.astype(float)
    event_type = fallback_raw.annotations.description.astype(str)
    return pd.DataFrame(
        {"onset_sec": onset_sec, "duration_sec": duration_sec, "event_type": event_type}
    )


def _load_electrodes_table_from_bids(*, bids_path: "BIDSPath") -> Optional[Mapping[str, Sequence[float]]]:
    """Load electrodes.tsv as a mapping channel_name -> (x,y,z) if present."""
    import pandas as pd

    electrodes_path = resolve_canonical_electrodes_tsv(
        bids_path=bids_path,
        bids_root=bids_path.root,
        derivatives_name="elec_recon",
        coords_space="MNI152"
    )
    if not electrodes_path.exists():
        return None

    df = pd.read_csv(electrodes_path, sep="\t")
    for col in ("name", "x", "y", "z"):
        if col not in df.columns:
            return None

    table: Dict[str, Tuple[float, float, float]] = {}
    for _, row in df.iterrows():
        name = str(row["name"])
        try:
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
        except Exception:  # noqa
            continue
        table[name] = (x, y, z)

    return table


def resolve_canonical_electrodes_tsv(
    *,
    bids_path: "BIDSPath",
    bids_root: Path,
    derivatives_name: str = "elec_recon",
    coords_space: str = "MNI152",
) -> Optional[Path]:
    """
    Resolve the canonical subject-level electrodes.tsv for a recording.

    Parameters
    ----------
    bids_path
        BIDSPath of the recording (run-level).
    bids_root
        Root of the BIDS dataset.
    derivatives_name
        Name of derivatives folder containing electrodes TSV.
    coords_space
        Coordinate space label used in the TSV filename.

    Returns
    -------
    Optional[Path]
        Path to electrodes TSV if it exists, else None.
    """
    if bids_path.subject is None:
        return None

    sub = bids_path.subject
    ses = bids_path.session

    base = (
        bids_root
        / "derivatives"
        / derivatives_name
        / f"sub-{sub}"
    )

    if ses is not None:
        base = base / f"ses-{ses}"

    fname_parts = [f"sub-{sub}"]
    if ses is not None:
        fname_parts.append(f"ses-{ses}")
    fname_parts.append(f"space-{coords_space}")
    fname_parts.append("electrodes.tsv")

    electrodes_path = base / "_".join(fname_parts)

    if not electrodes_path.exists():
        return None

    return electrodes_path
