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
from typing import Dict, Mapping, Optional, Sequence, Tuple

import mne
from mne_bids import BIDSPath
import pandas as pd


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
) -> dict[str, str]:
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
    ieeg_dir = (subj_dir / "ieeg") if session is None else (subj_dir / session / "ieeg")
    if not ieeg_dir.exists():
        return {}

    # Priority order: pick the first one that exists for each run
    preferred_exts = [".vhdr", ".edf", ".bdf", ".set", ".fif"]

    # Gather all candidate files that look like recordings (not JSON/TSV)
    pattern = f"{subject}_task-{task}_run-*_ieeg*"
    candidates = [p for p in ieeg_dir.glob(pattern) if p.is_file()]

    run_to_exts: dict[str, set[str]] = {}
    for fp in candidates:
        m = re.search(r"_run-(?P<run>[^_]+)_ieeg", fp.name)
        if m is None:
            continue
        run = m.group("run")
        ext = fp.suffix.lower()

        # Skip obvious sidecars
        if ext in {".json", ".tsv", ".gz"}:
            continue

        run_to_exts.setdefault(run, set()).add(ext)

    # Choose preferred extension per run
    run_to_pref: dict[str, str] = {}
    for run, exts in run_to_exts.items():
        chosen = None
        for ext in preferred_exts:
            if ext in exts:
                chosen = ext
                break

        # If we only see BrainVision data (.eeg) but no header (.vhdr), that's suspicious.
        if chosen is None:
            # fall back to any extension we saw, but keep it stable
            chosen = sorted(exts)[0]

        run_to_pref[run] = chosen

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

    run_to_ext = _discover_runs_with_extensions(
        bids_root=Path(bids_root),
        subject=subject_id_norm,
        session=session_id_norm,
        task=task_norm,
    )

    if run_id_norm is None:
        if len(run_to_ext) == 1:
            run_id_norm = next(iter(run_to_ext.keys()))
        elif len(run_to_ext) > 1:
            pretty = [f"run-{r}:{ext}" for r, ext in run_to_ext.items()]
            raise ValueError(
                "Multiple runs exist for this subject/task. Please specify --run. "
                f"Available: {pretty}"
            )

    # If we know the preferred extension for the selected run, pass it to BIDSPath
    extension = run_to_ext.get(run_id_norm, None)

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
        datatype="ieeg",
        extension=extension,
    )

    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()

    events_df = _load_events_df_from_bids(bids_path=bids_path, fallback_raw=raw)
    electrodes_table = _load_electrodes_table_from_bids(bids_path=bids_path)

    return raw, events_df, electrodes_table


def _load_events_df_from_bids(*, bids_path: "BIDSPath", fallback_raw: "mne.io.BaseRaw") -> pd.DataFrame:
    """Load events as a DataFrame, preferring *_events.tsv; else fall back to annotations."""
    import pandas as pd

    recording_path = Path(str(bids_path.fpath))
    events_path = recording_path.with_name(recording_path.name.replace(recording_path.suffix, "_events.tsv"))
    if events_path.exists():
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

    recording_path = Path(str(bids_path.fpath))
    electrodes_path = recording_path.with_name(
        recording_path.stem.replace("_ieeg", "") + "_channels.tsv"
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
