# =============================================================================
# =============================================================================
#                     ########################################
#                     #          CLI: dcap preprocess         #
#                     ########################################
# =============================================================================
# =============================================================================
"""
dcap preprocess

Config-driven preprocessing wrapper around:
    dcap.seeg.preprocessing.pipelines.standard

This CLI intentionally stays thin: it resolves input runs, loads Raw with MNE,
and delegates all processing to the pipeline.

Usage example
-------------
    dcap preprocess --config dcap_preprocess.yaml

    dcap preprocess --config dcap_preprocess.yaml --dry-run

    dcap preprocess --config dcap_preprocess.yaml --max-runs 2
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Tuple

import typer

import mne

from dcap.seeg.preprocessing.pipelines.standard import (
    PreprocessOutputs,
    StandardPipelineConfig,
    load_preprocess_yaml,
    run_preprocess_single_raw,
)


app = typer.Typer(
    help="Preprocess sEEG/iEEG runs (reref, resample, line noise, filtering) and save derivatives.",
    no_args_is_help=True,
)


Scope = Literal["run", "patient", "dataset"]


# =============================================================================
#                               Small run descriptor
# =============================================================================
@dataclass(frozen=True)
class RunSpec:
    """
    Minimal description of an input run to be processed.

    Attributes
    ----------
    subject
        Subject id without "sub-".
    session
        Session id without "ses-" (None if absent).
    task
        Task label (None if absent).
    run
        Run label (None if absent).
    in_path
        Path to the input file.
    """

    subject: str
    session: Optional[str]
    task: Optional[str]
    run: Optional[str]
    in_path: Path
    datatype: str = "ieeg"

    def base_stem(self) -> str:
        """
        Build a sensible derivative stem using BIDS-like entities.
        """
        parts: List[str] = [f"sub-{self.subject}"]
        if self.session is not None:
            parts.append(f"ses-{self.session}")
        if self.task is not None:
            parts.append(f"task-{self.task}")
        if self.run is not None:
            parts.append(f"run-{self.run}")
        return "_".join(parts)


# =============================================================================
#                               CLI entrypoint
# =============================================================================
@app.command("preprocess")
def preprocess_cmd(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False, readable=True, help="Path to preprocessing YAML config."),
    dry_run: bool = typer.Option(False, "--dry-run", help="List runs that would be processed, then exit."),
    max_runs: Optional[int] = typer.Option(None, "--max-runs", help="Process at most N runs (debug convenience)."),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging."),
) -> None:
    """
    Run preprocessing according to a YAML config.

    Exit code semantics
    -------------------
    - 0: all requested runs processed successfully (or dry-run)
    - 1: at least one run failed

    Notes
    -----
    This CLI uses a simple BIDS-ish globber to find input files.
    If you already have an official run enumerator, replace `_iter_runs_from_cfg`.
    """
    cfg = load_preprocess_yaml(config)
    runs = list(_iter_runs_from_cfg(cfg))

    if max_runs is not None:
        runs = runs[: int(max_runs)]

    if not runs:
        typer.echo("No input runs found for the provided selection config.")
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(runs)} run(s).")

    if dry_run:
        for rs in runs:
            typer.echo(f"- {rs.in_path}")
        raise typer.Exit(code=0)

    failures: List[Tuple[RunSpec, str]] = []
    for idx, rs in enumerate(runs, start=1):
        typer.echo(f"[{idx}/{len(runs)}] {rs.in_path}")

        try:
            raw = _read_raw(rs.in_path, preload=bool(cfg.raw.get("io", {}).get("preload", True)), verbose=verbose)
            out_dir = _resolve_out_dir(cfg=cfg, run_spec=rs)
            outputs = run_preprocess_single_raw(
                raw=raw,
                cfg=cfg,
                out_dir=out_dir,
                base_stem=rs.base_stem(),
            )
            _print_outputs(outputs, verbose=verbose)

        except Exception as exc:
            failures.append((rs, repr(exc)))
            typer.echo(f"  ERROR: {exc}")

            # If you wire your errors/ subpackage, this is where you'd call it.

            fail_fast = bool(cfg.raw.get("io", {}).get("fail_fast", False))
            if fail_fast:
                raise typer.Exit(code=1)

    if failures:
        typer.echo("\nSome runs failed:")
        for rs, msg in failures:
            typer.echo(f"- {rs.in_path}: {msg}")
        raise typer.Exit(code=1)

    typer.echo("\nAll runs processed successfully.")
    raise typer.Exit(code=0)


# =============================================================================
#                               Run enumeration
# =============================================================================
def _iter_runs_from_cfg(cfg: StandardPipelineConfig) -> Iterator[RunSpec]:
    """
    Enumerate runs based on selection settings in YAML.

    Current implementation
    ----------------------
    Uses simple filesystem globbing for iEEG sources:

    - Looks under <bids_root>/sub-*/(ses-*/)?{ieeg,eeg}/
    - Accepts: *.edf, *.bdf, *.vhdr, *.set, *.fif
    - Extracts task/run from filename tokens if present

    Replaceable
    -----------
    If you already have a robust BIDS enumerator in dcap, swap this out.
    """
    selection = cfg.raw.get("selection", {})
    bids_root = Path(selection.get("bids_root", ".")).expanduser().resolve()
    scope: Scope = str(selection.get("scope", "run"))  # type: ignore[assignment]

    subject = selection.get("subject", None)
    session = selection.get("session", None)
    task = selection.get("task", None)
    run = selection.get("run", None)

    include = selection.get("include", {}) if isinstance(selection.get("include", {}), dict) else {}
    exclude = selection.get("exclude", {}) if isinstance(selection.get("exclude", {}), dict) else {}

    subjects = _normalize_optional_str_list(include.get("subjects", None))
    if subjects is None and isinstance(subject, str) and scope in ("run", "patient"):
        subjects = [subject]

    sessions = _normalize_optional_str_list(include.get("sessions", None))
    if sessions is None and isinstance(session, str):
        sessions = [session]

    tasks = _normalize_optional_str_list(include.get("tasks", None))
    if tasks is None and isinstance(task, str) and scope in ("run", "patient"):
        tasks = [task]

    runs = _normalize_optional_str_list(include.get("runs", None))
    if runs is None and isinstance(run, str) and scope == "run":
        runs = [run]

    # Glob candidates
    sub_glob = "sub-*" if subjects is None else "{" + ",".join([f"sub-{s}" for s in subjects]) + "}"
    sub_dirs = sorted(bids_root.glob(sub_glob))
    for sub_dir in sub_dirs:
        if not sub_dir.is_dir():
            continue
        subj = sub_dir.name.replace("sub-", "")

        ses_dirs = [sub_dir]
        if any(sub_dir.glob("ses-*")):
            if sessions is None:
                ses_dirs = sorted(sub_dir.glob("ses-*"))
            else:
                ses_dirs = [sub_dir / f"ses-{s}" for s in sessions if (sub_dir / f"ses-{s}").exists()]

        for ses_dir in ses_dirs:
            ses = None if ses_dir == sub_dir else ses_dir.name.replace("ses-", "")

            for datatype in ("ieeg", "eeg"):
                data_dir = ses_dir / datatype
                if not data_dir.exists():
                    continue

                for in_path in _iter_recording_paths(data_dir):
                    parsed_task, parsed_run = _parse_task_run_from_name(in_path.name)

                    if tasks is not None and parsed_task is not None and parsed_task not in tasks:
                        continue
                    if runs is not None and parsed_run is not None and parsed_run not in runs:
                        continue

                    if _is_excluded(subj, ses, parsed_task, parsed_run, exclude=exclude):
                        continue

                    yield RunSpec(
                        subject=subj,
                        session=ses,
                        task=parsed_task,
                        run=parsed_run,
                        in_path=in_path,
                        datatype=datatype,
                    )


def _normalize_optional_str_list(value: object) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _iter_recording_paths(data_dir: Path) -> Iterator[Path]:
    patterns = ("*.edf", "*.bdf", "*.vhdr", "*.set", "*.fif")
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(data_dir.glob(pattern))

    for in_path in sorted(candidates):
        if in_path.is_file():
            yield in_path


def _parse_task_run_from_name(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse BIDS-like tokens: *_task-XXX_* and *_run-YYY_* from filename.
    """
    task = None
    run = None
    for token in filename.split("_"):
        if token.startswith("task-"):
            task = token.replace("task-", "")
        if token.startswith("run-"):
            run = token.replace("run-", "").split(".")[0]
    return task, run


def _is_excluded(
    subject: str,
    session: Optional[str],
    task: Optional[str],
    run: Optional[str],
    *,
    exclude: dict,
) -> bool:
    def _match(opt_list: Optional[List[str]], value: Optional[str]) -> bool:
        if opt_list is None:
            return False
        if value is None:
            return False
        return value in opt_list

    ex_sub = _normalize_optional_str_list(exclude.get("subjects", None))
    ex_ses = _normalize_optional_str_list(exclude.get("sessions", None))
    ex_task = _normalize_optional_str_list(exclude.get("tasks", None))
    ex_run = _normalize_optional_str_list(exclude.get("runs", None))

    if _match(ex_sub, subject):
        return True
    if _match(ex_ses, session):
        return True
    if _match(ex_task, task):
        return True
    if _match(ex_run, run):
        return True
    return False


# =============================================================================
#                               IO helpers
# =============================================================================
def _read_raw(path: Path, *, preload: bool, verbose: bool) -> mne.io.BaseRaw:
    """
    Read common BIDS EEG/iEEG formats with MNE.
    """
    suffix = path.suffix.lower()
    if suffix == ".edf":
        return mne.io.read_raw_edf(path, preload=preload, verbose="info" if verbose else False)
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(path, preload=preload, verbose="info" if verbose else False)
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(path, preload=preload, verbose="info" if verbose else False)
    if suffix == ".set":
        return mne.io.read_raw_eeglab(path, preload=preload, verbose="info" if verbose else False)
    if suffix == ".fif":
        return mne.io.read_raw_fif(path, preload=preload, verbose="info" if verbose else False)
    raise ValueError(f"Unsupported input file type: {path.name}")


def _resolve_out_dir(*, cfg: StandardPipelineConfig, run_spec: RunSpec) -> Path:
    selection = cfg.raw.get("selection", {})
    bids_root = Path(selection.get("bids_root", ".")).expanduser().resolve()

    io_cfg = cfg.raw.get("io", {})
    derivatives_root = selection.get("derivatives_root", None)
    derivatives_name = str(io_cfg.get("derivatives_name", "preproc"))

    root = Path(derivatives_root).expanduser().resolve() if derivatives_root else (bids_root / "derivatives")
    out_dir = root / derivatives_name / f"sub-{run_spec.subject}"
    if run_spec.session is not None:
        out_dir = out_dir / f"ses-{run_spec.session}"
    out_dir = out_dir / run_spec.datatype
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _print_outputs(outputs: PreprocessOutputs, *, verbose: bool) -> None:
    typer.echo(f"  Saved {len(outputs.saved_paths)} file(s):")
    for p in outputs.saved_paths:
        typer.echo(f"   - {p.name}")
    typer.echo(f"  Provenance: {outputs.provenance_path.name}")
    if verbose:
        typer.echo(f"  Artifacts: {len(outputs.artifacts)}")
