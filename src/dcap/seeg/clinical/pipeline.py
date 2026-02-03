# =============================================================================
# =============================================================================
#                       #####################################
#                       #   CLINICAL ANALYSIS ORCHESTRATOR  #
#                       #####################################
# =============================================================================
# =============================================================================
#
# Composition layer (clinical):
# ----------------------------
# This module is the *orchestrator* for a single clinical analysis run.
# It composes multiple library subsystems into one coherent pipeline and
# returns a "report-ready" bundle object.
#
# High-level data flow
# --------------------
# BIDS IO (outside this module) -> preprocessing -> analysis view selection
# -> optional gamma envelope -> optional TRF -> optional QC figures -> bundle
#
# Boundaries / responsibilities
# -----------------------------
# This layer is "logic only":
# - No file I/O (except optional figure writing via make_qc_figures; see below)
# - No CLI / printing
# - No interactive plotting
#
# It *does* coordinate:
# - which preprocessing view is used for analysis
# - whether envelope/TRF should be computed
# - provenance decisions to store in the preprocessing context
#
# Outputs
# -------
# The returned `ClinicalAnalysisBundle` is intended to be consumed by:
# - report renderers (markdown/PDF)
# - QC summaries
# - downstream export steps (handled elsewhere)
#
# Notes on determinism
# --------------------
# The orchestrator should be deterministic given:
# - the `raw` input
# - configs
# - electrode metadata
# Any randomness or model selection should be explicit and tracked in `ctx`.
#
# =============================================================================

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import mne
import numpy as np
import pandas as pd

from dcap.seeg.clinical.bundle import (ClinicalAnalysisBundle, ClinicalRunBundle, ClinicalSessionBundle,
                                       ClinicalAnalysisNotes)
from dcap.seeg.clinical.configs import ClinicalAnalysisConfig
from dcap.seeg.clinical.policy import choose_analysis_view
from dcap.seeg.preprocessing.blocks.filtering import compute_gamma_envelope
from dcap.seeg.preprocessing.configs import (
    ClinicalPreprocConfig,
    CoordinatesConfig,
    GammaEnvelopeConfig,
)
from dcap.seeg.preprocessing.pipelines.clinical import run_clinical_preproc
from dcap.seeg.preprocessing.types import PreprocContext
from dcap.seeg.trf.contracts import TRFConfig, TRFInput, TRFResult
from dcap.seeg.clinical.viz.qc_figures import make_qc_figures
from dcap.seeg.clinical.qc import compute_clinical_qc
from dcap.seeg.clinical.trf.runner_analysis_trf import run_trf_with_analysis_trf


# =============================================================================
# Electrode normalization
# =============================================================================

def _normalize_electrodes_df(
    *,
    electrodes_table: Optional[Any],
    coords_cfg: Optional[CoordinatesConfig],
) -> tuple[Optional[pd.DataFrame], Optional[str], list[str]]:
    """
    Convert and normalize electrode metadata into a canonical DataFrame.

    Parameters
    ----------
    electrodes_table
        Upstream electrode table. May be a DataFrame, dict-like, or None.
    coords_cfg
        Optional configuration describing coordinate semantics.

    Returns
    -------
    electrodes_df
        Canonical electrode table or None.
    coords_space
        Single coordinate space label if known/consistent (e.g. "MNI"), else None.
    warnings
        Human-readable warnings describing any normalization issues.

    Canonical schema (minimum)
    --------------------------
    | name | x | y | z | space |
    """
    warnings: list[str] = []

    if electrodes_table is None:
        return None, None, warnings

    # -------------------------------------------------------------------------
    # 1) Convert to DataFrame (best effort)
    # -------------------------------------------------------------------------
    if isinstance(electrodes_table, pd.DataFrame):
        df = electrodes_table.copy()
    elif isinstance(electrodes_table, dict):
        # Common case: {"LA1": [x, y, z], "LA2": [x, y, z], ...}
        try:
            df = pd.DataFrame(
                [{"name": str(k), "coords": v} for k, v in electrodes_table.items()]
            )
        except Exception as e:
            warnings.append(f"Could not convert electrodes_table dict to DataFrame: {e}")
            return None, None, warnings
    else:
        warnings.append(f"Unsupported electrodes_table type: {type(electrodes_table)!r}")
        return None, None, warnings

    if df.empty:
        return None, None, warnings

    # -------------------------------------------------------------------------
    # 2) Standardize electrode name column -> 'name'
    # -------------------------------------------------------------------------
    if "name" not in df.columns:
        for candidate in ("electrode", "electrode_name", "label", "contact", "channel"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "name"})
                break

    if "name" not in df.columns:
        warnings.append("Electrode table missing a name column (expected 'name').")
        return None, None, warnings

    df["name"] = df["name"].astype(str)

    # -------------------------------------------------------------------------
    # 3) Standardize coordinates -> 'x','y','z'
    # -------------------------------------------------------------------------
    # Accepted inputs:
    # - already has x/y/z
    # - has 'coords' (sequence length >=3)
    # - has 'x','y','z' but in different casing
    lower_cols = {c.lower(): c for c in df.columns}
    for axis in ("x", "y", "z"):
        if axis not in df.columns and axis in lower_cols:
            df = df.rename(columns={lower_cols[axis]: axis})

    if all(axis in df.columns for axis in ("x", "y", "z")):
        # Ensure numeric
        for axis in ("x", "y", "z"):
            df[axis] = pd.to_numeric(df[axis], errors="coerce")
    elif "coords" in df.columns:
        def _get_coord(v: Any, idx: int) -> float:
            try:
                return float(v[idx])
            except Exception:  # noqa
                return float("nan")

        df["x"] = df["coords"].map(lambda v: _get_coord(v, 0))
        df["y"] = df["coords"].map(lambda v: _get_coord(v, 1))
        df["z"] = df["coords"].map(lambda v: _get_coord(v, 2))
    else:
        warnings.append("Electrode table missing coordinates (need x/y/z or 'coords').")

    # Drop rows without usable names
    df = df.loc[df["name"].notna() & (df["name"].astype(str) != "")].copy()

    # -------------------------------------------------------------------------
    # 4) Coordinate space handling
    # -------------------------------------------------------------------------
    coords_space: Optional[str] = None

    # If a 'space' column exists, use it
    if "space" in df.columns:
        spaces = df["space"].dropna().astype(str).unique().tolist()
        if len(spaces) == 1:
            coords_space = spaces[0]
        elif len(spaces) > 1:
            warnings.append(f"Multiple coordinate spaces present in electrode table: {spaces}")
    else:
        # If coords_cfg has a space field, use it (depends on your CoordinatesConfig)
        # Update this block once you confirm field names.
        if coords_cfg is not None and hasattr(coords_cfg, "space"):
            value = getattr(coords_cfg, "space")
            coords_space = None if value is None else str(value)

        if coords_space is not None:
            df["space"] = coords_space

    # -------------------------------------------------------------------------
    # 5) Final canonical columns first (keep extras)
    # -------------------------------------------------------------------------
    canonical_first = [c for c in ["name", "x", "y", "z", "space"] if c in df.columns]
    other_cols = [c for c in df.columns if c not in canonical_first]
    df = df[canonical_first + other_cols]

    # Warn if coordinates are mostly missing (affects 3D plots)
    if not all(axis in df.columns for axis in ("x", "y", "z")):
        warnings.append("Electrode coordinates unavailable; 3D localization plots will be placeholders.")
    else:
        frac_nan = float(df[["x", "y", "z"]].isna().any(axis=1).mean())
        if frac_nan > 0.2:
            warnings.append(f"Many electrodes missing x/y/z (fraction with any NaN: {frac_nan:.2f}).")

    return df, coords_space, warnings


# =============================================================================
#                        Multi-run clinical analysis orchestrator
# =============================================================================

def run_clinical_analysis(
    *,
    raws: Sequence[mne.io.BaseRaw],
    bids_root: Optional[str],
    subject_id: str,
    session_id: Optional[str],
    run_ids: Sequence[str],
    preproc_cfg: ClinicalPreprocConfig,
    analysis_cfg: Optional[ClinicalAnalysisConfig] = None,
    electrodes_table: Optional[Mapping[str, Sequence[float]]] = None,
    coords_cfg: Optional[CoordinatesConfig] = None,
    envelope_cfg: Optional[GammaEnvelopeConfig] = None,
    trf_cfg: Optional[TRFConfig] = None,
    events_df: Optional[Any] = None,
    notes: Optional[Mapping[str, str]] = None,
    ctx: Optional[PreprocContext] = None,
    trf_runner: Optional[Any] = None,
    out_dir: Path | None = None,
) -> ClinicalAnalysisBundle:
    """
    Run a clinical analysis composition pipeline over multiple runs and return a report-ready bundle.

    This orchestrator is multi-run by default:
    - preprocessing is performed per run
    - (optional) gamma envelope is computed per run
    - (optional) QC figures are saved per run
    - (optional) TRF is fit ONCE using all runs as epochs (epoch == run), enabling CV across runs

    Parameters
    ----------
    raws
        Sequence of loaded MNE Raw objects, one per run. Each element must be a valid MNE Raw.
    bids_root
        Optional BIDS root path, carried through for provenance or downstream logic.
        This orchestrator does not read from BIDS; it only passes identifiers through.
    subject_id, session_id, run_ids
        Identifiers used for reporting and filenames downstream.
        `run_ids` must have the same length and order as `raws`.
    preproc_cfg
        Clinical preprocessing configuration controlling filtering, referencing,
        channel selection, etc.
    analysis_cfg
        Optional clinical analysis configuration controlling policy decisions,
        e.g. which view is preferred for analysis ("clean", "reref", etc.).
        If None, defaults to `ClinicalAnalysisConfig()`.
    electrodes_table, coords_cfg
        Optional coordinate attachment inputs:
        - `electrodes_table` maps channel name -> coordinates (or similar numeric vector)
        - `coords_cfg` controls coordinate interpretation/attachment policy
    envelope_cfg
        Optional gamma envelope configuration. If provided, an envelope Raw
        ("gamma") is computed from the chosen analysis view for EACH run.
    trf_cfg
        Optional TRF configuration. If provided, TRF computation is attempted ONCE
        over all runs (epochs). This requires:
        - `events_df` (alignment/provenance/events table; must be run-aware upstream)
        - `envelope_cfg` (since TRF here is defined on the gamma envelope)
    events_df
        Optional events table needed for TRF. Required when `trf_cfg` is provided.
        Type is kept as `Any` because this pipeline treats it as an opaque table-like
        object (usually a pandas DataFrame upstream).
    notes
        Optional key-value notes for reporting (freeform). These are carried into
        the returned bundle.
    ctx
        Optional existing `PreprocContext` to continue a preprocessing history.
        If None, preprocessing creates a new context per run.
    trf_runner
        Optional callable implementing the TRF runner contract:
            (TRFInput, TRFConfig, bids_root, subject_id) -> TRFResult
        If None and `trf_cfg` is provided, we fall back to `run_trf_with_analysis_trf`.
    out_dir
        Optional output directory for QC figures. If provided, figures are written
        PER run. Keep None for logic-only orchestration.

    Returns
    -------
    bundle
        ClinicalAnalysisBundle (multi-run shape) consumed by clinical reporting and export.

    Usage example
    -------------
        bundle = run_clinical_analysis(
            raws=[raw_run1, raw_run2, raw_run3, raw_run4],
            bids_root=str(bids_root),
            subject_id="sub-001",
            session_id="ses-01",
            run_ids=["01", "02", "03", "04"],
            preproc_cfg=ClinicalPreprocConfig(),
            envelope_cfg=GammaEnvelopeConfig(),
            trf_cfg=TRFConfig(...),
            events_df=events_df,
            out_dir=Path("out"),
        )
    """
    # -------------------------------------------------------------------------
    # Strict early validation
    # -------------------------------------------------------------------------
    if trf_cfg is not None and envelope_cfg is None:
        raise ValueError("envelope_cfg must be provided when trf_cfg is provided (TRF uses envelope).")

    if len(raws) == 0:
        raise ValueError("raws must be non-empty.")

    if len(raws) != len(run_ids):
        raise ValueError("raws and run_ids must have the same length (1:1 mapping).")

    for i, raw in enumerate(raws):
        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError(f"raws[{i}] is not an mne.io.BaseRaw.")

    analysis_cfg_final = analysis_cfg if analysis_cfg is not None else ClinicalAnalysisConfig()

    # -------------------------------------------------------------------------
    # Per-run containers (multi-run bundle data)
    # -------------------------------------------------------------------------
    raw_views_by_run: dict[str, dict[str, mne.io.BaseRaw]] = {}
    analysis_view_used_by_run: dict[str, str] = {}
    raw_analysis_by_run: dict[str, mne.io.BaseRaw] = {}
    preprocessing_artifacts_by_run: dict[str, list[Any]] = {}
    preprocessing_context_by_run: dict[str, PreprocContext] = {}
    envelopes_by_run: dict[str, dict[str, mne.io.BaseRaw]] = {}
    qc_by_run: dict[str, Any] = {}
    warnings_by_run: dict[str, list[str]] = {}

    # -------------------------------------------------------------------------
    # Step 1–3 per run: preprocessing -> choose view -> (optional) envelope
    # -------------------------------------------------------------------------
    for raw, run_id in zip(raws, run_ids):
        # Step 1: preprocessing
        preproc_result = run_clinical_preproc(
            raw=raw,
            electrodes_table=electrodes_table,
            cfg=preproc_cfg,
            coords_cfg=coords_cfg,
            ctx=ctx,
        )

        # Step 2: choose analysis view
        view_used, raw_analysis, view_warnings = choose_analysis_view(
            views=preproc_result.views,
            requested=analysis_cfg_final.analysis_view,
        )

        # Collect artifacts (may append envelope artifact below)
        artifacts = list(preproc_result.artifacts)

        # Step 3: optional gamma envelope per run
        run_envelopes: dict[str, mne.io.BaseRaw] = {}
        if envelope_cfg is not None:
            env_raw, env_artifact = compute_gamma_envelope(
                raw=raw_analysis,
                cfg=envelope_cfg,
                ctx=preproc_result.ctx,
            )
            artifacts.append(env_artifact)
            run_envelopes["gamma"] = env_raw

            # Provenance decisions (per run)
            preproc_result.ctx.decisions["analysis_view_requested"] = analysis_cfg_final.analysis_view
            preproc_result.ctx.decisions["analysis_view_used"] = view_used

        # Step 5 (optional, but per-run): QC figures
        if out_dir is not None:
            make_qc_figures(
                out_dir=out_dir,
                raw_original=preproc_result.views["original"],
                raw_analysis=raw_analysis,
                analysis_view_name=view_used,
                subject_id=subject_id,
                session_id=session_id,
                run_id=run_id,
            )

        # Step 6: QC summary per run
        qc = compute_clinical_qc(raw_views=preproc_result.views, include_channel_table=True)

        # Store per-run results
        raw_views_by_run[run_id] = preproc_result.views
        analysis_view_used_by_run[run_id] = view_used
        raw_analysis_by_run[run_id] = raw_analysis
        preprocessing_artifacts_by_run[run_id] = artifacts
        preprocessing_context_by_run[run_id] = preproc_result.ctx
        if run_envelopes:
            envelopes_by_run[run_id] = run_envelopes
        qc_by_run[run_id] = qc
        warnings_by_run[run_id] = list(view_warnings)

    # -------------------------------------------------------------------------
    # Step 4 (optional): TRF ONCE, using gamma envelope Epochs (epoch == run)
    # -------------------------------------------------------------------------
    trf_result: Optional[TRFResult] = None
    if trf_cfg is not None:
        if events_df is None:
            raise ValueError("events_df must be provided when trf_cfg is provided.")

        # Ensure gamma exists for ALL runs
        missing = [rid for rid in run_ids if rid not in envelopes_by_run or "gamma" not in envelopes_by_run[rid]]
        if missing:
            raise ValueError(f"Gamma envelope missing for runs: {missing}. Cannot run TRF.")

        gamma_env_raws = [envelopes_by_run[rid]["gamma"] for rid in run_ids]
        gamma_epochs = _gamma_raws_to_epochs(gamma_env_raws)

        runner = trf_runner if trf_runner is not None else run_trf_with_analysis_trf

        trf_input = TRFInput(
            signal_raw=gamma_epochs,
            events_df=events_df,
            subject_id=subject_id,
            task="diapix",  # DEBUG (should come from metadata)
            bids_root=bids_root,
        )

        trf_result = runner(trf_input, trf_cfg, bids_root, subject_id)

        # Export channel-wise TRF scores for reporting / 3D localization plot
        score_df = _build_trf_score_df(trf_result)
        trf_result.extra["score_df"] = score_df

        score_table_path = _maybe_write_trf_scores_table(score_df=score_df, out_dir=out_dir)
        if score_table_path is not None:
            trf_result.extra["score_table_path"] = str(score_table_path)

        # Helpful multi-run provenance
        trf_result.extra["run_ids"] = list(run_ids)
        trf_result.extra["analysis_view_used_by_run"] = dict(analysis_view_used_by_run)

    # -------------------------------------------------------------------------
    # Notes + electrodes normalization (shared across runs)
    # -------------------------------------------------------------------------
    bundle_notes = ClinicalAnalysisNotes(items=dict(notes) if notes is not None else {})

    electrodes_df, coords_space, electrode_warnings = _normalize_electrodes_df(
        electrodes_table=electrodes_table,
        coords_cfg=coords_cfg,
    )

    # Optionally attach electrode warnings into contexts (per run)
    if electrode_warnings:
        for rid in run_ids:
            preprocessing_context_by_run[rid].decisions["electrodes_normalization_warnings"] = electrode_warnings

    # -------------------------------------------------------------------------
    # Return multi-run bundle
    # -------------------------------------------------------------------------
    #
    # NOTE: This assumes ClinicalAnalysisBundle will be updated to a multi-run shape.
    # Next step: refactor the dataclass + report rendering to consume these fields.
    #
    run_bundles: dict[str, ClinicalRunBundle] = {}

    for rid in run_ids:
        run_bundles[rid] = ClinicalRunBundle(
            subject_id=subject_id,
            session_id=session_id,
            run_id=rid,
            raw_views=raw_views_by_run[rid],
            preprocessing_artifacts=preprocessing_artifacts_by_run[rid],
            preprocessing_context=preprocessing_context_by_run[rid],
            envelopes=envelopes_by_run.get(rid),
            qc=qc_by_run[rid],
            notes=ClinicalAnalysisNotes(items={}),  # or per-run notes if you have them
        )

    return ClinicalSessionBundle(
        subject_id=subject_id,
        session_id=session_id,
        run_ids=list(run_ids),
        runs=run_bundles,
        notes=bundle_notes,
        electrodes_df=electrodes_df,
        coords_space=coords_space,
        trf_result=trf_result,
        # optionally: trf=<converted ClinicalTrfResult>,
        warnings_by_run=warnings_by_run,
    )


# =============================================================================
#                                 Helpers
# =============================================================================

def _gamma_raws_to_epochs(env_raws: Sequence[mne.io.BaseRaw]) -> mne.EpochsArray:
    """
    Convert per-run gamma envelope Raw objects into an EpochsArray where epoch == run.

    Crops all runs to the common minimum length (n_times) and requires consistent
    channel ordering and sampling rate.

    Usage example
    -------------
        gamma_epochs = _gamma_raws_to_epochs([env_run1, env_run2, env_run3, env_run4])
    """
    if len(env_raws) == 0:
        raise ValueError("env_raws must be non-empty.")

    sfreq = float(env_raws[0].info["sfreq"])
    ch_names = list(env_raws[0].ch_names)

    for i, r in enumerate(env_raws[1:], start=1):
        if float(r.info["sfreq"]) != sfreq:
            raise ValueError(f"Gamma envelope run {i} has different sfreq.")
        if list(r.ch_names) != ch_names:
            raise ValueError(f"Gamma envelope run {i} has different channel names/order.")

    n_times = min(int(r.n_times) for r in env_raws)

    # EpochsArray expects shape (n_epochs, n_channels, n_times)
    data = np.stack([r.get_data()[:, :n_times] for r in env_raws], axis=0)

    info = env_raws[0].info.copy()

    # Dummy event table (one event per epoch)
    events = np.c_[
        np.arange(len(env_raws), dtype=int),
        np.zeros(len(env_raws), dtype=int),
        np.ones(len(env_raws), dtype=int),
    ]

    return mne.EpochsArray(data=data, info=info, events=events, tmin=0.0)


# =============================================================================
# TRF score export (for reporting / viz)
# =============================================================================

def _build_trf_score_df(trf_result: TRFResult) -> pd.DataFrame:
    """
    Build a channel-wise TRF score DataFrame from TRFResult.extra.

    Returns
    -------
    score_df
        Columns: channel, score
    """
    extra = trf_result.extra or {}
    channel_names = extra.get("channel_names", [])
    scores = extra.get("scores", None)

    if scores is None:
        raise ValueError("TRFResult.extra['scores'] is missing.")
    scores_arr = np.asarray(scores, dtype=float)

    if len(channel_names) != int(scores_arr.shape[0]):
        raise ValueError(
            "TRF scores shape does not match channel_names length: "
            f"{scores_arr.shape[0]} vs {len(channel_names)}."
        )

    return pd.DataFrame(
        {
            "channel": [str(x) for x in channel_names],
            "score": scores_arr.astype(float),
        }
    )


def _maybe_write_trf_scores_table(
    *,
    score_df: pd.DataFrame,
    out_dir: Optional[Path],
) -> Optional[Path]:
    """
    Write TRF score table to out_dir/tables if out_dir is provided.
    """
    if out_dir is None:
        return None
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "trf_scores.tsv"
    score_df.to_csv(out_path, sep="\t", index=False)
    return out_path
