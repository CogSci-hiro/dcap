# =============================================================================
# =============================================================================
#                         ###############################
#                         #   CLINICAL ANALYSIS BUNDLE  #
#                         ###############################
# =============================================================================
# =============================================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import mne
import numpy as np
import pandas as pd  # NEW

from dcap.seeg.preprocessing.types import BlockArtifact, PreprocContext
from dcap.seeg.clinical.qc import ClinicalQcSummary


@dataclass(frozen=True, slots=True)
class ClinicalTrfResult:
    backend: str
    analysis_view_name: str
    sfreq: float
    epoch_ids: List[str]

    # Backend-agnostic model params (from BackendFitResult)
    coef: np.ndarray
    intercept: np.ndarray

    # Optional summaries for report
    score_table_path: Optional[str]          # TSV/CSV path
    figures: Dict[str, str]                 # e.g. {"kernel": "...png", "scores": "...png"}
    warnings: List[str]


@dataclass(frozen=True)
class ClinicalAnalysisNotes:
    """
    Optional free-form notes attached to a clinical analysis run.

    Attributes
    ----------
    items
        Key-value notes.

    Usage example
    -------------
        notes = ClinicalAnalysisNotes(items={"rationale": "night-time recording"})
    """

    items: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ClinicalAnalysisBundle:
    """
    Immutable container passed from analysis to reporting.

    (docstring omitted for brevity)

    Electrode table format (example)
    -------------------------------
    | name | x     | y     | z     | space |
    |------|-------|-------|-------|-------|
    | LA1  | -34.2 | -12.0 | 18.5  | MNI   |
    | LA2  | -33.7 | -10.9 | 16.9  | MNI   |

    Usage example
    -------------
        bundle = ClinicalAnalysisBundle(
            subject_id="sub-001",
            session_id="ses-01",
            run_id="run-1",
            raw_views={"original": raw},
            preprocessing_artifacts=[],
            preprocessing_context=PreprocContext(),
            envelopes=None,
            qc=None,
            electrodes_df=None,
            coords_space=None,
            trf=None,
        )
    """

    subject_id: str
    session_id: Optional[str]
    run_id: Optional[str]

    raw_views: Mapping[str, mne.io.BaseRaw]

    preprocessing_artifacts: Sequence[BlockArtifact]
    preprocessing_context: PreprocContext

    envelopes: Optional[Mapping[str, mne.io.BaseRaw]] = None

    qc: Optional[ClinicalQcSummary] = None
    notes: ClinicalAnalysisNotes = field(default_factory=ClinicalAnalysisNotes)

    # -------------------------------------------------------------------------
    # Electrodes / localization
    # -------------------------------------------------------------------------
    electrodes_df: Optional[pd.DataFrame] = None
    coords_space: Optional[str] = None  # e.g. "MNI", "T1w", "patient"

    # -------------------------------------------------------------------------
    # TRF (typed, report-friendly)
    # -------------------------------------------------------------------------
    trf: Optional[ClinicalTrfResult] = None

    # -------------------------------------------------------------------------
    # Backward-compat placeholder (optional: remove once fully migrated)
    # -------------------------------------------------------------------------
    trf_result: Optional[Any] = None


# =============================================================================
#                     Clinical bundle types (run + session)
# =============================================================================

@dataclass(frozen=True)
class ClinicalRunBundle:
    """
    Immutable container for a SINGLE run (per-run outputs).

    This is the unit used for:
    - preprocessing views
    - per-run envelope computation
    - per-run QC summary
    - per-run QC figures (paths stored elsewhere)
    """
    subject_id: str
    session_id: Optional[str]
    run_id: str

    raw_views: Mapping[str, mne.io.BaseRaw]
    preprocessing_artifacts: Sequence[Any]
    preprocessing_context: Any

    envelopes: Optional[Mapping[str, mne.io.BaseRaw]] = None
    qc: Optional[Any] = None

    notes: Any = field(default_factory=lambda: ClinicalAnalysisNotes(items={}))


@dataclass(frozen=True)
class ClinicalSessionBundle:
    """
    Immutable container for MULTI-run clinical analysis passed from analysis to reporting.

    - `runs` stores per-run bundles.
    - `trf_result` (and/or `trf`) stores the single TRF fit over all runs (epochs).
    - electrodes metadata is shared across runs.

    Electrode table format (example)
    -------------------------------
    | name | x     | y     | z     | space |
    |------|-------|-------|-------|-------|
    | LA1  | -34.2 | -12.0 | 18.5  | MNI   |
    | LA2  | -33.7 | -10.9 | 16.9  | MNI   |
    """
    subject_id: str
    session_id: Optional[str]
    run_ids: Sequence[str]

    runs: Mapping[str, ClinicalRunBundle]

    # Shared notes for the whole session/report
    notes: Any = field(default_factory=lambda: ClinicalAnalysisNotes(items={}))

    # -------------------------------------------------------------------------
    # Electrodes / localization (shared)
    # -------------------------------------------------------------------------
    electrodes_df: Optional[pd.DataFrame] = None
    coords_space: Optional[str] = None

    # -------------------------------------------------------------------------
    # TRF (single model fit over all runs)
    # -------------------------------------------------------------------------
    trf: Optional[Any] = None
    trf_result: Optional[Any] = None

    # Optional: per-run warnings, decisions, etc.
    warnings_by_run: Optional[Mapping[str, Sequence[str]]] = None

    @property
    def _primary_run_id(self) -> str:
        return str(self.run_ids[0])

    @property
    def _primary_run(self) -> ClinicalRunBundle:
        return self.runs[self._primary_run_id]

    # ---------------- Legacy single-run aliases ----------------

    @property
    def preprocessing_context(self) -> Any:
        return self._primary_run.preprocessing_context

    @property
    def preprocessing_artifacts(self) -> Sequence[Any]:
        return self._primary_run.preprocessing_artifacts

    @property
    def raw_views(self) -> Mapping[str, mne.io.BaseRaw]:
        return self._primary_run.raw_views

    @property
    def envelopes(self) -> Optional[Mapping[str, mne.io.BaseRaw]]:
        return self._primary_run.envelopes

    @property
    def qc(self) -> Any:
        return self._primary_run.qc

    @property
    def run_id(self) -> str:
        return self._primary_run.run_id
