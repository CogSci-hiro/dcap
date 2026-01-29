# =============================================================================
# =============================================================================
#                         ###############################
#                         #   CLINICAL ANALYSIS BUNDLE  #
#                         ###############################
# =============================================================================
# =============================================================================
#
# A single immutable container that the clinical report consumes.
#
# Design rules
# - Logic only: no file I/O, no CLI, no printing.
# - Reporting consumes ONLY this bundle (plus static templates).
# - Bundle must remain valid even when optional analyses (e.g., TRF) are missing.
#
# =============================================================================

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import mne

from dcap.preprocessing.types import BlockArtifact, PreprocContext


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

    Attributes
    ----------
    subject_id
        BIDS subject identifier (e.g., "sub-001").
    session_id
        Optional BIDS session identifier (e.g., "ses-01").
    run_id
        Optional run identifier (e.g., "run-1").
    raw_views
        Mapping from view name to Raw object (e.g., "original", "car", "bipolar").
    preprocessing_artifacts
        Ordered artifacts emitted by preprocessing blocks/pipelines.
    preprocessing_context
        Provenance ledger + decisions (bad channels, geometry, etc.).
    envelopes
        Optional mapping from envelope name to Raw (e.g., {"gamma": RawArray}).
    trf_result
        Optional TRF result object (analysis-specific).
    notes
        Optional notes for reporting.

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
            trf_result=None,
        )
    """

    subject_id: str
    session_id: Optional[str]
    run_id: Optional[str]

    raw_views: Mapping[str, mne.io.BaseRaw]

    preprocessing_artifacts: Sequence[BlockArtifact]
    preprocessing_context: PreprocContext

    envelopes: Optional[Mapping[str, mne.io.BaseRaw]] = None
    trf_result: Optional[Any] = None

    notes: ClinicalAnalysisNotes = field(default_factory=ClinicalAnalysisNotes)
