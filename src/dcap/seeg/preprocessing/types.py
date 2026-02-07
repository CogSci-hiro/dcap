# =============================================================================
# =============================================================================
#                     ########################################
#                     #       PREPROCESSING CORE TYPES       #
#                     ########################################
# =============================================================================
# =============================================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import mne


@dataclass(frozen=True)
class BlockArtifact:
    """
    Artifact emitted by a preprocessing block.

    Attributes
    ----------
    name
        Block name (e.g., "line_noise").
    parameters
        Serialized parameters for provenance (dataclasses.asdict-friendly).
    summary_metrics
        Small scalar metrics for reporting/QC.
    warnings
        Human-readable warnings to surface in clinical reports.
    figures
        Paths or opaque figure handles (logic-only; rendering decides what to do).

    Usage example
    -------------
        artifact = BlockArtifact(
            name="resample",
            parameters={"sfreq_out": 512.0},
            summary_metrics={"changed": True},
            warnings=[],
            figures=[],
        )
    """

    name: str
    parameters: Mapping[str, Any]
    summary_metrics: Mapping[str, float]
    warnings: Sequence[str] = field(default_factory=list)
    figures: Sequence[Any] = field(default_factory=list)


@dataclass(frozen=True)
class Geometry:
    """
    Geometry information for electrodes.

    Attributes
    ----------
    coords_m
        Mapping channel -> (x, y, z) in meters.
    neighbors
        Optional neighbor graph for Laplacian-like referencing.
    shafts
        Optional mapping shaft -> ordered channel names.

    Usage example
    -------------
        geom = Geometry(coords_m={"A1": (0.0, 0.0, 0.0)}, neighbors={}, shafts={})
    """

    coords_m: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    neighbors: Mapping[str, Sequence[str]] = field(default_factory=dict)
    shafts: Mapping[str, Sequence[str]] = field(default_factory=dict)


@dataclass
class PreprocContext:
    """
    Preprocessing context: provenance ledger + decisions.

    Attributes
    ----------
    proc_history
        Ordered list of provenance records.
    decisions
        Free-form decisions made by blocks/pipelines (e.g., bad channels).
    geometry
        Optional electrode geometry.

    Usage example
    -------------
        ctx = PreprocContext()
        ctx.add_record("line_noise", {"method": "notch"})
    """

    proc_history: List[Dict[str, Any]] = field(default_factory=list)
    decisions: MutableMapping[str, Any] = field(default_factory=dict)
    geometry: Optional[Geometry] = None

    def add_record(self, step: str, parameters: Mapping[str, Any]) -> None:
        self.proc_history.append({"step": step, "parameters": dict(parameters)})


@dataclass(frozen=True)
class PreprocResult:
    """
    Result of a preprocessing pipeline.

    Attributes
    ----------
    views
        Mapping from view name to Raw object. Must contain "original".
    artifacts
        Ordered artifacts for reporting.
    ctx
        Context (provenance + decisions).

    Usage example
    -------------
        result = PreprocResult(views={"original": raw}, artifacts=[], ctx=PreprocContext())
    """

    views: Mapping[str, mne.io.BaseRaw]
    artifacts: Sequence[BlockArtifact]
    ctx: PreprocContext


@dataclass(frozen=True)
class BadChannelReason:
    """
    Explanation for why a channel was marked as bad.

    This object is designed to be:
    - human-readable (for reports and QC)
    - machine-readable (for filtering / aggregation)
    - serializable (stored in ctx.decisions and artifacts)

    Parameters
    ----------
    code
        Short machine-readable identifier for the reason.
        Examples:
        - "bids_channels_tsv"
        - "flat_signal"
        - "excessive_line_noise"
        - "manual_override"

    message
        Human-readable explanation.
        This should be suitable for direct display in logs or reports.

    source
        Origin of the decision.
        Examples:
        - "bids"
        - "automatic"
        - "manual"
        - "imported"

    metric
        Optional numeric metric associated with the decision
        (e.g., variance, z-score, noise ratio).

    threshold
        Optional threshold used to justify the decision.
    """

    code: str
    message: str
    source: str = "automatic"
    metric: Optional[float] = None
    threshold: Optional[float] = None
