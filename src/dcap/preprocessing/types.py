# =============================================================================
#                          ############################
#                          #    PREPROCESSING TYPES   #
#                          ############################
# =============================================================================
#
# Shared dataclasses and small utilities for preprocessing blocks and pipelines.
#
# Design rules
# - Logic only (no CLI, no file I/O, no printing).
# - JSON-serializable artifacts for reporting/auditing.
# - A single provenance ledger appended by each block/pipeline.
#
# =============================================================================

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple


@dataclass(frozen=True)
class ProcRecord:
    """
    A single preprocessing step applied to a Raw object.

    Parameters
    ----------
    block_name
        Name of the block or pipeline step, e.g. "line_noise" or "rereference".
    parameters
        JSON-serializable parameters used for this step.
    created_utc
        UTC timestamp in ISO format.

    Usage example
    -------------
        record = ProcRecord(
            block_name="resample",
            parameters={"sfreq_out": 512.0},
            created_utc="2026-01-29T12:00:00Z",
        )
    """

    block_name: str
    parameters: Dict[str, Any]
    created_utc: str


@dataclass
class BlockArtifact:
    """
    JSON-serializable record produced by a preprocessing block or pipeline.

    Notes
    -----
    This is intended for downstream reporting and auditing.

    Usage example
    -------------
        artifact = BlockArtifact(
            name="line_noise",
            parameters={"method": "notch", "freq_base": 50},
            summary_metrics={"line_ratio_reduction_mean": 0.42},
            warnings=["zapline not enabled; notch used"],
            figures=[],
        )
    """

    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    summary_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    figures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a JSON-serializable dictionary.

        Usage example
        -------------
            payload = artifact.to_dict()
        """
        return asdict(self)


@dataclass
class Geometry:
    """
    Geometry information attached to a recording.

    This module does not validate naming conventions; validation belongs to a
    separate standardization layer.

    Attributes
    ----------
    coords_m
        Mapping from channel name to coordinate (x, y, z) in meters.
    neighbors
        Optional neighbor graph, used by local rereferencing methods (e.g., Laplacian).
    shafts
        Optional mapping from shaft name to ordered channel names.

    Usage example
    -------------
        geom = Geometry(coords_m={"A1": (0.0, 0.0, 0.0)})
    """

    coords_m: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    neighbors: Dict[str, List[str]] = field(default_factory=dict)
    shafts: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class PreprocContext:
    """
    Shared context for preprocessing.

    Blocks should store *decisions* and *intermediate metadata* here, while keeping
    the Raw signal transformations explicit in their return values.

    Attributes
    ----------
    proc_history
        Ordered list of `ProcRecord` entries appended by blocks/pipelines.
    geometry
        Geometry derived/attached in coordinate block.
    decisions
        Human / semi-automatic decisions to carry forward (e.g., final bad channel list).
    scratch
        Ephemeral space for intermediate computations (not intended to be saved).

    Usage example
    -------------
        ctx = PreprocContext()
        ctx.add_record("resample", {"sfreq_out": 512.0})
    """

    proc_history: List[ProcRecord] = field(default_factory=list)
    geometry: Optional[Geometry] = None
    decisions: MutableMapping[str, Any] = field(default_factory=dict)
    scratch: MutableMapping[str, Any] = field(default_factory=dict)

    def add_record(self, block_name: str, parameters: Mapping[str, Any]) -> None:
        """
        Append a preprocessing record to the provenance ledger.

        Usage example
        -------------
            ctx.add_record("line_noise", {"method": "notch", "freq_base": 50})
        """
        created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.proc_history.append(
            ProcRecord(
                block_name=block_name,
                parameters=dict(parameters),
                created_utc=created_utc,
            )
        )


@dataclass(frozen=True)
class BadChannelReason:
    """
    Human-readable reason for suggesting a channel as bad.

    Parameters
    ----------
    reason_type
        Machine-readable label, e.g. "flat", "high_variance".
    value
        Observed metric value.
    threshold
        Threshold used to flag the channel.
    note
        Optional explanation.

    Usage example
    -------------
        r = BadChannelReason("flat", value=1e-12, threshold=1e-10, note="variance too low")
    """

    reason_type: str
    value: float
    threshold: float
    note: str = ""
