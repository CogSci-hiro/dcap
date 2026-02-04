from dataclasses import dataclass
from typing import Any, Mapping, Literal, Optional

@dataclass(frozen=True)
class DerivativeArtifact:
    pipeline: str                 # e.g. "dcap-qc"
    datatype: str                 # e.g. "ieeg"
    desc: str                     # e.g. "triggeralign"
    suffix: str                   # e.g. "qc"
    extension: str                # ".json"
    content: Mapping[str, Any]    # JSON-serializable
