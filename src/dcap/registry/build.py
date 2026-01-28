# dcap/registry/build.py
# =============================================================================
#                         Registry: public builder
# =============================================================================
#
# Build a sanitized, shareable public registry TSV from private metadata.
#
# Private metadata is authoritative and may contain sensitive information.
# Public registry is a derived artifact and must contain no sensitive fields.
#
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import csv

import yaml


# =============================================================================
# Constants
# =============================================================================

BIDS_SUBJECT_PATTERN = r"^sub-\d{3}$"

PUBLIC_REGISTRY_COLUMNS: Tuple[str, ...] = (
    "dataset_id",
    "bids_subject",
    "dcap_id",
    "session",
    "acquisition_id",
    "protocol_id",
    "n_acquisitions_in_session",
    "has_private_run_decisions",
    "source_private_subject_file",
)


# =============================================================================
# Data containers
# =============================================================================

@dataclass(frozen=True)
class PublicRegistryRow:
    """A single row in the public (sanitized) registry.

    Notes
    -----
    This row is intentionally structural and must not include sensitive fields
    (e.g., name, DOB, free-form notes).

    Usage example
    -------------
        row = PublicRegistryRow(
            dataset_id="my_dataset",
            bids_subject="sub-001",
            dcap_id="DCAP_000123",
            session="ses-001",
            acquisition_id="acq-01",
            protocol_id="prot-01",
            n_acquisitions_in_session=2,
            has_private_run_decisions=False,
            source_private_subject_file="subjects/sub-001.yaml",
        )
    """

    dataset_id: str
    bids_subject: str
    dcap_id: str
    session: str
    acquisition_id: str
    protocol_id: str
    n_acquisitions_in_session: int
    has_private_run_decisions: bool
    source_private_subject_file: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dict suitable for TSV writing.

        Usage example
        -------------
            as_dict = row.to_dict()
        """
        return {
            "dataset_id": self.dataset_id,
            "bids_subject": self.bids_subject,
            "dcap_id": self.dcap_id,
            "session": self.session,
            "acquisition_id": self.acquisition_id,
            "protocol_id": self.protocol_id,
            "n_acquisitions_in_session": str(self.n_acquisitions_in_session),
            "has_private_run_decisions": "1" if self.has_private_run_decisions else "0",
            "source_private_subject_file": self.source_private_subject_file,
        }


# =============================================================================
# I/O helpers
# =============================================================================

def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def _write_tsv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# =============================================================================
# Core build logic
# =============================================================================

def build_public_registry(
    *,
    public_registry_out: Path,
    private_root: Path,
    public_registry_in: Optional[Path] = None,
    dataset_id: Optional[str] = None,
    strict: bool = False,
) -> Path:
    """Build a sanitized public registry TSV from private metadata.

    Parameters
    ----------
    public_registry_out
        Output path for the derived, shareable `registry_public.tsv`.
    private_root
        Root directory containing private metadata:
        - subject_keys.yaml
        - subjects/sub-*.yaml
        - (optional) registry_private.tsv
    public_registry_in
        Optional existing public registry path. If provided, builder may later be
        extended to compare/confirm determinism. Currently ignored (placeholder).
    dataset_id
        Dataset scope identifier. If None, attempts to infer from `subject_keys.yaml`.
        If your `subject_keys.yaml` contains multiple dataset scopes, this must be set.
    strict
        If True, treat some soft issues as errors (e.g., missing optional files).

    Returns
    -------
    Path
        Path to the written `public_registry_out`.

    Notes
    -----
    The builder intentionally does *not* propagate any subject identity fields
    (name, DOB, notes, etc.). It only emits structural indexing fields.

    Output format example
    ---------------------
    When we say the builder outputs a TSV, this is the expected format:

    +-----------+------------+----------+---------+---------------+------------+--------------------------+--------------------------+----------------------------+
    | dataset_id| bids_subject| dcap_id  | session | acquisition_id | protocol_id| n_acquisitions_in_session| has_private_run_decisions| source_private_subject_file|
    +===========+============+==========+=========+===============+============+==========================+==========================+============================+
    | ds1       | sub-001    | D001     | ses-001  | acq-01        | prot-01    | 2                        | 0                        | subjects/sub-001.yaml      |
    +-----------+------------+----------+---------+---------------+------------+--------------------------+--------------------------+----------------------------+

    Usage example
    -------------
        from pathlib import Path
        from dcap.registry.build import build_public_registry

        out_path = build_public_registry(
            public_registry_out=Path("registry_public.tsv"),
            private_root=Path("/secure/DCAP_PRIVATE_ROOT"),
            dataset_id="ds1",
            strict=False,
        )
        print(out_path)
    """
    private_root = private_root.expanduser().resolve()
    subject_keys_path = private_root / "subject_keys.yaml"
    subjects_dir = private_root / "subjects"
    private_run_decisions_path = private_root / "registry_private.tsv"

    subject_keys = _read_yaml(subject_keys_path)

    # Expect: mapping by dataset_id -> list/dict of {bids_subject: ..., dcap_id: ...}
    # We'll support two shapes:
    # A) {dataset_id: {sub-001: D001, sub-002: D002}}
    # B) {dataset_id: [{bids_subject: sub-001, dcap_id: D001}, ...]}
    subject_keys = _read_yaml(subject_keys_path)

    datasets_block = subject_keys.get("datasets")
    if not isinstance(datasets_block, dict):
        raise ValueError("subject_keys.yaml must contain a top-level 'datasets' mapping.")

    available_dataset_ids = sorted(datasets_block.keys())

    chosen_dataset_id = dataset_id
    if chosen_dataset_id is None:
        if len(available_dataset_ids) == 1:
            chosen_dataset_id = available_dataset_ids[0]
        else:
            raise ValueError(
                "subject_keys.yaml contains multiple dataset scopes under 'datasets'; "
                "please pass dataset_id explicitly."
            )

    if chosen_dataset_id not in datasets_block:
        raise ValueError(
            f"dataset_id={chosen_dataset_id!r} not found in subject_keys.yaml.datasets. "
            f"Available: {available_dataset_ids}"
        )

    subject_mapping = _normalize_subject_keys(datasets_block[chosen_dataset_id])

    private_run_decisions = None
    if private_run_decisions_path.exists():
        private_run_decisions = _read_tsv(private_run_decisions_path)
    else:
        if strict:
            raise FileNotFoundError(
                f"Missing optional private run decisions TSV in strict mode: {private_run_decisions_path}"
            )

    decision_index = _index_private_run_decisions(private_run_decisions or [])

    rows: List[PublicRegistryRow] = []
    for bids_subject, dcap_id in sorted(subject_mapping.items()):
        subject_file = subjects_dir / f"{bids_subject}.yaml"
        if not subject_file.exists():
            # Builder is allowed to fail loudly: public registry must reflect private truth.
            raise FileNotFoundError(f"Missing private subject file: {subject_file}")

        subject_data = _read_yaml(subject_file)
        acquisitions = subject_data.get("acquisitions", [])
        if acquisitions is None:
            acquisitions = []
        if not isinstance(acquisitions, list):
            raise ValueError(f"Expected 'acquisitions' list in: {subject_file}")

        # Count acquisitions per session for convenience
        session_counts = _count_acquisitions_per_session(acquisitions)

        for acq in acquisitions:
            acquisition_id = str(acq.get("acquisition_id", "")).strip()
            session = str(acq.get("session", "")).strip()
            protocol_id = str(acq.get("protocol_id", "") or "").strip()

            if not acquisition_id or not session:
                # Let validation decide later if you want, but builder should avoid emitting junk.
                raise ValueError(
                    f"Acquisition missing required fields (acquisition_id/session) in {subject_file}: {acq}"
                )

            has_decisions = _has_decisions_for_subject_session_acq(
                decision_index=decision_index,
                bids_subject=bids_subject,
                session=session,
                acquisition_id=acquisition_id,
            )

            rows.append(
                PublicRegistryRow(
                    dataset_id=chosen_dataset_id,
                    bids_subject=bids_subject,
                    dcap_id=dcap_id,
                    session=session,
                    acquisition_id=acquisition_id,
                    protocol_id=protocol_id,
                    n_acquisitions_in_session=session_counts.get(session, 0),
                    has_private_run_decisions=has_decisions,
                    source_private_subject_file=str(Path("subjects") / f"{bids_subject}.yaml"),
                )
            )

    # Write TSV
    _write_tsv(
        public_registry_out,
        rows=[r.to_dict() for r in rows],
        fieldnames=PUBLIC_REGISTRY_COLUMNS,
    )
    return public_registry_out


# =============================================================================
# Normalization + indexing helpers
# =============================================================================

def _normalize_subject_keys(dataset_block: Any) -> Dict[str, str]:
    """Normalize subject_keys.yaml dataset block into {bids_subject: dcap_id}.

    Supported shapes
    ----------------
    A) Mapping:
        sub-001: D001
        sub-002: D002

    B) List of mappings:
        - bids_subject: sub-001
          dcap_id: D001
        - bids_subject: sub-002
          dcap_id: D002

    Usage example
    -------------
        mapping = _normalize_subject_keys({"sub-001": "D001"})
    """
    if isinstance(dataset_block, dict):
        # Assume {sub-001: D001}
        out: Dict[str, str] = {}
        for k, v in dataset_block.items():
            bids_subject = str(k).strip()
            dcap_id = str(v).strip()
            out[bids_subject] = dcap_id
        return out

    if isinstance(dataset_block, list):
        out = {}
        for item in dataset_block:
            if not isinstance(item, dict):
                raise ValueError("subject_keys.yaml list entries must be mappings.")
            bids_subject = str(item.get("bids_subject", "")).strip()
            dcap_id = str(item.get("dcap_id", "")).strip()
            if not bids_subject or not dcap_id:
                raise ValueError(f"Invalid subject_keys entry: {item}")
            out[bids_subject] = dcap_id
        return out

    raise ValueError("Unsupported subject_keys.yaml dataset block shape.")


def _count_acquisitions_per_session(acquisitions: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for acq in acquisitions:
        session = str(acq.get("session", "")).strip()
        if session:
            counts[session] = counts.get(session, 0) + 1
    return counts


def _index_private_run_decisions(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, str, str], bool]:
    """Index private run decisions at (bids_subject, session, acquisition_id) granularity.

    This is deliberately coarse for v1. Later, you can index by record_id/run_id.

    Usage example
    -------------
        idx = _index_private_run_decisions([{"bids_subject":"sub-001","session":"ses-001","acquisition_id":"acq-01"}])
    """
    index: Dict[Tuple[str, str, str], bool] = {}
    for row in rows:
        bids_subject = str(row.get("bids_subject", "")).strip()
        session = str(row.get("session", "")).strip()
        acquisition_id = str(row.get("acquisition_id", "")).strip()

        if bids_subject and session and acquisition_id:
            index[(bids_subject, session, acquisition_id)] = True
    return index


def _has_decisions_for_subject_session_acq(
    *,
    decision_index: Dict[Tuple[str, str, str], bool],
    bids_subject: str,
    session: str,
    acquisition_id: str,
) -> bool:
    return decision_index.get((bids_subject, session, acquisition_id), False)
