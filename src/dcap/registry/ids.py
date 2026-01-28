# src/dcap/registry/ids.py
# =============================================================================
#                       Registry IDs: stable record_id
# =============================================================================
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class RecordIdSpec:
    """
    Specification for constructing stable record identifiers.

    A record_id must be:
    - deterministic
    - derived from non-sensitive identifiers
    - stable across machines / paths

    Notes
    -----
    The default format is a pipe-separated key:

        "{dataset_id}|{subject}|{session}|{task}|{run}|{datatype}"

    Usage example
    -------------
        spec = RecordIdSpec()
        rid = spec.make_record_id(
            dataset_id="siteA_2024",
            subject="sub-001",
            session="ses-01",
            task="conversation",
            run="1",
            datatype="ieeg",
        )
    """

    delimiter: str = "|"

    def make_record_id(
        self,
        dataset_id: str,
        subject: str,
        session: str,
        task: str,
        run: str,
        datatype: str,
    ) -> str:
        """
        Construct a stable record_id.

        Parameters
        ----------
        dataset_id
            Short identifier for the dataset (e.g., "siteA_2024").
        subject
            BIDS subject (e.g., "sub-001").
        session
            BIDS session (e.g., "ses-01"). Use "ses-01" even if you only have one.
        task
            BIDS task (e.g., "conversation").
        run
            Run index as string (e.g., "1"). Keep it string to avoid "01" vs "1" issues.
        datatype
            BIDS datatype (e.g., "ieeg", "beh", "eeg").

        Returns
        -------
        str
            Stable record identifier.

        Usage example
        -------------
            spec = RecordIdSpec()
            rid = spec.make_record_id(
                dataset_id="siteA_2024",
                subject="sub-001",
                session="ses-01",
                task="conversation",
                run="1",
                datatype="ieeg",
            )
        """
        parts = (dataset_id, subject, session, task, run, datatype)
        for part in parts:
            if self.delimiter in part:
                raise ValueError(
                    f"Delimiter {self.delimiter!r} may not appear in record_id parts; got part={part!r}."
                )
        return self.delimiter.join(parts)


def parse_record_id(record_id: str, delimiter: str = "|") -> tuple[str, str, str, str, str, str]:
    """
    Parse a record_id into its component fields.

    Parameters
    ----------
    record_id
        Stable record identifier string.
    delimiter
        Delimiter used in record_id.

    Returns
    -------
    tuple
        (dataset_id, subject, session, task, run, datatype)

    Usage example
    -------------
        dataset_id, subject, session, task, run, datatype = parse_record_id("siteA_2024|sub-001|ses-01|conversation|1|ieeg")
    """
    parts = record_id.split(delimiter)
    if len(parts) != 6:
        raise ValueError(f"Expected 6-part record_id, got {len(parts)} parts: {record_id!r}")
    dataset_id, subject, session, task, run, datatype = parts
    return dataset_id, subject, session, task, run, datatype
