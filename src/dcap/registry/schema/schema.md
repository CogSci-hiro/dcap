# Registry schema (dcap)

This document defines the **public** and **private** registry schemas and the
rules for merging them at runtime.

`dcap` assumes **irregular participation**: not all subjects have all tasks,
sessions, or runs.

## Join keys (required)

Public and private registry tables are merged on the following keys:

- `subject`
- `session`
- `task`
- `run`

Rules:
- The join is a **left join** from public → (optional) private.
- Private metadata **must not** overwrite any public columns.
- Duplicate rows per join key are invalid (must be resolved upstream).

## Public registry (shareable)

Public registry data can be committed and shared. It must contain **no direct
identifiers** and no clinical notes.

### Required columns

| Column | Type | Notes |
|---|---|---|
| subject | str | BIDS subject label, e.g. `sub-001` |
| session | str | BIDS session label, e.g. `ses-01` (use `ses-01` even if only one session) |
| task | str | BIDS task label |
| run | int | Run index (1-based recommended) |
| bids_root | str | Path to the BIDS dataset root (can be relative) |
| qc_status | str | One of: `pass`, `fail`, `review`, `unknown` |

### Optional columns

| Column | Type | Notes |
|---|---|---|
| exclude_reason | str | Short machine-readable reason |
| dataset | str | Dataset label if multiple BIDS roots exist |
| notes_public | str | Non-sensitive note (avoid free text if possible) |

## Private registry (never committed)

Private registry lives outside the repo (e.g. `~/.dcap_private/registry_private.parquet`),
and may contain identifiers and clinical notes.

### Required columns

Must contain all join keys:

- `subject`, `session`, `task`, `run`

### Allowed private columns (examples)

| Column | Type | Notes |
|---|---|---|
| subject_key | str | Mapping key to local hospital identifier |
| recording_site | str | Site/hospital |
| acquisition_notes | str | Free text (sensitive) |
| clinician_notes | str | Free text (sensitive) |
| qc_reviewer | str | Initials/name |
| qc_timestamp | str | ISO timestamp |
| include_override | bool | Optional override flag |

## Output (merged / runtime view)

The merged registry is used by workflows. It includes public columns plus any
**non-conflicting** private columns (private columns are prefixed by default).

Default prefix for private columns: `private__`

