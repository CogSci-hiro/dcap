# Specification: Private Run-Level Registry (`registry_private.tsv`)

## Purpose
The private run-level registry provides **run-specific annotations** that are
**never shared or version-controlled**. It augments the public registry with
human decisions, provenance, and sensitive notes, without redefining dataset
structure.

This file is designed to be **manually editable** (e.g. in a spreadsheet editor).

---

## Privacy level
**Private / local only**

- Must live outside the repository (e.g. under `$DCAP_PRIVATE_ROOT`)
- Must never be committed to Git
- May contain sensitive or identifying information

---

## File format

- Tab-separated values (TSV)
- UTF-8 encoded
- One header row
- One row per run-level record

---

## Canonical template

| record_id | dcap_id | exclude_reason | review_date | notes   |
|-----------|---------|----------------|-------------|---------|
| 001       | JoDo    | NA             | 2000-01-01  | example |


---

## Semantics

### One row represents
One **run-level record**, identified by `record_id`, corresponding to exactly
one row in the public registry.

---

## Columns (in order)

### `record_id` (required)
- **Type:** string
- **Role:** Primary key
- **Description:**  
  Stable identifier defined by the public registry.

Must match exactly one `record_id` in `registry_public`.

---

### `dcap_id` (required)
- **Type:** string
- **Role:** Reviewer / annotator identifier
- **Description:**  
  Initials or short identifier of the person who reviewed or annotated this run.

Examples:
- `HY`
- `JD`
- `JoDo`

---

### `exclude_reason` (optional)
- **Type:** string
- **Role:** Human-readable exclusion rationale
- **Description:**  
  Free-text reason explaining why a run should be excluded or treated with
  caution.

Conventions:
- Use `NA` if not excluded
- Short, descriptive phrases preferred

Examples:
- `amplifier dropout`
- `missing audio`
- `excessive line noise`

---

### `review_date` (optional but recommended)
- **Type:** date (ISO 8601)
- **Format:** `YYYY-MM-DD`
- **Role:** Provenance
- **Description:**  
  Date on which the run was reviewed or annotated.

---

### `notes` (optional)
- **Type:** string
- **Role:** Free private annotation
- **Description:**  
  Arbitrary free-text notes. May contain sensitive information and clinical
  context.

No length or content restrictions apply.

---

## Interpretation rules

- Presence of a row indicates that the run has been **explicitly reviewed**
- Absence of a row implies:
  - no private decision recorded
  - default handling determined by public metadata
- This file:
  - **does not define availability**
  - **does not define dataset structure**
  - **does not override BIDS entities**

All joins with public metadata are performed explicitly via `record_id`.

---

## Forbidden content

The private registry must **not** redefine or include:

- subject identifiers (`sub-XXX`)
- session, task, or run columns
- acquisition dates or locations
- medication history
- demographic fields

Such information belongs in subject-level private YAML files.

---

## Design rationale

This minimal schema intentionally avoids:
- wide tables
- variable-length columns
- implicit logic

It provides:
- human traceability
- review provenance
- maximal privacy isolation

---

## Summary

- **Public registry** answers *what exists*
- **Private run-level registry** answers *what we think about it*
- Linking is explicit, local, and reversible
