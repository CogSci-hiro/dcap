# Specification: Private Run-Level Registry (`registry_private.tsv`)

## Purpose
Run-level **private overlay** for QC decisions, exclusions, and notes.

Augments the public registry but never redefines it.

## Privacy level
**Private / local only**  
Must never be committed.

## One row represents
One `record_id` from the public registry.

## Required columns

| Column | Type | Description |
|------|------|-------------|
| record_id | str | Join key to public registry |

## Recommended columns

| Column | Type | Description |
|------|------|-------------|
| qc_status | enum | pass / fail / review / unknown |
| exclude | 0/1 | Hard exclusion flag |
| exclude_reason | str | Free text (private) |
| notes | str | Free text (private) |
| tags | str | Comma-separated tags |
| reviewer | str | Initials or name |
| review_date | date | ISO `YYYY-MM-DD` |

## Rules
- Cannot redefine subject/session/task/run
- Missing rows imply default QC = unknown
