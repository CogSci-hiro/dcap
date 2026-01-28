# Specification: Public Registry (`registry_public`)

## Purpose
Canonical, shareable inventory of all available data units in a BIDS dataset.

Defines **what exists**, never why or how it should be used.

## Privacy level
**Public / shareable**  
Safe for Git, collaborators, publications.

## One row represents
One `(dataset_id, subject, session, task, run, datatype)` tuple.

## Required columns

| Column | Type | Description |
|------|------|-------------|
| dataset_id | str | Short identifier for dataset/site |
| bids_root | str | Path or token for BIDS root |
| subject | str | BIDS subject ID (`sub-XXX`) |
| session | str | BIDS session ID (`ses-XX`) |
| task | str | BIDS task name |
| run | str | Run index (string, not int) |
| datatype | str | BIDS datatype (`ieeg`, `beh`, etc.) |
| record_id | str | Stable unique identifier |

## record_id format
```
{dataset_id}|{subject}|{session}|{task}|{run}|{datatype}
```

## Optional columns
- bids_relpath
- exists

## Forbidden content
- Names, DOB, medication, acquisition dates
- QC decisions or notes
