# Public registry template

This is a **shareable** template for the public registry.

- It must contain **no direct identifiers**
- It can be committed to GitHub
- It describes the structure of available data (subject/session/task/run) and QC state

## Minimum required columns

- subject
- session
- task
- run
- bids_root
- qc_status

Allowed values for `qc_status`:
- pass
- fail
- review
- unknown
