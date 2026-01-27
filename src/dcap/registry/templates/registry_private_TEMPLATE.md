# Private registry template (NEVER COMMIT)

This is a **local-only** template for the private registry.

- It may contain identifiers and clinical notes
- It must live outside the repo (e.g. `~/.dcap_private/registry_private.csv`)
- It must **never** be committed or synced to public remotes

## Required columns

- subject
- session
- task
- run

## Suggested sensitive columns (examples)

- subject_key
- clinician_notes
- acquisition_notes
- qc_reviewer
- qc_timestamp
