# Specification: Subject-Level Private Metadata (`subjects/sub-XXX.yaml`)

## Purpose
Subject-centric private metadata capturing **identity, history, and context**.

Designed for manual editing.

## Privacy level
**Private / local only**

## File scope
One file per subject.

## Required top-level keys
- subject
- dataset_id

## Allowed sections

### identity
Stable subject identity.
- name (private)
- sex
- date_of_birth

### implantation
Hardware and implantation details.
- electrode_type
- manufacturer
- notes

### acquisitions (list)
Repeatable acquisition events.
- acquisition_id (unique)
- date
- place
- session
- notes

### medication (list)
Time-indexed medication history.
- start_date
- end_date (optional)
- drugs (list of name + dose)

### protocols (list)
Experimental protocol definitions.
- protocol_id (unique)
- task
- sessions
- setup
- notes

### notes (list)
Free annotations.
- date
- author
- text

## Design rules
- No run-level annotations
- No public exports without sanitization
