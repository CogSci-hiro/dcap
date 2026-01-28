# Specification: Subject Re-identification Map (`subject_keys.yaml`)

## Purpose
Map anonymized BIDS subjects to **local clinical identifiers**.

Used only for raw data handling and audits.

## Privacy level
**Highly sensitive / private**

## Structure
Top-level mapping by dataset_id.

## Required fields per entry
- bids_subject
- dcap_id

## Optional fields
- site
- implant_date
- notes

## Forbidden usage
- Must never be merged into registries
- Must never be exported or sanitized
