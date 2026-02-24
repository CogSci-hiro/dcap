# Diapix New-Data Intake Workflow (Current Repo Procedure)

This document describes the current end-to-end workflow in this repo for bringing newly acquired **Diapix** data into `dcap`, from **private metadata updates** to **BIDS conversion** and registry outputs.

It reflects the current code behavior, not just the high-level privacy design docs.

## Scope

This covers:

- private subject mapping (`subject_keys.yaml`)
- private subject metadata (`subjects/sub-XXX.yaml`)
- optional private run-level review registry (`registry_private.tsv`)
- Diapix task assets required for conversion
- Diapix BIDS conversion (`dcap bids-convert`)
- public registry build/validation (`dcap registry ...`)

## Important Ordering Note (Current Implementation)

The high-level registry privacy doc describes a conceptual order of:

1. convert raw to BIDS
2. build public registry
3. update private metadata

For **Diapix conversion in the current code**, that order is not sufficient in practice because `dcap bids-convert` must resolve the private `dcap_id` from `subject_keys.yaml` before it can find the raw source folder and task metadata.

In practice, you must do a **minimal private metadata update first** (at least `subject_keys.yaml`, and usually `subjects/sub-XXX.yaml` too).

## Directory / File Overview

### Private metadata root (local only)

Set a private root (never committed):

- `$DCAP_PRIVATE_ROOT/subject_keys.yaml`
- `$DCAP_PRIVATE_ROOT/subjects/sub-XXX.yaml`
- `$DCAP_PRIVATE_ROOT/registry_private.tsv` (optional)

Templates live in:

- `/Users/hiro/PycharmProjects/dcap/docs/registry/templates/subject_keys_TEMPLATE.yaml`
- `/Users/hiro/PycharmProjects/dcap/docs/registry/templates/subject_TEMPLATE.yaml`
- `/Users/hiro/PycharmProjects/dcap/docs/registry/templates/registry_private_TEMPLATE.tsv`

### Raw Diapix source layout expected by current task code

`dcap bids-convert` resolves `dcap_id` and then looks under:

- `<source-root>/<dcap_id>/`

Inside that folder, the Diapix task discovers runs from:

- `conversation_*.vhdr`

For each run `N`, it requires:

- `conversation_N.vhdr`
- `conversation_N.wav`

Optional:

- `conversation_N.asf` (copied to BIDS `video/`)

### Diapix task assets directory (required)

Pass via `--task-assets-dir`. Current Diapix task factory expects:

- `audio_onsets.tsv`
- `beeps_pre-task-1-sec_post-task-4-sec.wav`
- `elec2atlas.mat`

`audio_onsets.tsv` is used for audio cropping and must include:

- `dcap_id`
- `run`
- `onset` (or `onset_s`)

## Step-by-Step Workflow

## 1. Set / confirm private metadata root

```bash
export DCAP_PRIVATE_ROOT=/absolute/path/to/dcap_private
mkdir -p "$DCAP_PRIVATE_ROOT/subjects"
```

If you do not export `DCAP_PRIVATE_ROOT`, you must pass `--private-root` to commands.

## 2. Add the subject mapping in `subject_keys.yaml` (required before conversion)

File:

- `$DCAP_PRIVATE_ROOT/subject_keys.yaml`

Add/update the dataset entry mapping the BIDS subject to the private clinical ID (`dcap_id`):

```yaml
version: 1
datasets:
  Timone2025:
    - bids_subject: sub-001
      dcap_id: Nic-Ele
```

Notes:

- `dataset_id` must match the value you pass to `--dataset-id`.
- `bids_subject` is the anonymized BIDS-facing ID.
- `dcap_id` is the private/raw identifier and is used to locate `<source-root>/<dcap_id>`.

## 3. Add/update subject private metadata YAML (recommended before conversion; required for registry build)

File:

- `$DCAP_PRIVATE_ROOT/subjects/sub-001.yaml`

Populate at least:

- `subject`
- `dataset_id`
- `acquisitions` (with `acquisition_id`, `session`)
- `protocols` (with `protocol_id`, `task`, `sessions`)

For Diapix, ensure there is a protocol entry with `task: diapix` and the correct `sessions` list.

This file drives the public registry builder (the current `build-public` command builds from private YAML, not by scanning BIDS outputs).

## 4. (If needed) add trigger IDs for the new subject in code

Current Diapix event generation uses a hardcoded trigger lookup:

- `/Users/hiro/PycharmProjects/dcap/src/dcap/bids/tasks/diapix/triggers.py`

If the new `dcap_id` is not present in `_TRIGGER_ID_MAP`, conversion will fail during event preparation.

Add an entry for the subject and each run, e.g.:

```python
"New-Subj": {"1": 10005, "2": 10004, "3": 10005, "4": 10005}
```

## 5. Prepare / verify Diapix task assets directory

Create or reuse a Diapix assets folder (private/local), then confirm it contains:

- `audio_onsets.tsv`
- `beeps_pre-task-1-sec_post-task-4-sec.wav`
- `elec2atlas.mat`

Also ensure `audio_onsets.tsv` contains rows for the new `dcap_id` and each run.

Current code caveat:

- The task factory also performs an existence check on `--task-assets-dir/<dcap_id>` (likely unintended, but it is current behavior). If conversion fails with a missing-file error that points there, inspect `/Users/hiro/PycharmProjects/dcap/src/dcap/bids/tasks/registry.py`.

## 6. Prepare / verify raw Diapix source files

Assume:

- `--source-root /data/raw/diapix`
- `dcap_id = Nic-Ele`

Then the converter expects raw files in:

- `/data/raw/diapix/Nic-Ele/`

Example contents:

- `conversation_1.vhdr`
- `conversation_1.wav`
- `conversation_1.asf` (optional)
- `conversation_2.vhdr`
- `conversation_2.wav`
- ...

## 7. Run a dry-run BIDS conversion first

Use the CLI entrypoint installed by this repo (`dcap`).

```bash
dcap bids-convert \
  --task diapix \
  --dataset-id Timone2025 \
  --source-root /data/raw/diapix \
  --bids-root /data/bids/diapix_bids \
  --subject sub-001 \
  --private-root "$DCAP_PRIVATE_ROOT" \
  --task-assets-dir /data/private_assets/diapix \
  --dry-run \
  --preload-raw \
  --line-freq-hz 50
```

What this does:

- resolves `dcap_id` from `subject_keys.yaml`
- reads raw data from `<source-root>/<dcap_id>`
- discovers Diapix runs from `conversation_*.vhdr`
- prepares Diapix events
- validates BIDS write inputs
- skips actual BIDS writing because of `--dry-run`

## 8. Run the real BIDS conversion

Rerun without `--dry-run` (and use `--overwrite` if you need to replace existing outputs):

```bash
dcap bids-convert \
  --task diapix \
  --dataset-id Timone2025 \
  --source-root /data/raw/diapix \
  --bids-root /data/bids/diapix_bids \
  --subject sub-001 \
  --private-root "$DCAP_PRIVATE_ROOT" \
  --task-assets-dir /data/private_assets/diapix \
  --preload-raw \
  --line-freq-hz 50
```

Diapix-specific post-write behavior currently includes:

- writing iEEG BIDS files (via MNE-BIDS)
- writing cropped audio to `sub-XXX/audio/`
- copying optional `.asf` video to `sub-XXX/video/`
- writing trigger-alignment QC JSON (when available)

## 9. Build/update the public registry (shareable)

After conversion, build the public registry TSV from the private YAML metadata:

```bash
dcap registry build-public \
  --private-root "$DCAP_PRIVATE_ROOT" \
  --dataset-id Timone2025 \
  --out /Users/hiro/PycharmProjects/dcap/data/registry_public.tsv \
  --validate
```

Important:

- Current `build-public` uses private YAML (`subject_keys.yaml` + `subjects/*.yaml`) and does not scan the BIDS dataset.
- Make sure your `acquisitions` + `protocols` in subject YAML actually reflect the converted Diapix runs/sessions.

## 10. Validate registry metadata explicitly (recommended)

```bash
dcap registry validate \
  --public-registry /Users/hiro/PycharmProjects/dcap/data/registry_public.tsv \
  --private-root env
```

Use `--strict` if you want warnings to fail the command.

## 11. Update `registry_private.tsv` (optional, after `record_id` exists)

File:

- `$DCAP_PRIVATE_ROOT/registry_private.tsv`

This file is run-level review/QC metadata keyed by `record_id`, so it is usually easiest to populate **after** building the public registry (or after inspecting generated `record_id`s).

Columns are:

- `record_id`
- `dcap_id` (reviewer/annotator initials, not necessarily the subject clinical ID)
- `exclude_reason`
- `review_date`
- `notes`

## 12. (Optional) Build local registry view / sanitized availability exports

Local merged view (may include private overlays):

```bash
dcap registry view \
  --public-registry /Users/hiro/PycharmProjects/dcap/data/registry_public.tsv \
  --private-root env \
  --summary-only
```

Sanitized availability export (shareable):

```bash
dcap registry export-availability \
  --public-registry /Users/hiro/PycharmProjects/dcap/data/registry_public.tsv \
  --private-root env \
  --out /Users/hiro/PycharmProjects/dcap/data/availability_diapix.tsv
```

## Quick Checklist (Per New Diapix Subject)

- Add mapping in `$DCAP_PRIVATE_ROOT/subject_keys.yaml`
- Add/update `$DCAP_PRIVATE_ROOT/subjects/sub-XXX.yaml` (acquisitions + protocols include Diapix)
- Ensure `_TRIGGER_ID_MAP` contains the subject `dcap_id` and run trigger IDs
- Ensure raw folder exists at `<source-root>/<dcap_id>/` with `conversation_*.vhdr` and matching `.wav`
- Ensure task assets dir contains required Diapix files and onset rows
- Run `dcap bids-convert --dry-run`
- Run real `dcap bids-convert`
- Run `dcap registry build-public --validate`
- Update `$DCAP_PRIVATE_ROOT/registry_private.tsv` using generated `record_id`s (optional)

