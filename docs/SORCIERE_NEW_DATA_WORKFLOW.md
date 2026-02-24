# Sorciere New-Data Intake Workflow (Current Repo Procedure)

This document describes the current end-to-end workflow in this repo for bringing newly acquired **Sorciere** (passive listening) data into `dcap`, using the current task-based BIDS conversion pipeline.

It reflects the current code behavior (including current assumptions/limitations), not only the long-term intended design.

## Scope

This covers:

- private subject mapping (`subject_keys.yaml`)
- Sorciere raw source layout expected by current task code
- Sorciere task assets (shared stimulus WAV)
- Sorciere BIDS conversion (`dcap bids-convert --task sorciere`)
- subject anatomy / recon export (`dcap bids-anat`)

## Important Ordering Note (Current Implementation)

As with Diapix, `dcap bids-convert` resolves the private `dcap_id` from `subject_keys.yaml` before it can find the raw source folder.

In practice, you must do a **minimal private metadata update first** (at least `subject_keys.yaml`).

## Directory / File Overview

### Private metadata root (local only)

Set a private root (never committed):

- `$DCAP_PRIVATE_ROOT/subject_keys.yaml`
- `$DCAP_PRIVATE_ROOT/subjects/sub-XXX.yaml` (recommended; needed for registry workflows)

Templates live in:

- `/Users/hiro/PycharmProjects/dcap/docs/registry/templates/subject_keys_TEMPLATE.yaml`
- `/Users/hiro/PycharmProjects/dcap/docs/registry/templates/subject_TEMPLATE.yaml`

### Raw Sorciere source layout expected by current task code

`dcap bids-convert` resolves `dcap_id` and then looks under:

- `<source-root>/<dcap_id>/`

Current Sorciere task behavior assumes **one recording per subject** and discovers a single raw file in that folder.

Supported raw formats:

- `.vhdr` (BrainVision)
- `.edf`
- `.fif`

If multiple supported files exist, the task tries to prefer names containing:

- `sorciere`
- `ispeech`
- `passive`

If multiple candidates still tie, conversion fails with an ambiguity error.

### Sorciere task assets directory (required)

Pass via `--task-assets-dir`.

Current Sorciere task factory expects:

- one shared stimulus `.wav` file (used as sync reference and copied to BIDS `stimuli/`)

If multiple WAV files exist, the task tries to pick one using filename preference:

- `sorciere` > `ispeech` > `passive`

If ambiguity remains, conversion fails and you should keep only the intended Sorciere WAV (or rename it more specifically).

## Current Sorciere Implementation Assumptions / Caveats

This is a first-pass Sorciere adapter. Current assumptions:

- one run only (BIDS `run` is omitted)
- one shared stimulus WAV (copied once into `bids_root/stimuli/`)
- trigger-train synchronization uses the shared trigger matching logic extracted from Diapix
- default trigger code is currently hardcoded to `10004`
- stimulus onset is currently assumed to start at the sync-aligned reference point (`stimulus_start_delay_s = 0.0`)

Not yet implemented in the Sorciere task:

- Sorciere-specific channel renaming / channel dropping based on `elec2tissues.mat`
- montage injection from `elec2atlas.mat`
- post-write patching of `channels.tsv` status/status_description
- subject-specific trigger-ID overrides (if needed)

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
    - bids_subject: sub-003
      dcap_id: Some-Clinical-ID
```

Notes:

- `dataset_id` must match the value you pass to `--dataset-id`
- `bids_subject` is the anonymized BIDS-facing ID
- `dcap_id` is used to locate `<source-root>/<dcap_id>`

## 3. (Recommended) Add/update subject private metadata YAML

File:

- `$DCAP_PRIVATE_ROOT/subjects/sub-003.yaml`

This is not required for Sorciere BIDS conversion itself, but it is recommended if you will later build/update the registry.

## 4. Prepare / verify Sorciere raw source files

Assume:

- `--source-root /data/raw/sorciere`
- `dcap_id = Some-Clinical-ID`

Then the converter expects the subject folder:

- `/data/raw/sorciere/Some-Clinical-ID/`

Inside it, ensure there is exactly one intended Sorciere raw recording file (or a clearly preferred one), for example:

- `Some-Clinical-ID_task-iSpeech_ieeg_raw.fif`

or

- `sorciere.vhdr` (+ matching `.eeg` / `.vmrk`)

## 5. Prepare / verify Sorciere task assets directory

Create or reuse a Sorciere assets folder (private/local), then confirm it contains the shared stimulus WAV:

- `sorciere.wav` (example name)

If there are multiple `.wav` files, either:

- keep only the Sorciere stimulus WAV in that folder, or
- rename it so it clearly contains `sorciere` (preferred by current auto-selection)

## 6. Run a dry-run BIDS conversion first

```bash
dcap bids-convert \
  --task sorciere \
  --dataset-id Timone2025 \
  --source-root /data/raw/sorciere \
  --bids-root /data/bids/sorciere_bids \
  --subject sub-003 \
  --private-root "$DCAP_PRIVATE_ROOT" \
  --task-assets-dir /data/private_assets/sorciere \
  --dry-run \
  --preload-raw \
  --line-freq-hz 50
```

What this does:

- resolves `dcap_id` from `subject_keys.yaml`
- reads raw data from `<source-root>/<dcap_id>`
- discovers a single Sorciere raw file
- prepares `stimulus_start` / `stimulus_end` events by trigger-train alignment
- validates BIDS write inputs
- skips actual BIDS writing because of `--dry-run`

## 7. Run the real BIDS conversion

Rerun without `--dry-run` (and use `--overwrite` if you need to replace existing outputs):

```bash
dcap bids-convert \
  --task sorciere \
  --dataset-id Timone2025 \
  --source-root /data/raw/sorciere \
  --bids-root /data/bids/sorciere_bids \
  --subject sub-003 \
  --private-root "$DCAP_PRIVATE_ROOT" \
  --task-assets-dir /data/private_assets/sorciere \
  --preload-raw \
  --line-freq-hz 50 \
  --overwrite
```

Current Sorciere post-write behavior includes:

- writing iEEG BIDS files (via MNE-BIDS)
- writing `events.tsv` with `stimulus_start` / `stimulus_end`
- copying the shared stimulus WAV once into:
  - `<bids-root>/stimuli/`

## 8. Validate timing on the first converted subject (recommended)

Because the current Sorciere adapter uses first-pass assumptions, validate on one subject before bulk conversion:

- confirm triggers were detected (conversion should fail if trigger code `10004` is absent)
- inspect `events.tsv` start/end timing against known Sorciere stimulus timing
- confirm stimulus duration matches expectations

If you observe a systematic onset shift, the next code change should be to add a Sorciere task config/asset parameter for `stimulus_start_delay_s`.

## 9. Write anatomy and copy recon derivatives (optional but recommended)

Anatomy is handled separately from task conversion using `dcap bids-anat`.

Example:

```bash
dcap bids-anat \
  --bids-root /data/bids/sorciere_bids \
  --bids-subject sub-003 \
  --subjects-dir /data/raw/sourcedata/subjects_dir \
  --dataset-id Timone2025 \
  --mapping-yaml "$DCAP_PRIVATE_ROOT/subject_keys.yaml" \
  --overwrite
```

Current `bids-anat` behavior:

- writes T1w from `<subjects-dir>/<original_id>/mri/T1.mgz` (if present)
- copies `elec_recon` to BIDS derivatives
- exports a normalized electrodes TSV from `elec2atlas.mat` (if present)

## 10. Convert additional subjects only after first-subject validation

After the first Sorciere subject looks correct, repeat Steps 2-9 for additional subjects.

If you find subject-specific trigger IDs or alignment offsets are required, capture those in code/assets before bulk conversion.

