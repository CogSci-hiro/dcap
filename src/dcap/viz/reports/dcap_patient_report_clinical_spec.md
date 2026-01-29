# DCAP — Patient Report Specification (mode=clinical)

This document specifies the **patient-specific clinical report** for the DCAP package.
Clinical mode is **always generated per patient** (one subject at a time) and is intended to support
**clinical evaluation and QC-oriented interpretation** (not statistical inference).

---

## 1. Scope and Key Principles

### Scope
- **Unit of report:** one patient (`subject`) per report
- **Audience:** clinicians and data managers
- **Purpose:** provide a structured, readable overview of:
  - anonymized patient snapshot
  - available tasks and recordings
  - clinical-task results (Naming, Sorcière, Diapix)
  - QC context sufficient to interpret plots as sanity checks

### Non-goals
- No hypothesis testing, no p-values, no claims of significance
- No group-level comparisons
- No interpretation beyond descriptive QC framing

### Severity language (recommended)
- `OK / WARN / FAIL` at section and report level
- “sanity check”, “consistency”, “coverage”, “stability”, “alignment”

---

## 2. Invocation and Output Layout

### CLI invocation (proposed)
- `dcap report patient --subject <sub-XXX> --mode clinical --out-dir <OUT>`

### Standard outputs
- `report.html` (primary; PDF optional later)
- `figures/` (PNG or SVG)
- `tables/` (TSV)
- `summary.json` (KPIs and gate outcomes)
- `manifest.json` (inputs + versions + parameters)

### Deterministic naming
- Figures and tables must have stable, deterministic filenames (avoid random IDs).
- Include `subject`, `task`, `session`, `run` in drilldown outputs where applicable.

---

## 3. Report Structure (Clinical Mode)

### Section 0 — Cover / Header Ribbon
**Purpose:** immediate context and navigation.

Include:
- Subject ID (anonymized)
- Sessions included (IDs; dates optional and ideally coarsened)
- Tasks present (and which are “clinical tasks”)
- Report generation timestamp
- `dcap` version + pipeline version(s)
- “Clinical mode — patient-specific” label
- Brief caveat: descriptive/QC, not inference

---

### Section 1 — Patient Snapshot (Anonymized)
**Purpose:** quick clinical framing without identifiers.

**Patient data panel**
- Age (prefer binning if needed for anonymization; e.g., 30–34)
- Sex (standardized vocabulary)
- Optional (if available and safe): handedness, language, site code (non-identifying)

**Task participation**
- Table: tasks × runs × total minutes × usable minutes (if “usable” is tracked)
- Plot: minutes per task (bar)

**Sensitive field leakage check**
- Show a compact panel: “Forbidden identifiers detected: YES/NO”
- If YES: list fields detected (table), and mark report as `FAIL` at top-level gates

---

### Section 2 — Common Preprocessing Summary (Global)
**Purpose:** a single place to describe preprocessing common to all tasks.

Include:
- Input data scope:
  - sessions / runs included
  - modalities used (e.g., iEEG, audio, annotations)
- Channel handling:
  - inclusion/exclusion rules (bad channels, missing coords)
  - referencing scheme (if used)
- Filtering/bands (high-level, not task-specific):
  - e.g., notch, broad bandpass (task-specific filtering goes in task sections)
- Resampling policy (if any):
  - target sampling rates by modality
- Timebase conventions:
  - `first_samp` handling
  - units (seconds) and conversion conventions
- Artifact handling:
  - any global artifact masks or segment exclusions
- Versions and reproducibility:
  - pipeline version, parameter set IDs
  - hash/manifest fields if available

**Recommended outputs**
- Short bullet list in report body
- Table `tables/preprocessing_common.tsv` (key/value)

---

## 4. Clinical Task Sections (Plug-ins)

Clinical tasks currently defined:
- **Naming**
- **Sorcière**
- **Diapix**

Each task section follows a consistent internal template:
1) Task status & inputs
2) Task-specific preprocessing
3) Summary panels (key KPIs + 2–4 figures)
4) Drilldown panels (selected channels/electrodes)
5) 3D topography panels (when applicable)
6) Notes & QC caveats (short)

---

# 4A. Naming Task (High Gamma)

## Task status & inputs
Include:
- runs included / excluded (and why)
- number of channels used
- number of trials/events used (if applicable)
- data quality context:
  - usable time fraction
  - artifact rate proxy
  - number of bad channels excluded

**Recommended tables**
- `tables/naming_inputs.tsv` (run list, durations, usable minutes, n_channels)

## Task-specific preprocessing
Define explicitly (for report reproducibility):
- High gamma band (e.g., 70–150 Hz; configurable)
- Baseline window definition
- Time-lock definition (primary anchor)
- Metric definition:
  - % change, z-score, log ratio, etc.
- Aggregation:
  - mean/median across trials; robust option recommended

**Recommended table**
- `tables/naming_preprocessing.tsv` (key/value)

## Naming — Figures

### (1) High gamma activity per electrode (full)
- Electrode × time heatmap (trial-aggregated)
- Sorting:
  - by shaft/group, or peak latency/amplitude
- Overlay key time markers (stimulus/response windows)
- Add side strip summaries:
  - peak amplitude, peak latency, trials contributing

### (2) Selected electrodes (details)
Define selection rule (must be documented and deterministic):
- Top N electrodes by peak HG (within a specified window)
- Optional stability constraint across runs/folds
- Exclude bad/noisy electrodes

For each selected electrode:
- time course line plot (mean ± variability band)
- optional trial raster for context (if not too heavy)
- QC mini-box (n_trials, artifact fraction, SNR proxy)

### (3) 3D topography (summary)
- Map per-electrode scalar onto 3D brain surface(s)
- Scalar definition must be explicit:
  - peak HG in window or AUC in window
- Minimum views:
  - lateral + medial (or left + right)
- Mask electrodes with insufficient evidence (e.g., too few trials)

**Recommended tables**
- `tables/naming_hg_summary.tsv` (electrode, value, latency, region, hemisphere, inclusion flags)
- `tables/naming_top_electrodes.tsv` (top N list)

---

# 4B. Sorcière Task (TRF-based)

## Task status & inputs
Include:
- runs included / excluded (and why)
- channels included
- predictors used (names + versions)
- CV strategy summary
- data quality context strip (usable minutes, exclusions)

## Task-specific preprocessing
Document:
- modality alignment approach (audio↔iEEG)
- lag window(s) used
- regularization (if any)
- scoring metric (e.g., Pearson r, R²) and what it’s computed on
- null/permutation setup (sanity-only; number of perms if used)

## Sorcière — Figures & Tables

### (1) TRF score summary table
Rows: predictors or predictor sets (and/or runs)  
Columns (recommended):
- score (metric name)
- CV folds
- n_timepoints / n_samples used
- n_channels used
- null mean (optional) and delta vs null
- best lag / lag window
- status (OK/WARN/FAIL by thresholds)

### (2) TRF score plot
Recommended default plots:
- score distribution across runs/folds (box/violin)
- score trend by run/session (line/scatter)

### (3) TRF score topography in 3D
- map per-channel score (or delta vs null) to 3D surfaces
- mask unstable channels (low fold stability) or poor quality channels
- provide two views minimum

### (4) TRF kernel for selected channels
Selection rule (deterministic):
- top N channels by score or delta vs null
- plus fold-stability constraint

Kernel views:
- small multiples: kernel over lag for each selected channel
- mean kernel across selected channels with variability band
- optional: compare two canonical predictors (avoid clutter)

**Recommended outputs**
- `tables/sorciere_trf_scores.tsv`
- `tables/sorciere_trf_lag_curves.tsv`
- `tables/sorciere_trf_top_channels.tsv`
- `tables/sorciere_predictor_qc.tsv`

---

# 4C. Diapix Task (TRF-based)

Diapix follows the same TRF template as Sorcière with task-appropriate predictors and metadata.

## Diapix — Required elements
- TRF score summary table (same schema as Sorcière)
- TRF score distribution plot + trend plot
- TRF score 3D topography
- TRF kernels for selected channels

**Recommended outputs**
- `tables/diapix_trf_scores.tsv`
- `tables/diapix_trf_lag_curves.tsv`
- `tables/diapix_trf_top_channels.tsv`
- `tables/diapix_predictor_qc.tsv`

---

## 5. Cross-Task Improvements and Standards

### Selection criteria (centralized)
A single, centralized definition for:
- “selected electrodes” in Naming
- “selected channels” in TRF tasks
- stability criteria (across runs/folds)
- minimum usable data thresholds (minutes, trials)

Store in:
- `manifest.json` and/or `tables/selection_criteria.tsv`

### QC context strip per task (recommended)
Before each task’s main figures, show:
- channels included / excluded
- usable minutes / total minutes
- artifact rate proxy
- run exclusions list

### Skimmability
Each task should have:
- a short “Summary” subsection (2–4 figures)
- a “Details” subsection (selected channels/electrodes)

---

## 6. Summary JSON and Top-Level Gates

`summary.json` should include:
- subject, sessions included, tasks included
- top-level gate outcomes: identifiers leakage / integrity / sampling / channels / events / annotations / TRF sanity
- per-task KPIs (minutes, runs, usable fraction)
- lists of “top issues” by code

---

## 7. Required Input Tables (Recommended)

The clinical patient report should be constructible from viz-ready tables.

Minimum recommended inputs:
- `inventory_df` (files/runs presence and metadata)
- `artifact_presence_df` (expected artifacts present/empty)
- `file_integrity_df` (read status, sizes, errors)
- `sampling_df` (+ optional event mapping residuals)
- `channels_df` (+ coords coverage metrics)
- `events_df` (basic checks and boundaries)
- `annotations_df` (coverage, label stats)
- task-specific:
  - `naming_hg_df` / summaries
  - `trf_scores_df`, `trf_kernels_df`, `predictor_qc_df`

---

**End of clinical patient report specification**
