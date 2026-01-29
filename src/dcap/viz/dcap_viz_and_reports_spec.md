# DCAP Visualization & Reporting Specification

This document summarizes the agreed design for the **dcap.viz** subpackage and the associated **reporting system**.  
It is a *developer-facing specification* (not a README) describing scope, structure, visualization content, and report definitions.

---

## 1. Scope and Philosophy

### Purpose of `dcap.viz`
- Dataset **overview, validation, and QC**, not scientific inference
- Emphasis on:
  - completeness
  - consistency
  - temporal correctness
  - pipeline health
- Minimal analyses allowed **only as sanity checks**
  - clinical overview (placeholder, QC-oriented)
  - TRF baseline models (known predictors only)

### Non-goals
- No hypothesis testing
- No group comparisons
- No publication-grade analysis figures

### Architectural principles
- `dcap.viz` is a **pure plotting/reporting library**
- **All orchestration happens in CLI**
  - `dcap/cli/main.py`
  - `dcap/cli/commands/...`
- Viz functions consume **tables + configs**, never raw filesystem logic

---

## 2. Package Structure

```
dcap/
  viz/
    api.py
    models.py
    style.py
    export.py

    overview/
      inventory.py
      timeline.py
      missingness.py

    validation/
      file_integrity.py
      sampling.py
      channels.py
      events.py
      annotations.py

    minimal_analyses/
      trf_baseline.py
      clinical.py

    tasks/
      <task_name>/
        overview.py
        validation.py
```

---

## 3. Overview Modules

### 3.1 `overview/inventory.py`
**Goal:** What exists in the dataset? How is it distributed?

Key visualizations:
- Modality & artifact counts (files + bytes)
- Runs per subject / session / task
- Subject × expected-artifact coverage heatmaps
- Storage footprint summaries
- File format / extension breakdown
- Version & provenance distributions
- Companion artifact consistency (A exists → B exists)
- Duplicate/collision detection (entity-level)
- Ingestion/conversion status overview

Key tables:
- Top missing artifacts
- Top duplication/collision cases
- Completeness scores per subject/task

---

### 3.2 `overview/timeline.py`
**Goal:** How does time behave in the dataset?

Key visualizations:
- Recording duration distributions
- Subject/session Gantt-style timelines
- Cumulative coverage curves
- Multimodal start/end offset distributions
- Duration mismatches across modalities
- Drift and alignment residuals (if available)
- Within-run gaps and segmentation
- Annotation coverage over time
- Event density over time (coarse)

Key tables:
- Runs with largest offsets
- Runs with gaps or truncation
- Duration mismatch outliers

---

### 3.3 `overview/missingness.py`
**Goal:** Where are the holes?

Key visualizations:
- Subject × expected-artifact missingness heatmaps
- Run-level completeness grids
- Task × artifact completion rates
- Companion dependency matrices
- Column-level schema completeness (TSV)
- JSON key presence summaries
- NaN-rate distributions
- Present-but-empty artifact detection
- Completeness scorecards
- Pareto charts of missingness causes

Key concepts:
- Central **Expected Artifacts Registry**
  - required / recommended / optional / conditional

---

## 4. Validation Modules

### 4.1 `validation/file_integrity.py`
**Goal:** Can the files be trusted?

Key visualizations:
- Read/parse status by file family
- Error code Pareto charts
- File size distributions & outliers
- Bytes-per-second vs duration sanity
- Duplicate detection via hashes
- Near-duplicate detection
- Sidecar ↔ data mismatches
- Channels.tsv ↔ raw mismatches
- Events outside recording bounds
- Read performance anomalies

Key outputs:
- integrity_summary.json
- worst_files.tsv
- duplicate_groups.tsv

---

### 4.2 `validation/sampling.py`
**Goal:** Is the timebase correct?

Key visualizations:
- Sampling frequency distributions
- Within-subject fs consistency
- Sidecar vs reader fs scatter
- Event time ↔ sample residual histograms
- Residuals vs time (drift)
- first_samp distributions
- Out-of-bounds events
- Duration-from-samples vs sidecar duration
- Mixed-rate modality pairs
- Drift slopes and step changes

Key outputs:
- Per-run sampling QC scores
- Worst-run summaries

---

### 4.3 `validation/channels.py`
**Goal:** Are channels consistent and well-described?

Key visualizations:
- Channel count distributions
- Channel type composition
- Naming convention violations
- Raw ↔ channels.tsv overlap
- Coordinate coverage completeness
- Bad channel prevalence
- Reference channel presence
- Channel set stability across runs
- Electrode group coverage

---

### 4.4 `validation/events.py`
**Goal:** Are events temporally and structurally sane?

Key visualizations:
- Event counts by type
- Event rate over time
- Ordering violations (onset/offset)
- Duplicate/near-duplicate events
- Inter-event interval distributions
- Out-of-bounds events
- Cross-modality alignment checks
- Segment overlap and gaps
- Vocabulary drift

---

### 4.5 `validation/annotations.py`
**Goal:** Are annotation labels, timing, and sources consistent?

Key visualizations:
- Annotated fraction of time
- Annotation density over time
- Label frequency and long-tail plots
- Label co-occurrence/exclusivity
- Vocabulary drift over time/source
- Source/annotator breakdown
- Agreement proxies (coverage/timing)
- Segment duration distributions
- Boundary quantization checks

---

## 5. Minimal Analyses

### 5.1 `minimal_analyses/trf_baseline.py`
**Purpose:** Sanity-check known predictors (not inference).

Predictor QC:
- Missingness heatmaps
- Variance and sparsity distributions
- Correlation/collinearity heatmaps

Model sanity:
- CV score distributions
- Observed vs null/permuted comparison
- Performance vs lag window

Weights sanity:
- TRF weights over lags
- Stability across folds
- Top-channel summaries (QC framing only)

---

### 5.2 `minimal_analyses/clinical.py`
**Purpose:** Clinical QC scaffolding (future-proof).

Sections:
- Clinical metadata completeness
- Sensitive-field leakage checks (private/public separation)
- Cohort overview (counts, ranges only)
- Electrode/implant summaries
- Hemisphere and region coverage
- Clinical event counts and timelines
- State/medication placeholders

Key concept:
- Clinical Field Registry
  - required level
  - sensitive flag
  - public/shareable flag

---

## 6. Report Types

### 6.1 Dataset Report
**Scope:** Entire dataset

Sections:
- Inventory & composition
- Timeline & coverage
- Missingness
- QC health snapshot

KPIs:
- # subjects/sessions/runs
- Total minutes per task/modality
- % required artifacts complete
- % QC failures
- Duplicate counts

---

### 6.2 Task Report
**Scope:** Single task

Sections:
- Task inventory
- Task timing & coverage
- Task validation
- Task-specific plug-ins

KPIs:
- Participants & runs
- Task completeness
- Task QC fail rate

---

### 6.3 Patient Report
**Scope:** Single subject (private by default)

Sections:
- Patient inventory
- Timing & synchronization
- Validation overview
- Optional minimal analyses

KPIs:
- Total minutes
- Completeness score
- FAIL/WARN counts
- Top issues

---

### 6.4 Quality Check (QC) Report
**Scope:** Pipeline health

Sections:
- Integrity gate
- Sampling gate
- Channels/events/annotations gates
- TRF baseline gate
- Final traffic-light summary

KPIs:
- Pass/fail per gate
- Failure counts by code
- TRF sanity metrics
- Recommended actions

---

## 7. Outputs and Formats

All reports export:
- `report.html` or `report.pdf`
- `figures/`
- `tables/`
- `summary.json`
- `manifest.json` (inputs + versions)

Severity language:
- OK / WARN / FAIL
- QC framing only

---

## 8. CLI Mapping

Suggested commands:
- `dcap report dataset`
- `dcap report task --task <name>`
- `dcap report patient --subject <id>`
- `dcap report qc`

CLI:
- loads standardized tables
- calls `dcap.viz.api`
- exports via `dcap.viz.export`

---

**End of specification**
