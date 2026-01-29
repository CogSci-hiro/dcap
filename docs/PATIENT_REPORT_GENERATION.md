# PATIENT_REPORT_GENERATION.md

This document explains **how to generate a patient report** in the current DCAP skeleton (v2),
and how to extend it when real viz-ready tables become available.

> Clinical mode is **patient-specific by design**.  
> Research mode is currently **NotImplemented**.

---

## 1) Install / run the CLI

### Option A — editable install (recommended during development)
From the root of the extracted package:

```bash
pip install -e .
```

Then the `dcap` CLI should be available:

```bash
dcap --help
dcap report --help
dcap report patient --help
```

### Option B — run without installing (module mode)
From the root directory:

```bash
python -m dcap.cli.main report patient --subject sub-001 --mode clinical --out-dir ./out/sub-001
```

---

## 2) Generate a clinical patient report (current skeleton behavior)

### Minimal command
```bash
dcap report patient --subject sub-001 --mode clinical --out-dir ./out/sub-001
```

### What you should see in the output directory
```
out/sub-001/
  report.html
  summary.json
  manifest.json
  figures/
    header_placeholder.png
    patient_minutes_per_task.png
    naming_hg_full.png
    naming_hg_selected.png
    naming_hg_topography.png
    sorciere_trf_scores.png
    sorciere_trf_topography.png
    sorciere_trf_kernels.png
    diapix_trf_scores.png
    diapix_trf_topography.png
    diapix_trf_kernels.png
  tables/
    (empty in the skeleton unless you pass tables programmatically)
```

### Notes
- The current implementation generates **placeholder figures** (to validate wiring).
- As you add real data and plot logic, the same filenames/IDs remain stable
  thanks to `PatientClinicalReportSpec`.

---

## 3) Research mode (not implemented)

```bash
dcap report patient --subject sub-001 --mode research --out-dir ./out/sub-001
```

Expected behavior:
- The command raises `NotImplementedError`.

---

## 4) Programmatic usage (recommended for integration with pipeline code)

The CLI wrapper currently calls the report builder with **empty tables**.
In a real pipeline, you’ll construct **viz-ready DataFrames** and pass them in.

### Core entrypoint
- `dcap.viz.reports.patient.build_patient_report(...)`

Example:

```python
from pathlib import Path
import pandas as pd

from dcap.viz.reports.patient import build_patient_report

tables: dict[str, pd.DataFrame] = {
    # Minimal examples (replace with real tables):
    "clinical_subject_df": pd.DataFrame([
        {"subject": "sub-001", "age_bin": "30-34", "sex": "F"}
    ]),
    "preprocessing_common_df": pd.DataFrame([
        {"key": "ieeg_reference", "value": "bipolar"},
        {"key": "time_units", "value": "seconds"},
    ]),

    # Optional task tables:
    "sorciere_trf_scores_df": pd.DataFrame([
        {"run": "run-1", "predictor_set": "speech_env", "score": 0.12, "metric": "pearson_r"}
    ]),
}

build_patient_report(
    subject="sub-001",
    mode="clinical",
    tables=tables,
    out_dir=Path("./out/sub-001"),
    strict=False,
)
```

### What `strict=True` does
If you set `strict=True`, the builder will require certain expected inputs
(see the spec’s `required_table_keys`) and raise an error if they are missing.

---

## 5) The clinical report spec (what it controls)

### Spec location
- `dcap/viz/reports/patient_spec.py`

### Key object
- `PatientClinicalReportSpec.default()`

This spec defines:
- section IDs (e.g., `task_naming`, `task_sorciere`)
- figure IDs (stable filenames)
- table IDs
- a list of “expected” input table keys for validation

Example:

```python
from dcap.viz.reports.patient_spec import PatientClinicalReportSpec

spec = PatientClinicalReportSpec.default()
print(spec.all_figure_ids())
print(spec.all_table_ids())
print(spec.required_table_keys)
```

---

## 6) How to extend from placeholders to real figures

### Where to implement plotting logic
- `dcap/viz/reports/patient.py`

Replace placeholder figure creation with:
- real Matplotlib figure generation
- calls into existing module plotters:
  - `dcap.viz.validation.*`
  - `dcap.viz.overview.*`
  - `dcap.viz.minimal_analyses.*`
  - task-specific modules under `dcap.viz.tasks.*`

### Recommended approach
1. Build **viz-ready tables** upstream (pipeline code).
2. Write plot functions that accept those tables.
3. Keep **figure IDs stable** (use spec IDs as output names).
4. Add tests asserting:
   - `report.html` exists
   - expected figure files exist
   - required tables export properly

---

## 7) Known limitations of the skeleton (by design)
- The CLI does not yet build viz-ready tables.
- No BIDS scanning or file IO happens in `dcap.viz.reports.patient`.
- “research” mode intentionally not implemented.

---

## 8) Quick troubleshooting

### `dcap` command not found
- Install with `pip install -e .`
- Or use module mode: `python -m dcap.cli.main ...`

### Output directory is empty
- Ensure `--out-dir` exists or is creatable
- Check permissions
- Confirm the command finished without exceptions

### Want more content in the report?
- Use programmatic mode and pass DataFrames in `tables`
- Then gradually replace placeholders with real plotting functions

---

**End of document**
