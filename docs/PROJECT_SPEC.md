# dcap — Project Specification

## 1. Purpose & Scope

`dcap` is a reusable Python package providing **infrastructure-level tooling**
for working with irregular, clinically acquired sEEG datasets across multiple
tasks and studies.

The package is designed to:

- Standardize heterogeneous clinical recordings into **BIDS-compliant datasets**
- Validate data integrity, quality, and internal consistency
- Provide shared preprocessing, visualization, and analysis *primitives*
- Enable **reproducible, multi-project reuse** across independent scientific questions
- Enforce **strict separation between shareable code and sensitive metadata**

`dcap` is **not** a paper-specific analysis repository.  
Scientific interpretation, hypothesis-specific decisions, and figure generation
belong in downstream project repositories that *depend on* `dcap`.

---

## 2. Design Principles

### 2.1 Separation of concerns
- **Infrastructure vs science**: `dcap` provides tools, not conclusions.
- **Reusable vs project-specific**: no paper-specific logic in `dcap`.
- **Sensitive vs shareable**: identifiers and clinical metadata must never leak.

### 2.2 Explicitness over convenience
- No hidden subject lists or implicit assumptions.
- All dataset composition is driven by explicit metadata queries.
- All preprocessing and analysis steps must be configurable and logged.

### 2.3 Clinical reality–first
- Irregular participation is expected.
- Missing tasks, sessions, and runs are normal.
- Pipelines must gracefully handle partial data.

### 2.4 BIDS as a contract, not a straitjacket
- BIDS is the *external interface*.
- Internals may carry richer metadata, but only via controlled layers.

---

## 3. Non-Goals

The following are explicitly **out of scope** for `dcap`:

- Paper-specific statistical models
- Hypothesis-driven contrasts or ROIs
- Manuscript figures or tables
- Dataset hosting or data distribution
- End-user tutorials (belongs in README / docs later)

---

## 4. Sensitive Metadata Strategy

### 4.1 Problem statement
Some metadata required for analysis (patient identifiers, clinical notes,
acquisition quirks) **must not be committed to GitHub or shared publicly**.

### 4.2 Solution: three-layer metadata model

#### Layer 1 — Public / Shareable (version controlled)
- BIDS-mandated metadata (anonymized)
- Non-identifying task/session/run structure
- Code, schemas, validators

**Location**

dcap/
└── src/dcap/registry/schema/


#### Layer 2 — Private Local Metadata (never committed)
- Subject re-identification keys
- Clinical notes
- Site-specific acquisition details
- Internal QC decisions

**Location (example, user-defined)**


~/.dcap_private/
├── registry_private.parquet
├── subject_keys.yaml
└── notes/


This path is:
- User-configurable via environment variable (`DCAP_PRIVATE_ROOT`)
- Explicitly excluded via `.gitignore`
- Never imported implicitly

#### Layer 3 — Derived / Sanitized Views
- Filtered, anonymized metadata tables
- Task availability summaries
- Analysis-ready indices

Generated **programmatically** by joining Layer 1 + Layer 2 at runtime,
never stored in the repo.

---

## 5. Metadata Registry

### 5.1 Concept

`dcap` maintains a **dataset registry** that answers questions like:

- Which subjects did task X?
- Which runs passed QC?
- Which datasets can be combined safely?

The registry is **data-driven**, not hardcoded.

### 5.2 Canonical registry fields (conceptual)

| Field | Description |
|------|------------|
| subject | Anonymized subject ID |
| session | Session identifier |
| task | BIDS task name |
| run | Run index |
| bids_root | Path to BIDS dataset |
| qc_status | pass / fail / review |
| exclude_reason | Optional |
| notes | Optional (private layer only) |

### 5.3 API goals

Registry access must be via **query functions**, e.g.:

- `list_runs(task="conversation", qc="pass")`
- `subjects_with_tasks(["conversation", "localizer"])`
- `available_tasks(subject="sub-001")`

Snakemake, CLIs, and analyses **must not** hardcode subject lists.

---

## 6. Package Responsibilities (Tiered)

### Tier A — Data Plumbing (Most Stable)

#### A1. BIDS conversion
- Task-specific raw → BIDS converters
- sEEG-aware (contacts, montages, annotations)
- Deterministic and testable

#### A2. Validation & QC
- BIDS validation wrappers
- Signal-level QC (dropouts, flat channels, noise)
- Machine-readable QC outputs

#### A3. Metadata registry
- Schema definitions
- Loaders for public + private layers
- Query API

---

### Tier B — Preprocessing & Visualization

#### B1. Preprocessing primitives
- Referencing
- Filtering
- Epoching
- Artifact marking (not rejection decisions)

Pipelines must be:
- Parameterized
- Logged
- Re-runnable

#### B2. Visualization
- Standard QC plots
- Signal summaries
- Channel/task diagnostics

No publication figures here.

---

### Tier C — Analysis Primitives (Carefully Scoped)

`dcap` may include **analysis machinery**, but not analysis decisions.

Examples:
- TRF fitting functions
- Mutual information estimators
- Cluster-test wrappers
- ERP / rERP map builders

**Not included**:
- Predictor choices
- Time windows
- Statistical thresholds
- Interpretation logic

---

## 7. Repository Structure (Proposed)


```
dcap/
├── src/dcap/
│ ├── bids/
│ │ ├── conversation.py
│ │ ├── rest.py
│ │ └── ...
│ ├── qc/
│ ├── registry/
│ │ ├── schema/
│ │ ├── loader.py
│ │ └── queries.py
│ ├── preproc/
│ ├── viz/
│ ├── analysis/
│ ├── cli/
│ └── utils/
│
├── docs/
│ └── PROJECT_SPEC.md
│
├── tests/
│ ├── test_bids.py
│ ├── test_registry.py
│ └── ...
│
├── pyproject.toml
├── .gitignore
└── README.md (written later)


---

## 8. Interaction With Analysis Projects

Downstream project repositories:

- Depend on `dcap` as a package
- Provide:
  - Analysis configs
  - Workflow orchestration (e.g. Snakemake)
  - Figures, stats, paper logic

They **must not**:
- Reimplement BIDS logic
- Store sensitive metadata
- Duplicate preprocessing code
```

---

## 9. Reproducibility & Provenance

`dcap` should support:

- Versioned pipelines
- Config hashing
- Automatic recording of:
  - `dcap` version
  - Parameters
  - Input dataset identifiers

All derived data must be traceable.

---

## 10. Open Questions / TODOs

### Immediate
- [ ] Define registry schema (public vs private split)
- [ ] Decide on registry storage format (CSV vs Parquet)
- [ ] Define environment-variable conventions
- [ ] Minimal BIDS converter for one task

### Medium-term
- [ ] QC report standardization
- [ ] Preprocessing provenance logging
- [ ] Registry → Snakemake integration helpers

### Long-term
- [ ] Dataset “release” versioning
- [ ] Cross-dataset compatibility checks
- [ ] Optional anonymized registry export for publications

---

## 11. Success Criteria

`dcap` is successful if:

- New projects can start without touching raw data
- Irregular participation never breaks pipelines
- Sensitive metadata never appears in Git history
- Analyses are reproducible months later
- Multiple projects can coexist without forking logic

---