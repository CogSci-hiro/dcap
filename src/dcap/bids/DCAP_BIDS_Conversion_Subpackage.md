# DCAP — BIDS Conversion Subpackage (Handoff Summary)

## Scope

This document summarizes the **DCAP BIDS conversion subpackage**.  
It is an internal developer-facing handoff describing architecture, responsibilities, and extension points.

This is **not** a user README.

---

## Design Goals

The BIDS conversion layer is designed to:

- Convert heterogeneous, messy clinical recordings into valid BIDS
- Enforce strict separation between:
  - **BIDS mechanics** (core)
  - **Experimental semantics** (tasks)
  - **Private identifiers** (registry / private YAML)
- Support multiple tasks without modifying core logic
- Prevent leakage of sensitive identifiers into BIDS outputs
- Provide safe dry-run and overwrite behavior

---

## High-Level Architecture

```
dcap/
└── bids/
    ├── core/          # Task-agnostic BIDS engine (frozen)
    └── tasks/         # Task adapters (Diapix, future tasks)
```

A **task registry layer** connects CLI → tasks without coupling core to task logic.

---

## Core Subpackage (`dcap/bids/core/`)

The core implements **how to write BIDS correctly**, and nothing else.

### Frozen Structure

```
dcap/bids/core/
├── config.py        # Task-agnostic conversion config
├── bids_paths.py    # BIDSPath construction + normalization
├── converter.py     # Core conversion engine
├── io.py            # Generic loaders / hygiene
├── transforms.py    # Safe raw transforms only
├── events.py        # PreparedEvents container
├── sync.py          # Generic temporal alignment utilities
├── sidecars.py      # Optional JSON helpers
├── writers.py       # Thin wrappers around mne-bids writers
├── anat.py          # Task-independent anatomy handling
```

This structure should only change if **BIDS or MNE-BIDS changes**.

---

## Core Responsibilities

### Conversion Orchestration (`converter.py`)

- Iterate over task-discovered recording units
- Normalize subject/session/run labels
- Build `BIDSPath`
- Call task hooks:
  - `discover`
  - `load_raw`
  - `prepare_events`
  - `post_write`
- Write data via `mne-bids`
- Enforce safety checks

**Core never:**
- inspects filenames
- interprets events
- touches audio/video
- handles private identifiers
- branches on task names

---

### Configuration (`config.py`)

`BidsCoreConfig` defines:

- `source_root`
- `bids_root`
- `subject` (BIDS target ID only)
- `session`
- `datatype`
- `overwrite`, `dry_run`
- `preload_raw`
- `line_freq`

No task-specific parameters belong here.

---

### BIDSPath Construction (`bids_paths.py`)

- Normalize `sub-`, `ses-`, `run-` labels
- Enforce required entities
- Single source of truth for BIDSPath creation

---

### Raw Transforms (`transforms.py`)

Reusable, safe operations only:

- set line frequency
- rename channels
- set channel types
- drop/pick channels defensively

No task heuristics, atlases, or NULL logic.

---

### Events Container (`events.py`)

Defines `PreparedEvents` with strict invariants:

- Either both `(events, event_id)` are present
- Or both are `None`

Core does not interpret event meaning.

---

### Synchronization (`sync.py`)

Generic interval-based alignment utilities.

Tasks extract onsets and pass arrays in — no assumptions about triggers or audio.

---

### Writers (`writers.py`)

Thin wrappers around `mne_bids.write_raw_bids`.

Centralizes validation and default behavior.

---

### Anatomy (`anat.py`)

Task-independent anatomy writing:

- T1w via `write_anat`
- Optional copying of recon outputs to `derivatives/`

Anatomy is **not task-specific**.

---

## Task Layer (`dcap/bids/tasks/`)

Tasks define **what the data mean**, not how BIDS works.

Each task lives under:

```
dcap/bids/tasks/<task_name>/
```

Example (Diapix):

```
dcap/bids/tasks/diapix/
├── task.py        # BidsTask implementation
├── events.py      # Task-specific event logic
├── audio.py       # Task-specific audio handling
├── heuristics.py  # Encoding fixes, quirks
├── sidecars.py    # Task JSON fields
└── models.py      # RecordingUnit definitions
```

---

## Task Interface (Contract)

All tasks must implement:

```python
class BidsTask(Protocol):
    name: str

    def discover(self, source_root) -> Sequence[RecordingUnit]: ...
    def load_raw(self, unit, preload: bool): ...
    def prepare_events(self, raw, unit, bids_path) -> PreparedEvents: ...
    def post_write(self, unit, bids_path) -> None: ...
```

This is the **only interface** between core and tasks.

---

## RecordingUnit

Tasks define their own `RecordingUnit` dataclasses describing:

- run
- session
- raw file paths
- auxiliary artifacts (audio/video/etc.)

Core treats these as opaque objects.

---

## Task Registry Layer (`dcap/bids/tasks/registry.py`)

The registry layer resolves:

- which task to run (`--task diapix`)
- private subject identifiers (`dcap_id`) via YAML
- task-specific assets

This prevents:

- core importing tasks
- CLI hardcoding task names
- leakage of private identifiers

---

## Private Subject Mapping (YAML)

Private YAML (outside Git) maps:

```
(dataset_id, bids_subject) -> dcap_id
```

Example:

```yaml
datasets:
  Timone2025:
    - bids_subject: sub-001
      dcap_id: Nic-Ele
```

Rules:

- `bids_subject` is always the BIDS target ID
- `dcap_id` is private and never written to BIDS
- Core never sees this mapping

---

## Task Assets Directory (`--task-assets-dir`)

A task-specific private directory containing auxiliary files needed for conversion.

Example (Diapix):

```
diapix_assets/
├── audio_onsets.tsv
├── beeps_*.wav
├── elec2atlas.mat
├── channels.tsv
└── notes.md
```

Not raw data.  
Not BIDS output.  
Explicitly passed.

---

## CLI Design

Generic conversion entry point:

```
dcap bids convert   --task diapix   --dataset-id Timone2025   --subject sub-001   --source-root /path/to/raw   --bids-root /path/to/bids   --private-root env   --task-assets-dir /path/to/assets
```

CLI always speaks **BIDS-facing identifiers**.

---

## Explicit Non-Goals

The BIDS conversion subpackage will never:

- parse clinical identifiers directly
- infer runs from filenames in core
- read onset TSVs in core
- crop audio in core
- branch on task names in core
- write files from tasks directly

If it feels task-like, it belongs in `tasks/<task>/`.

---

## Ready State

- Core conversion layer complete and frozen
- Task interface stable
- Private identifier handling isolated
- Multiple tasks supported without touching core

**Next work belongs exclusively in task implementations and tests.**
