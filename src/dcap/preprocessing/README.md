
# DCAP sEEG preprocessing blocks — skeleton

This is a **logic-free API skeleton** for reusable preprocessing blocks:

- block 2: coordinates
- block 3: line noise (notch / zapline)
- block 4: high-pass OR gamma envelope path
- block 5: resample
- block 6: bad channel suggestion
- block 7: rereferencing (CAR / bipolar / Laplacian)

All blocks are passthrough right now, but they:
- return `BlockArtifact`
- append to `PreprocContext.proc_history`
- avoid any file I/O

## Run tests

```bash
python -m pip install -e .
pytest -q
```
