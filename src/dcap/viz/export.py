# =============================================================================
#                              Export Utilities
# =============================================================================
"""Export helpers for figures, tables, and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir = ensure_dir(out_dir)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.tsv", sep="\t", index=False)


def save_summary(summary: dict[str, Any], out_file: Path) -> None:
    ensure_dir(out_file.parent)
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def save_manifest(manifest: dict[str, Any], out_file: Path) -> None:
    ensure_dir(out_file.parent)
    out_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def export_report_html(title: str, figure_paths: list[Path], out_file: Path) -> None:
    """Very small HTML report wrapper (placeholder)."""
    ensure_dir(out_file.parent)
    items = "\n".join([f'<div><img src="figures/{p.name}" style="max-width: 100%;"/></div>' for p in figure_paths])
    html = f"""<!doctype html>
    <html>
    <head><meta charset="utf-8"><title>{title}</title></head>
    <body>
      <h1>{title}</h1>
      {items}
    </body>
    </html>
    """
    out_file.write_text(html, encoding="utf-8")
