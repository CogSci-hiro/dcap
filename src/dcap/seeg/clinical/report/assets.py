# =============================================================================
#                     ########################################
#                     #        CLINICAL REPORT ASSETS        #
#                     ########################################
# =============================================================================

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# A tiny 1x1 transparent PNG (safe placeholder)
_TRANSPARENT_1PX_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
    b"ASsJTYQAAAAASUVORK5CYII="
)


@dataclass(frozen=True, slots=True)
class ReportAssetDirs:
    """
    Standard asset directories inside an output report folder.

    Usage example
    -------------
        dirs = ReportAssetDirs.from_out_dir(Path("out"))
        dirs.ensure()
    """

    out_dir: Path
    figures_dir: Path
    tables_dir: Path

    @staticmethod
    def from_out_dir(out_dir: Path) -> "ReportAssetDirs":
        return ReportAssetDirs(
            out_dir=out_dir,
            figures_dir=out_dir / "figures",
            tables_dir=out_dir / "tables",
        )

    def ensure(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)


def write_placeholder_png(path: Path) -> None:
    """
    Write a small placeholder PNG to `path` if it doesn't exist.

    Parameters
    ----------
    path
        Output .png path.

    Usage example
    -------------
        write_placeholder_png(Path("out/figures/plot.png"))
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_bytes(_TRANSPARENT_1PX_PNG)


def relpath_for_embed(target: Path, *, base_dir: Path) -> str:
    """
    Compute a POSIX relative path for embedding in HTML/Markdown.

    Parameters
    ----------
    target
        The file being embedded.
    base_dir
        The directory of the report file.

    Returns
    -------
    rel
        POSIX relative path string.

    Usage example
    -------------
        rel = relpath_for_embed(Path("out/figures/a.png"), base_dir=Path("out"))
    """
    rel = target.relative_to(base_dir)
    return rel.as_posix()
