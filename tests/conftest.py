from __future__ import annotations
from pathlib import Path
import pytest

@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def test_data_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "data"

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "visual: tests that optionally save numerical artifacts/figures for manual inspection",
    )

@pytest.fixture
def artifacts_dir(tmp_path):
    """Temporary directory for optional TRF artifacts.

    Enabled by:
        DCAP_TRF_SAVE_ARTIFACTS=1
        DCAP_TRF_SAVE_FIGURES=1
    """
    return tmp_path / "artifacts"
