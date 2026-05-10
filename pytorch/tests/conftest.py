"""pytest fixtures and path setup for the PyTorch reference tests."""
import os
import sys
from pathlib import Path

import pytest

# Make the localvqe package importable without installing.
TESTS_DIR = Path(__file__).resolve().parent
PYTORCH_DIR = TESTS_DIR.parent
sys.path.insert(0, str(PYTORCH_DIR))


@pytest.fixture(scope="session")
def pytorch_dir() -> Path:
    return PYTORCH_DIR


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return TESTS_DIR / "fixtures"


@pytest.fixture(scope="session")
def checkpoint_dir() -> Path:
    """Where to look for downloaded .pt files.

    Resolves in order: $LOCALVQE_CKPT_DIR, then ./checkpoints/ relative
    to the pytorch/ directory. Tests that need a specific checkpoint
    skip themselves if it isn't present, so this never fails.
    """
    env = os.environ.get("LOCALVQE_CKPT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (PYTORCH_DIR / "checkpoints").resolve()
