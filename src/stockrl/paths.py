"""Default machine-local storage paths for StockRL runtime state."""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_NAME = "runtime-global-korea-live"


def default_runtime_dir() -> Path:
    configured = os.environ.get("STOCKRL_RUNTIME_DIR")
    candidate = (Path(configured).expanduser() if configured else
                 PROJECT_ROOT / RUNTIME_NAME)
    root = PROJECT_ROOT.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"StockRL runtime must stay inside the project folder: {root}")
    return resolved
