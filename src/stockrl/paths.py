"""Default machine-local storage paths for StockRL runtime state."""
from __future__ import annotations

import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKET = "korea"


def default_runtime_dir() -> Path:
    configured = os.environ.get("STOCKRL_RUNTIME_DIR")
    market = os.environ.get("STOCKRL_MARKET", DEFAULT_MARKET).strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", market):
        raise ValueError("STOCKRL_MARKET must be a simple market name such as 'korea' or 'nasdaq'")
    candidate = (Path(configured).expanduser() if configured else
                 PROJECT_ROOT / "runtime" / "markets" / market)
    root = PROJECT_ROOT.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"StockRL runtime must stay inside the project folder: {root}")
    return resolved
