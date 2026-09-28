"""Default machine-local storage paths for StockRL runtime state."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def default_runtime_dir() -> Path:
    configured = os.environ.get("STOCKRL_RUNTIME_DIR")
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "StockRL" / "runtime-global-korea-live"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "StockRL" / "runtime-global-korea-live"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "StockRL" / "runtime-global-korea-live"
