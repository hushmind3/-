"""Default machine-local storage paths for StockRL runtime state."""
from __future__ import annotations

import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKET = "korea"
DEFAULT_MODEL_DIR = Path.home() / "Desktop" / "모델"


def ensure_project_path(path: str | Path, label: str = "runtime") -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    root = PROJECT_ROOT.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"StockRL {label} must stay inside the project folder: {root}")
    return resolved


def validate_model_dir(path: str | Path | None = None) -> Path:
    expected = DEFAULT_MODEL_DIR.resolve()
    candidate = Path(path).expanduser() if path is not None else expected
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    if resolved != expected:
        raise ValueError(f"StockRL model checkpoints must stay in the model folder: {expected}")
    if not resolved.is_dir():
        raise FileNotFoundError(f"Model folder does not exist: {resolved}")
    return resolved


def default_runtime_dir() -> Path:
    configured = os.environ.get("STOCKRL_RUNTIME_DIR")
    market = os.environ.get("STOCKRL_MARKET", DEFAULT_MARKET).strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", market):
        raise ValueError("STOCKRL_MARKET must be a simple market name such as 'korea' or 'nasdaq'")
    candidate = (Path(configured).expanduser() if configured else
                 PROJECT_ROOT / "runtime" / "markets" / market)
    return ensure_project_path(candidate, "runtime")
