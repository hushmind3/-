"""Default machine-local storage paths for StockRL runtime state."""
from __future__ import annotations

import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_TRASH_DIR = PROJECT_ROOT.parent / (PROJECT_ROOT.name + "-휴지통")
DEFAULT_MARKET = "korea"
DEFAULT_MODEL_DIR = Path.home() / "Desktop" / "모델"
# All operating and original expert weights belong on the Desktop. Vendor
# code, config, native data and Python environments belong to this project.
EXPERT_ASSETS_DIR = PROJECT_ROOT / "artifacts" / "experts"
EXPERT_WEIGHTS_DIR = DEFAULT_MODEL_DIR / "experts"
TRADING_MOE_CHECKPOINT = DEFAULT_MODEL_DIR / "TradingMoE.pt"
GPU_OWNER_LOCK = PROJECT_ROOT / "runtime" / "gpu-owner.lock"


def expert_weight_path(path: str | Path) -> Path:
    """Resolve original weights separately from vendor code/config/data.

    Temporary source trees reconstructed from an embedded PT and caller-owned
    test directories keep their own paths; only project expert assets relocate.
    """
    path = Path(path)
    # Registries/checkpoints may have been produced under another Windows user.
    if path.is_absolute() and "Desktop" in path.parts and "모델" in path.parts:
        marker = path.parts.index("모델")
        return DEFAULT_MODEL_DIR.joinpath(*path.parts[marker + 1:])
    try:
        relative = path.resolve().relative_to(EXPERT_ASSETS_DIR.resolve())
    except ValueError:
        return path
    groups = {"checkpoints": "market", "stock-policies": "stock",
              "sources": "vendor", "fusion": "fusion"}
    if relative.parts[0] not in groups:
        return path
    weight = path.suffix.lower() in (".pt", ".pth", ".safetensors", ".bin")
    weight |= path.suffix.lower() == ".zip" and relative.parts[0] in ("checkpoints", "stock-policies")
    weight |= path.name == "best_model.pkl" and relative.parts[0] == "sources"
    return EXPERT_WEIGHTS_DIR / groups[relative.parts[0]] / Path(*relative.parts[1:]) if weight else path


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
