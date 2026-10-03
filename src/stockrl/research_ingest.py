"""Adapters for public market research datasets into the fixed global model schema.

These adapters preserve data semantics (LOB snapshots versus human portfolio
actions) and never alter the production Transformer architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn

from .market_panel import GLOBAL_FEATURES


@dataclass
class TeacherRecord:
    source: str
    date: str
    symbol: str
    weights: np.ndarray
    behavior: str


class MacroHFTSubagent(nn.Module):
    """Exact small Q-subagent architecture used by the public MacroHFT weights."""
    def __init__(self, state_dim_1: int = 36, state_dim_2: int = 9,
                 action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim_1, hidden_dim)
        self.fc2 = nn.Linear(state_dim_2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.embedding = nn.Embedding(action_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2*hidden_dim))
        self.advantage = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*4),
                                       nn.GELU(approximate="tanh"), nn.Linear(hidden_dim*4, action_dim))
        self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*4),
                                   nn.GELU(approximate="tanh"), nn.Linear(hidden_dim*4, 1))
        self.register_buffer("max_punish", torch.tensor(1e12))

    def forward(self, single_state: torch.Tensor, trend_state: torch.Tensor,
                previous_action: torch.Tensor) -> torch.Tensor:
        c = self.embedding(previous_action) + self.fc2(trend_state)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm(self.fc1(single_state)) * (1 + scale) + shift
        advantage = self.advantage(x)
        return self.value(x) + advantage - advantage.mean(dim=-1, keepdim=True)


def macrophft_features(frame: pd.DataFrame, feature_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read MacroHFT's native ETHUSDT feature names in the checkpoint's order."""
    feature_dir = Path(feature_dir)
    single = np.load(feature_dir / "single_features.npy", allow_pickle=False).tolist()
    trend = np.load(feature_dir / "trend_features.npy", allow_pickle=False).tolist()
    missing = sorted(set(single + trend) - set(frame.columns))
    if missing:
        raise ValueError(f"MacroHFT frame missing checkpoint inputs: {missing}")
    x1 = frame[single].to_numpy(dtype=np.float32)
    x2 = frame[trend].to_numpy(dtype=np.float32)
    return np.nan_to_num(x1), np.nan_to_num(x2)


def _clip(x: float, scale: float = 1.0) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x / scale, -10.0, 10.0))


def trademaster_row_to_features(row: pd.Series) -> np.ndarray:
    """Map BTC 15-level snapshot and flow into the existing 17 features."""
    x = np.zeros(len(GLOBAL_FEATURES), np.float32)
    mid = max(float(row["midpoint"]), 1e-12)
    x[7] = _clip(float(row["spread"]) / mid * 1e4, 100.0)
    bids = np.array([float(row[f"bids_notional_{i}"]) for i in range(15)])
    asks = np.array([float(row[f"asks_notional_{i}"]) for i in range(15)])
    bid, ask = float(bids.sum()), float(asks.sum())
    x[8] = _clip((bid - ask) / max(bid + ask, 1e-12))
    buys, sells = float(row["buys"]), float(row["sells"])
    x[9] = _clip((buys - sells) / max(buys + sells, 1e-12))
    x[10] = _clip(np.log1p(max(buys + sells, 0.0)), 20.0)
    # Best 5-level distances encode book slope/shape within available channels.
    bd = np.array([float(row[f"bids_distance_{i}"]) for i in range(5)])
    ad = np.array([float(row[f"asks_distance_{i}"]) for i in range(5)])
    x[3] = _clip((ad.mean() - abs(bd.mean())) * 1e4, 20.0)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def trademaster_csv_to_global(path: str | Path, output: str | Path) -> dict:
    """Convert TradeMaster public BTC book snapshots to project OHLCV schema."""
    df = pd.read_csv(path)
    required = {"system_time", "midpoint", "spread", "buys", "sells"}
    if not required.issubset(df.columns):
        raise ValueError(f"TradeMaster columns missing: {sorted(required-set(df.columns))}")
    original_rows = len(df)
    df["date"] = pd.to_datetime(df.system_time, utc=True)
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    feats = np.stack([trademaster_row_to_features(row) for _, row in df.iterrows()])
    out = pd.DataFrame({
        "date": df.date.astype(str), "symbol": "BTCUSDT", "market": "CRYPTO",
        "asset_class": "SPOT", "open": df.midpoint, "high": df.midpoint,
        "low": df.midpoint, "close": df.midpoint,
        "volume": np.maximum(df.buys.to_numpy(float)+df.sells.to_numpy(float), 0.0),
    })
    for i, name in enumerate(GLOBAL_FEATURES):
        # Features derived from 15-level depth/flow override OHLCV placeholders.
        out[name] = feats[:, i]
    returns = df.midpoint.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["ret1"] = returns.clip(-10, 10)
    out["ret5"] = (df.midpoint.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0)).clip(-10,10)
    out["ret20"] = (df.midpoint.pct_change(20).replace([np.inf, -np.inf], np.nan).fillna(0.0)).clip(-10,10)
    out["volume_z"] = np.log1p(out.volume).diff().replace([np.inf,-np.inf],np.nan).fillna(0.0).clip(-10,10)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return {"rows": len(out), "symbols": 1, "duplicates_removed": int(original_rows-len(df)),
            "columns": list(out.columns), "output": str(output)}


def fi2010_file_to_arrays(path: str | Path, sample_stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Read FI-2010 feature-major text into ``(events, 40)`` and 5 labels.

    Verified FI-2010 layout is (149, events): rows 0:40 are the raw 10-level
    snapshot; rows 40:144 handcrafted fields; rows 144:149 labels for horizons
    10/20/30/50/100, encoded 1=down, 2=stationary, 3=up. We deliberately keep
    only the public raw 40 fields and fail closed on unexpected orientations.
    """
    arr = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if arr.shape[0] != 149 and arr.shape[1] == 149:
        arr = arr.T
    if arr.shape[0] != 149:
        raise ValueError(f"unexpected FI-2010 shape {arr.shape}; expected (149, events)")
    features = arr[:40].T
    labels = arr[144:149].T.astype(np.int64)
    observed = set(np.unique(labels).tolist())
    if not observed.issubset({1, 2, 3}):
        raise ValueError(f"unexpected FI-2010 label values {sorted(observed)}")
    if sample_stride > 1:
        features, labels = features[::sample_stride], labels[::sample_stride]
    return np.nan_to_num(features), labels


def fi2010_book_to_features(book40: np.ndarray) -> np.ndarray:
    """Summarize FI-2010 raw features in conventional ask/bid price/size order."""
    x = np.zeros(len(GLOBAL_FEATURES), np.float32)
    aprice, asize, bprice, bsize = book40[0:10], book40[10:20], book40[20:30], book40[30:40]
    x[7] = _clip(float(aprice[0]) - float(bprice[0]), 1.0)
    bs, ass = float(np.sum(bsize)), float(np.sum(asize))
    x[8] = _clip((bs-ass)/max(bs+ass, 1e-12))
    x[3] = _clip((float(aprice[-1])-float(aprice[0])) +
                 (float(bprice[0])-float(bprice[-1])), 2.0)
    # These are per-feature standardized inputs, not physical prices or sizes.
    return np.nan_to_num(x)
