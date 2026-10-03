"""Shared experience schema, reward version, horizon and market snapshots."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import io
import re
import hashlib
import numpy as np
from .market_panel import GlobalMarketPanel

REWARD_VERSION="symbol_and_portfolio_v5"

REWARD_DEFINITION=("symbol_and_portfolio_v5: net-return percentage points; new paper decisions "
                   "use cumulative executed outcomes across the configured credit observations plus "
                   "a detached successor value, ending at account reset; legacy short outcomes remain "
                   "learnable; per-symbol contribution and whole-account result stay separate")

class IncrementalMarketCSV:
    """Read an append-only feed once, then parse only complete appended rows."""
    def __init__(self, path: str | Path, retain_timestamps: int = 4096):
        self.path=Path(path); self.retain_timestamps=retain_timestamps
        self.frame=None; self.offset=0; self.partial=b""; self.header=b""; self.fingerprint=None
        self.processed_through=None

    def _fingerprint(self):
        stat=self.path.stat()
        with self.path.open("rb") as stream:
            prefix=stream.read(2048)
        return stat.st_size,stat.st_mtime_ns,getattr(stat,"st_ino",0),hashlib.sha256(prefix).digest()

    def _trim(self, frame):
        import pandas as pd
        if frame.empty or "date" not in frame or "symbol" not in frame:
            return frame
        stamps=pd.to_datetime(frame["date"],utc=True,errors="coerce")
        unique=stamps.dropna().drop_duplicates().sort_values()
        if len(unique)<=self.retain_timestamps:
            return frame
        cutoff=unique.iloc[-self.retain_timestamps]
        # Keep all unprocessed bars, plus the model's input history. A restart
        # or a slow observer must not silently discard the next rolling windows.
        if self.processed_through is None:
            return frame
        processed=pd.Timestamp(self.processed_through)
        if processed.tzinfo is None:
            processed=processed.tz_localize("UTC")
        else:
            processed=processed.tz_convert("UTC")
        prior=unique.loc[unique<=processed]
        if len(prior):
            cutoff=min(cutoff,prior.iloc[-min(128,len(prior))])
        recent=frame.loc[stamps>=cutoff]
        carry=frame.groupby("symbol",sort=False).tail(128)
        carry=carry.loc[stamps.loc[carry.index]<cutoff]
        return pd.concat((carry,recent),ignore_index=True).sort_values(
            ["date","symbol"],kind="stable").reset_index(drop=True)

    def refresh(self):
        import pandas as pd
        signature=self._fingerprint()
        reset=(self.frame is None or signature[0]<self.offset or
               (self.fingerprint is not None and signature[2:]!=self.fingerprint[2:]))
        if reset:
            data=self.path.read_bytes()
            end=data.rfind(b"\n")+1
            complete=data[:end]
            self.partial=data[end:]
            self.header=complete.splitlines(keepends=True)[0] if complete else b""
            self.frame=pd.read_csv(io.BytesIO(complete)) if complete else pd.DataFrame()
            self.offset=len(data)
            self.frame=self._trim(self.frame)
        elif signature[0]>self.offset:
            with self.path.open("rb") as stream:
                stream.seek(self.offset); appended=stream.read()
            self.offset+=len(appended)
            combined=self.partial+appended
            end=combined.rfind(b"\n")+1
            complete=combined[:end]; self.partial=combined[end:]
            if complete:
                fresh=pd.read_csv(io.BytesIO(self.header+complete))
                self.frame=pd.concat((self.frame,fresh),ignore_index=True)
                self.frame=self.frame.drop_duplicates(["date","symbol"],keep="last")
                self.frame=self._trim(self.frame)
        self.fingerprint=signature
        return self.frame,signature

@dataclass
class Experience:
    features: np.ndarray
    symbol_ids: np.ndarray
    market_ids: np.ndarray
    asset_ids: np.ndarray
    valid_mask: np.ndarray
    symbol_index: int
    action: int
    reward: float
    timestamp: str
    source: str = "paper"
    regime: float = 0.0
    reward_version: str = "symbol_and_portfolio_v4"
    market_context: np.ndarray | None = None
    multiscale_state: np.ndarray | None = None
    portfolio_state: np.ndarray | None = None
    account_state: np.ndarray | None = None
    portfolio_reward: float | None = None
    portfolio_transition: bool = False
    portfolio_value_transition: bool = False
    forward_return: float | None = None
    behavior_log_prob: float | None = None
    trade_executed: bool = True
    origin_model: str = "champion"
    daily_history: np.ndarray | None = None
    credit_observations: int = 0
    bootstrap_window_key: str | None = None
    bootstrap_symbol_index: int | None = None
    bootstrap_discount: float = 0.0
    goal_state: np.ndarray | None = None
    goal_reward_points: float = 0.0
    portfolio_goal_reward_points: float = 0.0
    goal_terminal: bool = False
    goal_episode_id: str | None = None
    reward_settlement: str | None = None
    reward_end_timestamp: str | None = None
    reward_quote_timestamp: str | None = None

def parse_horizon(value: int | str) -> tuple[str, int]:
    """Return (bars|seconds, amount); durations resolve to the first observed bar at/after target."""
    if isinstance(value, int):
        if value < 1: raise ValueError("horizon bars must be >= 1")
        return "bars", value
    match=re.fullmatch(r"\s*(\d+)\s*(bars?|s|sec|seconds?|m|min|minutes?|h|hours?)\s*",str(value).lower())
    if not match: raise ValueError("horizon must be e.g. 30s, 1m, 5m, 1bar, or an integer bar count")
    amount=int(match.group(1)); unit=match.group(2)
    if amount<1: raise ValueError("horizon duration must be >= 1")
    if unit.startswith("bar"): return "bars",amount
    scale=1 if unit in ("s","sec","second","seconds") else 60 if unit in ("m","min","minute","minutes") else 3600
    return "seconds",amount*scale

class MarketObservation:
    """Bounded immutable market input for asynchronous portfolio validation."""
    window=GlobalMarketPanel.window

    def __init__(self,panel,index,length):
        start=max(0,index-length+1)
        self.dates=panel.dates[start:index+1].copy()
        self.symbols=list(panel.symbols)
        self.groups=dict(panel.groups)
        self.features=panel.features[start:index+1].copy()
        self.observed=panel.observed[start:index+1].copy()
        self.ever_observed=panel.observed[:index+1].any(axis=0)
        self.closes=panel.closes[start:index+1].copy()
        self.symbol_ids=panel.symbol_ids.copy()
        self.market_ids=panel.market_ids.copy()
        self.asset_ids=panel.asset_ids.copy()
        self.market_context=(panel.market_context[start:index+1].copy()
                             if panel.market_context is not None else None)
        self.multiscale=panel.multiscale_at(index).copy()
        history=(panel.daily_history_at(index) if hasattr(panel,"daily_history_at") else None)
        self.daily_history=history.copy() if history is not None else None

    def multiscale_at(self,index):
        if index!=len(self.dates)-1:
            raise ValueError("observation has only its captured multiscale state")
        return self.multiscale

    def daily_history_at(self,index):
        return self.daily_history
