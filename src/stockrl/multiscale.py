"""Bounded completed-bar context for the live 1-minute trading model.

The large Transformer still reads its original 128-step 1-minute window.
These compact, point-in-time summaries give a small adapter longer views
without changing the existing checkpoint's backbone or symbol IDs.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .paths import ensure_project_path


TIMEFRAME_NAMES = ("1m", "3m", "5m", "15m", "60m", "1d", "1w", "1mo")
TIMEFRAME_FEATURE_NAMES = ("ret1_pct", "ret_window_pct", "range_pct",
                           "log_volume_ratio", "freshness", "coverage")
MULTISCALE_FEATURE_COUNT = len(TIMEFRAME_NAMES) * len(TIMEFRAME_FEATURE_NAMES)
_LOOKBACK_BARS = {"1m": 128, "3m": 64, "5m": 64, "15m": 32,
                  "60m": 4, "1d": 64, "1w": 52, "1mo": 24}
_MINUTE_NS = 60_000_000_000
_DAY_NS = 86_400_000_000_000
_DURATIONS_NS = {
    "1m": _MINUTE_NS, "3m": 3 * _MINUTE_NS, "5m": 5 * _MINUTE_NS,
    "15m": 15 * _MINUTE_NS, "60m": 60 * _MINUTE_NS,
    "1d": _DAY_NS, "1w": 7 * _DAY_NS, "1mo": 30 * _DAY_NS,
}


class DailyBarStore:
    """At most 600 completed daily OHLCV rows per symbol in one SQLite file."""

    MAX_BARS_PER_SYMBOL = 600

    def __init__(self, path: str | Path):
        self.path = ensure_project_path(path, "multiscale daily bars")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute("PRAGMA auto_vacuum=INCREMENTAL")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA journal_size_limit=4194304")
        page_bytes=int(self.db.execute("PRAGMA page_size").fetchone()[0])
        self.db.execute(f"PRAGMA max_page_count={max(1,32*1024*1024//page_bytes)}")
        self.db.execute("""CREATE TABLE IF NOT EXISTS daily_bars (
            symbol TEXT NOT NULL, stamp_ns INTEGER NOT NULL,
            open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
            close REAL NOT NULL, volume REAL NOT NULL,
            PRIMARY KEY(symbol, stamp_ns))""")
        self.db.commit()

    def has_symbol(self, symbol: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM daily_bars WHERE symbol=? LIMIT 1", (symbol,)).fetchone() is not None

    def symbol_count(self) -> int:
        return int(self.db.execute("SELECT COUNT(DISTINCT symbol) FROM daily_bars").fetchone()[0])

    def upsert(self, rows: list[dict]) -> int:
        values = []
        for row in rows:
            try:
                stamp = int(pd.Timestamp(row["date"]).value)
                bar = tuple(float(row[key]) for key in ("open", "high", "low", "close", "volume"))
                if not row.get("symbol") or not all(math.isfinite(x) for x in bar) or bar[3] <= 0:
                    continue
                values.append((str(row["symbol"]), stamp, *bar))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        if not values:
            return 0
        with self.db:
            self.db.executemany("""INSERT INTO daily_bars VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(symbol,stamp_ns) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low,
                close=excluded.close, volume=excluded.volume""", values)
            for symbol in {value[0] for value in values}:
                boundary = self.db.execute(
                    "SELECT stamp_ns FROM daily_bars WHERE symbol=? ORDER BY stamp_ns DESC LIMIT 1 OFFSET ?",
                    (symbol, self.MAX_BARS_PER_SYMBOL - 1)).fetchone()
                if boundary:
                    self.db.execute("DELETE FROM daily_bars WHERE symbol=? AND stamp_ns<?",
                                    (symbol, int(boundary[0])))
        if self.path.stat().st_size > 16 * 1024 * 1024:
            self.db.execute("PRAGMA incremental_vacuum(128)")
        return len(values)

    def close(self) -> None:
        self.db.close()


def _load_daily(path: Path, symbols: set[str], latest_ns: int) -> dict[str, list[tuple]]:
    if not path.is_file() or not symbols:
        return {}
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=5) as db:
            rows = db.execute("""SELECT symbol,stamp_ns,open,high,low,close,volume
                FROM daily_bars WHERE stamp_ns<=? ORDER BY symbol,stamp_ns""",
                (latest_ns,)).fetchall()
    except (sqlite3.Error, OSError):
        return {}
    out: dict[str, list[tuple]] = defaultdict(list)
    for symbol, stamp, op, high, low, close, volume in rows:
        if symbol in symbols:
            out[symbol].append((int(stamp), float(op), float(high), float(low),
                                float(close), float(volume)))
    return out


def _aggregate(rows: list[tuple], period: str) -> list[tuple]:
    """Return (available_ns, open, high, low, close, volume) completed bars."""
    buckets: dict[object, list] = {}
    for stamp, op, high, low, close, volume in rows:
        if period in ("1m", "3m", "5m", "15m", "60m"):
            duration = _DURATIONS_NS[period]
            key = stamp // duration
            available = (key + 1) * duration
        else:
            date = datetime.fromtimestamp(stamp / 1e9, tz=timezone.utc).date()
            if period == "1d":
                key = date
                available = stamp + _DAY_NS
            elif period == "1w":
                iso = date.isocalendar()
                key = (iso.year, iso.week)
                next_monday = date + timedelta(days=7 - date.weekday())
                available = int(datetime.combine(
                    next_monday, datetime.min.time(), timezone.utc).timestamp() * 1e9)
                available = max(available, stamp + _DAY_NS)
            else:
                key = (date.year, date.month)
                next_month = (date.replace(day=1) + timedelta(days=32)).replace(day=1)
                available = int(datetime.combine(
                    next_month, datetime.min.time(), timezone.utc).timestamp() * 1e9)
                available = max(available, stamp + _DAY_NS)
        current = buckets.get(key)
        if current is None:
            buckets[key] = [available, op, high, low, close, volume]
        else:
            current[0] = max(current[0], available)
            current[2] = max(current[2], high)
            current[3] = min(current[3], low)
            current[4] = close
            current[5] += volume
    return [tuple(bucket) for bucket in sorted(buckets.values(), key=lambda x: x[0])]


class MultiscaleFeatures:
    """Point-in-time summaries. Never expose an unfinished higher-timeframe bar."""

    def __init__(self, frame: pd.DataFrame, symbols: list[str], daily_path: Path,
                 latest_timestamp: np.datetime64):
        self.symbols = symbols
        latest_ns = int(latest_timestamp.astype("datetime64[ns]").astype(np.int64))
        daily = _load_daily(daily_path, set(symbols), latest_ns)
        self.bars: dict[tuple[str, str], tuple[np.ndarray, list[tuple]]] = {}
        grouped = {str(symbol): group.sort_values("date")
                   for symbol, group in frame.groupby("symbol", sort=False)}
        for symbol in symbols:
            source = grouped.get(symbol)
            intraday = []
            if source is not None and not source.empty:
                for row in source.itertuples(index=False):
                    close=float(row.close)
                    if not math.isfinite(close) or close<=0:
                        continue
                    def clean(value, default):
                        number=float(value)
                        return number if math.isfinite(number) else default
                    stamp = int(pd.Timestamp(row.date).value)
                    intraday.append((stamp, clean(row.open,close), clean(row.high,close),
                                     clean(row.low,close), close, max(0.0,clean(row.volume,0.0))))
            for period in TIMEFRAME_NAMES:
                rows = intraday if period.endswith("m") and period != "1mo" else daily.get(symbol, [])
                bars = _aggregate(rows, period) if rows else []
                self.bars[(symbol, period)] = (
                    np.asarray([bar[0] for bar in bars], dtype=np.int64), bars)

    def at(self, timestamp: np.datetime64) -> np.ndarray:
        # The one-minute bar stamped T becomes available after its T+1m close.
        asof_ns = int(timestamp.astype("datetime64[ns]").astype(np.int64)) + _MINUTE_NS
        output = np.zeros((len(self.symbols), MULTISCALE_FEATURE_COUNT), dtype=np.float32)
        width = len(TIMEFRAME_FEATURE_NAMES)
        for j, symbol in enumerate(self.symbols):
            for scale, period in enumerate(TIMEFRAME_NAMES):
                available, bars = self.bars[(symbol, period)]
                end = int(np.searchsorted(available, asof_ns, side="right"))
                if not end:
                    continue
                lookback = _LOOKBACK_BARS[period]
                recent = bars[max(0, end - lookback - 1):end]
                newest = recent[-1]
                previous_close = recent[-2][4] if len(recent) > 1 else newest[4]
                window_close = recent[0][4] if len(recent) > lookback else newest[4]
                ret1 = 100.0 * (newest[4] / max(previous_close, 1e-9) - 1.0)
                ret_window = 100.0 * (newest[4] / max(window_close, 1e-9) - 1.0)
                spread = 100.0 * (newest[2] - newest[3]) / max(newest[4], 1e-9)
                prior_volume = (np.mean([math.log1p(max(0.0, bar[5])) for bar in recent[-21:-1]])
                                if len(recent) > 1 else 0.0)
                volume_delta = math.log1p(max(0.0, newest[5])) - prior_volume
                age = max(0, asof_ns - int(newest[0]))
                freshness = math.exp(-age / max(_DURATIONS_NS[period], 1))
                coverage = min(1.0, max(0, len(recent) - 1) / lookback)
                start = scale * width
                output[j, start:start + width] = np.clip(
                    (ret1, ret_window, spread, volume_delta, freshness, coverage), -10.0, 10.0)
        return output
