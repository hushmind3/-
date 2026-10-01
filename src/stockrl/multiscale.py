"""Bounded completed-bar context for the live 1-minute trading model.

The large Transformer still reads its original 128-step 1-minute window.
These compact, point-in-time summaries give a small adapter longer views
without changing the existing checkpoint's backbone or symbol IDs.
"""
from __future__ import annotations

from collections import defaultdict, OrderedDict
from contextlib import closing
from datetime import datetime, timedelta, timezone
import math
import hashlib
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .paths import ensure_project_path
from .operating_rules import operating_rules


TIMEFRAME_NAMES = ("1m", "3m", "5m", "15m", "60m", "1d", "1w", "1mo")
TIMEFRAME_FEATURE_NAMES = ("ret1_pct", "ret_window_pct", "range_pct",
                           "log_volume_ratio", "freshness", "coverage")
LONG_CONTEXT_FEATURE_NAMES=("return_pct","volatility_pct","range_pct",
                            "log_volume_ratio","freshness","coverage")
_RULES=operating_rules()
LONG_CONTEXT_WINDOWS=tuple(int(x) for x in _RULES["daily_context_windows"])
LONG_CONTEXT_SPECS={**{f"{x}d":("1d",x) for x in LONG_CONTEXT_WINDOWS},
    f"{_RULES['weekly_context_bars']}w":("1w",int(_RULES['weekly_context_bars'])),
    f"{_RULES['monthly_context_bars']}mo":("1mo",int(_RULES['monthly_context_bars'])),
    **_RULES.get("extra_context_windows",{})}
LONG_CONTEXT_NAMES=tuple(LONG_CONTEXT_SPECS)
DAILY_ENCODER_BARS=int(_RULES.get("daily_encoder_bars",1300))
MULTISCALE_FEATURE_ORDER=tuple(f"{scale}:{feature}" for scale in TIMEFRAME_NAMES for feature in TIMEFRAME_FEATURE_NAMES)+tuple(
    f"{scale}:{feature}" for scale in LONG_CONTEXT_NAMES for feature in LONG_CONTEXT_FEATURE_NAMES)
BASE_MULTISCALE_FEATURE_COUNT=len(TIMEFRAME_NAMES)*len(TIMEFRAME_FEATURE_NAMES)
MULTISCALE_FEATURE_COUNT = len(MULTISCALE_FEATURE_ORDER)
_LOOKBACK_BARS = {"1m": 128, "3m": 64, "5m": 64, "15m": 32,
                  "60m": 4, "1d": 64, "1w": 52, "1mo": 24}
_MINUTE_NS = 60_000_000_000
_DAY_NS = 86_400_000_000_000
_DURATIONS_NS = {
    "1m": _MINUTE_NS, "3m": 3 * _MINUTE_NS, "5m": 5 * _MINUTE_NS,
    "15m": 15 * _MINUTE_NS, "60m": 60 * _MINUTE_NS,
    "1d": _DAY_NS, "1w": 7 * _DAY_NS, "1mo": 30 * _DAY_NS,
}


_DAILY_ROWS_CACHE=None
_DAILY_AGGREGATE_CACHE=OrderedDict()

class DailyBarStore:
    """Bounded multi-year completed daily OHLCV rows per symbol."""

    MAX_BARS_PER_SYMBOL = int(_RULES.get("daily_store_bars_per_symbol",1500))

    def __init__(self, path: str | Path):
        self.path = ensure_project_path(path, "multiscale daily bars")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute("PRAGMA auto_vacuum=INCREMENTAL")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA journal_size_limit=4194304")
        self.db.execute("PRAGMA max_page_count=4294967294")
        self.db.execute("""CREATE TABLE IF NOT EXISTS daily_bars (
            symbol TEXT NOT NULL, stamp_ns INTEGER NOT NULL,
            open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
            close REAL NOT NULL, volume REAL NOT NULL,
            PRIMARY KEY(symbol, stamp_ns))""")
        self.db.commit()

    def has_symbol(self, symbol: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM daily_bars WHERE symbol=? LIMIT 1", (symbol,)).fetchone() is not None

    def needs_long_history(self,symbol):
        count=self.db.execute("SELECT COUNT(*) FROM daily_bars WHERE symbol=?",(symbol,)).fetchone()[0]
        return count<DAILY_ENCODER_BARS

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
    global _DAILY_ROWS_CACHE
    if not path.is_file() or not symbols:
        return {}
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        signature=(str(path.resolve()),tuple((p.stat().st_size,p.stat().st_mtime_ns)
            if p.exists() else None for p in (path,Path(str(path)+"-wal"))))
        if _DAILY_ROWS_CACHE is not None and _DAILY_ROWS_CACHE[0]==signature:
            return {symbol:[row for row in _DAILY_ROWS_CACHE[1].get(symbol,[]) if row[0]<=latest_ns]
                for symbol in symbols}
        with closing(sqlite3.connect(uri,uri=True,timeout=5)) as db:
            rows=db.execute("SELECT symbol,stamp_ns,open,high,low,close,volume FROM daily_bars ORDER BY symbol,stamp_ns").fetchall()
    except (sqlite3.Error,OSError):
        return {}
    out=defaultdict(list)
    for symbol,stamp,op,high,low,close,volume in rows:
        out[symbol].append((int(stamp),float(op),float(high),float(low),float(close),float(volume)))
    _DAILY_ROWS_CACHE=(signature,out)
    return {symbol:[row for row in out.get(symbol,[]) if row[0]<=latest_ns] for symbol in symbols}


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
        self.summary_cache={}
        self.daily_token_cache={}
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
                if rows and period in ("1d","1w","1mo"):
                    key=(symbol,period,hashlib.sha256(np.asarray(rows,dtype=np.float64).tobytes()).digest())
                    aggregate=_DAILY_AGGREGATE_CACHE.get(key)
                    if aggregate is None:
                        bars=_aggregate(rows,period)
                        aggregate=(np.asarray([bar[0] for bar in bars],dtype=np.int64),bars)
                        _DAILY_AGGREGATE_CACHE[key]=aggregate
                        if len(_DAILY_AGGREGATE_CACHE)>1024:
                            _DAILY_AGGREGATE_CACHE.popitem(last=False)
                    else:
                        _DAILY_AGGREGATE_CACHE.move_to_end(key)
                    self.bars[(symbol,period)]=aggregate
                else:
                    bars=_aggregate(rows,period) if rows else []
                    self.bars[(symbol,period)]=(np.asarray([bar[0] for bar in bars],dtype=np.int64),bars)

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
            for k,name in enumerate(LONG_CONTEXT_NAMES):
                period,lookback=LONG_CONTEXT_SPECS[name]
                available,bars=self.bars[(symbol,period)]
                end=int(np.searchsorted(available,asof_ns,side="right"))
                if not end: continue
                key=(symbol,name,end)
                content=self.summary_cache.get(key)
                if content is None:
                    recent=bars[max(0,end-lookback):end]
                    closes=np.asarray([bar[4] for bar in recent],dtype=np.float64)
                    returns=np.diff(np.log(np.maximum(closes,1e-9)))
                    content=(100*(closes[-1]/max(closes[0],1e-9)-1),
                        100*float(returns.std()) if len(returns) else 0,
                        100*(max(bar[2] for bar in recent)-min(bar[3] for bar in recent))/max(closes[-1],1e-9),
                        math.log1p(max(0,bars[end-1][5]))-float(np.mean([math.log1p(max(0,bar[5])) for bar in recent])),
                        min(1,len(recent)/lookback))
                    self.summary_cache[key]=content
                freshness=math.exp(-max(0,asof_ns-int(available[end-1]))/_DURATIONS_NS[period])
                start=BASE_MULTISCALE_FEATURE_COUNT+k*width
                output[j,start:start+width]=np.clip((*content[:4],freshness,content[4]),-10,10)
        return output

    def daily_at(self,timestamp):
        """Every available completed day enters the small learnable encoder."""
        asof=int(timestamp.astype("datetime64[ns]").astype(np.int64))+_MINUTE_NS
        result=np.zeros((len(self.symbols),DAILY_ENCODER_BARS,6),dtype=np.float16)
        for j,symbol in enumerate(self.symbols):
            available,bars=self.bars[(symbol,"1d")]
            end=int(np.searchsorted(available,asof,side="right"))
            if not end:
                continue
            key=(symbol,end)
            tokens=self.daily_token_cache.get(key)
            if tokens is None:
                rows=bars[max(0,end-DAILY_ENCODER_BARS):end]
                prices=np.asarray([[row[1],row[2],row[3],row[4]] for row in rows],dtype=np.float64)
                previous=np.concatenate(([prices[0,3]],prices[:-1,3]))
                returns=np.clip(100*(prices/np.maximum(previous[:,None],1e-9)-1),-10,10)
                volume=np.log1p(np.maximum([row[5] for row in rows],0))
                delta=np.clip(volume-volume.mean(),-10,10)
                tokens=np.concatenate((returns,delta[:,None],np.ones((len(rows),1))),axis=1).astype(np.float16)
                self.daily_token_cache[key]=tokens
            result[j,-len(tokens):]=tokens
        return result

    def daily_status_at(self,timestamp):
        asof=int(timestamp.astype("datetime64[ns]").astype(np.int64))+_MINUTE_NS
        rows=[]
        for symbol in self.symbols:
            if symbol.startswith("__PAD__"):
                continue
            available,bars=self.bars[(symbol,"1d")]
            end=int(np.searchsorted(available,asof,side="right"))
            count=min(end,DAILY_ENCODER_BARS)
            years=((available[end-1]-available[max(0,end-DAILY_ENCODER_BARS)])/_DAY_NS/365.25 if count else 0)
            rows.append({"symbol":symbol,"bars":count,"years":float(years),"five_years_available":bool(years>=5)})
        return {"max_bars":DAILY_ENCODER_BARS,"symbols":rows,
            "five_year_symbols":sum(row["five_years_available"] for row in rows),
            "observed_symbols":len(rows),"missing_symbols":sum(row["bars"]==0 for row in rows)}
