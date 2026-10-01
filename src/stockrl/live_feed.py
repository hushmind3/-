"""Reconnectable public-market-data poller for the global CSV paper agent.

Yahoo's chart endpoint is an unofficial, best-effort 1m OHLCV source. Kraken's
public OHLC endpoint supplies exchange candles without an API key. Providers
are polled independently so one failing market never stops the others.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import queue
import shutil
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
from .paths import ensure_project_path
from .multiscale import DailyBarStore, TIMEFRAME_NAMES

import pandas as pd
import requests

FEED_COLUMNS = ["date", "symbol", "market", "asset_class", "open", "high", "low", "close", "volume",
                "bid", "ask", "bid_size", "ask_size", "buy_volume", "sell_volume", "trade_count",
                "implied_volatility", "open_interest", "yield_change", "days_to_expiry"]

_MARKET_PROVIDER_FACTORIES: dict[str, Any] = {}


def _market_session_open(item: dict, now_utc: datetime) -> bool:
    """Skip historical quote downloads while a market is closed.

    This is a polling schedule, not an exchange holiday calendar. Freshness is
    still decided from actual bar timestamps, never from this schedule alone.
    """
    market = str(item.get("market", ""))
    if item.get("poll_outside_session"):
        return True
    if market == "CRYPTO" or item.get("asset_class") == "crypto":
        return True
    zone = {
        "KRX": "Asia/Seoul", "KOSDAQ": "Asia/Seoul", "Japan": "Asia/Tokyo",
        "HongKong": "Asia/Hong_Kong", "Germany": "Europe/Berlin",
        "UK": "Europe/London",
    }.get(market, "America/New_York")
    local = now_utc.astimezone(ZoneInfo(zone))
    weekday, minute = local.weekday(), local.hour * 60 + local.minute
    if market in ("CME", "COMEX", "NYMEX"):
        return (weekday == 6 and minute >= 18 * 60) or (
            weekday in (0, 1, 2, 3) and not 17 * 60 <= minute < 18 * 60
        ) or (weekday == 4 and minute < 17 * 60)
    if market == "FX":
        return (weekday == 6 and minute >= 17 * 60) or weekday in (0, 1, 2, 3) or (
            weekday == 4 and minute < 17 * 60)
    if weekday >= 5:
        return False
    windows = {
        "KRX": (9 * 60, 15 * 60 + 30), "KOSDAQ": (9 * 60, 15 * 60 + 30),
        "US": (4 * 60, 20 * 60), "US_Treasury": (8 * 60, 17 * 60),
        "Japan": (9 * 60, 15 * 60 + 30), "HongKong": (9 * 60 + 30, 16 * 60),
        "Germany": (9 * 60, 17 * 60 + 30), "UK": (8 * 60, 16 * 60 + 30),
    }
    start, end = windows.get(market, (0, 24 * 60))
    return start <= minute <= end + 2


class MarketDataProviderAdapter:
    """Provider interface; implement fetch(instrument) -> normalized bar rows."""
    def __init__(self, collector: "LiveMarketCollector"):
        self.collector=collector

    def fetch(self, instrument: dict) -> list[dict]:
        raise NotImplementedError


def register_market_data_provider(name: str, adapter_factory) -> None:
    """Register an adapter factory before constructing the collector."""
    key=name.strip().lower()
    if not key or not callable(adapter_factory): raise ValueError("provider name and callable factory are required")
    _MARKET_PROVIDER_FACTORIES[key]=adapter_factory


class YahooChartAdapter(MarketDataProviderAdapter):
    def fetch(self,instrument): return self.collector._fetch_yahoo(instrument)


class KrakenOHLCAdapter(MarketDataProviderAdapter):
    def fetch(self,instrument): return self.collector._fetch_kraken(instrument)


register_market_data_provider("yahoo",YahooChartAdapter)
register_market_data_provider("kraken",KrakenOHLCAdapter)


def _utc_stamp(epoch: int | float) -> str:
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat(timespec="seconds")


class AppendOnlyMarketCSV:
    """Append bars once; compact only history acknowledged by the observer."""
    RETAIN_TIMESTAMPS = 512
    REFERENCE_CARRY_ROWS = 128
    DEDUPE_DAYS = 8
    # One rolling cache, compacted in place before it consumes the runtime budget.
    COMPACT_CSV_BYTES = 16 * 1024 * 1024
    COMPACT_INDEX_BYTES = 16 * 1024 * 1024

    def __init__(self, path: str | Path):
        self.path = ensure_project_path(path, "market data")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path.with_suffix(self.path.suffix + ".sqlite3")
        self.db = sqlite3.connect(self.db_path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS seen (symbol TEXT NOT NULL, stamp_ns INTEGER NOT NULL, PRIMARY KEY(symbol,stamp_ns))")
        self.db.execute("CREATE INDEX IF NOT EXISTS seen_stamp_idx ON seen(stamp_ns)")
        self.db.commit()
        self._next_csv_compaction = self.COMPACT_CSV_BYTES
        self._next_index_compaction = self.COMPACT_INDEX_BYTES
        self._seed_index()

    def _seed_index(self) -> None:
        # Rebuild/index existing CSV rows at startup. Handles a crash after the
        # CSV flush but before the SQLite key transaction was committed.
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        for chunk in pd.read_csv(self.path, usecols=["date", "symbol"], chunksize=100_000):
            stamps = pd.to_datetime(chunk["date"], utc=True, errors="coerce")
            rows = [(str(symbol), int(stamp.value)) for symbol, stamp in zip(chunk.symbol, stamps) if not pd.isna(stamp)]
            with self.db:
                self.db.executemany("INSERT OR IGNORE INTO seen VALUES (?,?)", rows)

    def append(self, rows: list[dict[str, Any]]) -> int:
        unique: list[dict[str, Any]] = []
        batch: set[tuple[str, int]] = set()
        for row in rows:
            row = {key: row.get(key) for key in FEED_COLUMNS}
            stamp = pd.to_datetime(row["date"], utc=True, errors="coerce")
            if pd.isna(stamp) or not row["symbol"]:
                continue
            row["date"] = stamp.isoformat()
            key = (str(row["symbol"]), int(stamp.value))
            if key in batch or self.db.execute("SELECT 1 FROM seen WHERE symbol=? AND stamp_ns=?", key).fetchone():
                continue
            batch.add(key)
            unique.append(row)
        if not unique:
            return 0
        fresh = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FEED_COLUMNS, extrasaction="ignore")
            if fresh:
                writer.writeheader()
            writer.writerows(unique)
            f.flush()
            os.fsync(f.fileno())
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO seen VALUES (?,?)",
                                [(str(r["symbol"]), int(pd.Timestamp(r["date"]).value)) for r in unique])
        self._compact_if_needed()
        return len(unique)

    def _compact_if_needed(self) -> None:
        wal=self.db_path.with_name(self.db_path.name+"-wal")
        csv_bytes=self.path.stat().st_size if self.path.exists() else 0
        index_bytes=sum(p.stat().st_size for p in (self.db_path,wal) if p.exists())
        if csv_bytes < self._next_csv_compaction and index_bytes < self._next_index_compaction:
            return
        if not self.path.exists() or csv_bytes == 0:
            return
        temporary=self.path.with_suffix(self.path.suffix+".compact.tmp")
        recent_temporary=self.path.with_suffix(self.path.suffix+".recent.tmp")
        try:
            latest_rows=self.db.execute(
                "SELECT DISTINCT stamp_ns FROM seen ORDER BY stamp_ns DESC LIMIT ?",
                (self.RETAIN_TIMESTAMPS,)).fetchall()
            if not latest_rows:
                return
            cutoff_ns=int(latest_rows[-1][0]) if len(latest_rows)>=self.RETAIN_TIMESTAMPS else None
            latest_ns=int(latest_rows[0][0])
            # Never compact unobserved bars. The saved cursor advances only
            # after the corresponding pending/replay input is durable.
            cursor_path=self.path.parent/"agent"/"live_cursor.json"
            try:
                cursor=json.loads(cursor_path.read_text(encoding="utf-8")).get("last_timestamp")
                cursor_ns=int(pd.to_datetime(cursor,utc=True).value) if cursor else None
            except (OSError,ValueError,TypeError,json.JSONDecodeError):
                cursor_ns=None
            if cursor_ns is None:
                return
            context_rows=self.db.execute(
                "SELECT DISTINCT stamp_ns FROM seen WHERE stamp_ns<=? ORDER BY stamp_ns DESC LIMIT 128",
                (cursor_ns,)).fetchall()
            if not context_rows:
                return
            protected_ns=int(context_rows[-1][0])
            cutoff_ns=min(cutoff_ns,protected_ns) if cutoff_ns is not None else None
            carry_by_symbol={}
            recent_counts={}
            wrote_recent=False
            for chunk in pd.read_csv(self.path,chunksize=100_000):
                stamps=pd.to_datetime(chunk["date"],utc=True,errors="coerce")
                if cutoff_ns is None:
                    recent_mask=stamps.notna()
                else:
                    stamp_ns=stamps.astype("int64",copy=False)
                    recent_mask=stamps.notna() & (stamp_ns>=cutoff_ns)
                    old_reference=chunk.loc[stamps.notna() & (stamp_ns<cutoff_ns)]
                    for symbol,rows in old_reference.groupby("symbol",sort=False):
                        prior=carry_by_symbol.get(symbol)
                        merged=rows if prior is None else pd.concat((prior,rows),ignore_index=True)
                        carry_by_symbol[symbol]=merged.sort_values("date",kind="stable").tail(
                            self.REFERENCE_CARRY_ROWS)
                recent=chunk.loc[recent_mask]
                if not recent.empty:
                    for symbol,count in recent.groupby("symbol").size().items():
                        recent_counts[symbol]=recent_counts.get(symbol,0)+int(count)
                    recent.to_csv(recent_temporary,index=False,columns=FEED_COLUMNS,
                                  mode="a" if wrote_recent else "w",header=not wrote_recent)
                    wrote_recent=True
            carry_rows=[rows.tail(max(0,self.REFERENCE_CARRY_ROWS-recent_counts.get(symbol,0)))
                        for symbol,rows in carry_by_symbol.items()]
            carry=(pd.concat(carry_rows,ignore_index=True)
                   if carry_rows else pd.DataFrame(columns=FEED_COLUMNS))
            if not carry.empty:
                carry=carry.sort_values(["date","symbol"],kind="stable")
            carry.to_csv(temporary,index=False,columns=FEED_COLUMNS)
            if wrote_recent:
                with recent_temporary.open("rb") as source, temporary.open("ab") as target:
                    source.readline()  # The final file already has its CSV header.
                    shutil.copyfileobj(source,target,1024*1024)
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary,self.path)
            keep_after=latest_ns-int(pd.Timedelta(days=self.DEDUPE_DAYS).value)
            with self.db:
                self.db.execute("DELETE FROM seen WHERE stamp_ns < ?",(keep_after,))
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.db.execute("VACUUM")
        except (OSError,sqlite3.Error,pd.errors.ParserError,ValueError) as exc:
            logging.getLogger(__name__).warning("Could not compact live market cache: %s",exc)
        finally:
            temporary.unlink(missing_ok=True)
            recent_temporary.unlink(missing_ok=True)
        if self.path.exists():
            current_csv=self.path.stat().st_size
            current_index=sum(p.stat().st_size for p in (self.db_path,wal) if p.exists())
            self._next_csv_compaction=self.COMPACT_CSV_BYTES
            self._next_index_compaction=self.COMPACT_INDEX_BYTES

    def close(self):
        self.db.close()


class LiveMarketCollector:
    def __init__(self, config_path: str | Path, output: str | Path, poll_seconds: float = 15.0,
                 timeout: float = 15.0, log_path: str | Path | None = None,
                 stop_file: str | Path | None = None):
        self.config_path = Path(config_path)
        self.output = ensure_project_path(output, "market data")
        self.stop_file = ensure_project_path(stop_file, "stop marker") if stop_file else None
        self.errors_path = (ensure_project_path(log_path, "runtime log") if log_path
                            else self.output.with_name("live_feed_errors.jsonl"))
        if self.errors_path.exists() and self.errors_path.stat().st_size >= 2*1024*1024:
            self.errors_path.write_text("",encoding="utf-8")
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.instruments = config["instruments"]
        configured_timeframes=tuple(config.get("decision_timeframes",TIMEFRAME_NAMES))
        if configured_timeframes!=TIMEFRAME_NAMES:
            raise ValueError(f"decision_timeframes must match the model feature order: {TIMEFRAME_NAMES}")
        self.daily_store=DailyBarStore(self.output.with_name("timeframes.sqlite3"))
        history_counts=dict(self.daily_store.db.execute(
            "SELECT symbol,COUNT(*) FROM daily_bars GROUP BY symbol"))
        # Fill missing/new instruments before refreshing already complete history.
        self.daily_instruments=sorted(self.instruments,key=lambda item:(
            history_counts.get(str(item["symbol"]),0)>0,
            history_counts.get(str(item["symbol"]),0)>=1300))
        self.daily_next_due: dict[str,float]={}
        self.daily_failures: dict[str,int]={}
        self.daily_pool=ThreadPoolExecutor(max_workers=5,thread_name_prefix="daily-context")
        self.daily_futures={}
        self.index = AppendOnlyMarketCSV(self.output)
        self.poll_seconds = max(1.0, poll_seconds)
        self.timeout = timeout
        self.stop = threading.Event()
        self.failures: dict[str, int] = {}
        self.next_due: dict[str, float] = {}
        self.latest_completed: dict[str, float] = {}
        self.http = requests.Session()
        self.http.headers.update({"User-Agent": "Mozilla/5.0 (compatible; GlobalPaperMarketObserver/1.0)"})
        self._http_local = threading.local()
        self.provider_adapters={key:factory(self) for key,factory in _MARKET_PROVIDER_FACTORIES.items()}
        self.metrics_path = self.output.with_name("live_feed_metrics.json")
        self.runtime_dir = self.output.parent.parent
        try:
            from .provider_credentials import read_settings
            broker_settings = read_settings(self.runtime_dir)
        except Exception:
            broker_settings = {"provider": "yahoo", "environment": "paper"}
        self.broker_provider = broker_settings.get("provider", "yahoo")
        self.broker_instruments = [x for x in self.instruments
            if self.output.parent.name == "live" and self.broker_provider == "kiwoom"
            and x.get("market") in (("KRX", "KOSDAQ") if broker_settings.get("environment") == "real" else ("KRX",))
            and x.get("asset_class") == "equity"]
        self.broker_symbols = {x["symbol"] for x in self.broker_instruments}
        self.broker_rows: queue.Queue = queue.Queue()
        self.broker_stop = threading.Event()
        self.broker_thread = None
        self.broker_status = {"source": self.broker_provider, "connected": False,
                              "symbols": len(self.broker_instruments), "last_message_utc": None,
                              "last_error": ""}

    def _get(self, url: str, params=None) -> dict:
        # requests.Session is not documented as thread-safe. Each bounded
        # collector worker therefore owns a small reusable connection pool.
        session = getattr(self._http_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(self.http.headers)
            self._http_local.session = session
        response = session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _bar_seconds(interval) -> int:
        if isinstance(interval,int): return max(60,interval*60)
        value=str(interval or "1m").lower()
        try:
            amount=int(value[:-1]); unit=value[-1]
            return amount*({"m":60,"h":3600,"d":86400}.get(unit,60))
        except (ValueError,IndexError):
            return 60

    def _fetch_yahoo(self, item: dict) -> list[dict]:
        symbol = item["provider_symbol"]
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(symbol, safe='^=.-')}"
        data = self._get(url, {"range": item.get("range", "1d"), "interval": item.get("interval", "1m"),
                               "includePrePost": "true"})
        result = (data.get("chart") or {}).get("result")
        if not result:
            raise RuntimeError((data.get("chart") or {}).get("error") or "Yahoo returned no chart data")
        chart = result[0]
        stamps = chart.get("timestamp") or []
        quote = chart.get("indicators", {}).get("quote", [{}])[0]
        rows = []
        bar_seconds=self._bar_seconds(item.get("interval","1m")); now=time.time()
        for i, stamp in enumerate(stamps):
            # Persist completed bars only. Chart APIs revise the current candle
            # in place; appending its early snapshot would freeze a partial close.
            if float(stamp)+bar_seconds>now: continue
            def val(name, default=0.0):
                values = quote.get(name) or []
                x = values[i] if i < len(values) else None
                return default if x is None else float(x)
            close = val("close", math.nan)
            if not math.isfinite(close) or close <= 0:
                continue
            rows.append(self._row(item, stamp, val("open", close), val("high", close), val("low", close), close,
                                  val("volume", 0.0)))
        if not rows:
            raise RuntimeError("Yahoo chart contained no usable OHLCV bars")
        return rows

    def _fetch_kraken(self, item: dict) -> list[dict]:
        interval = int(item.get("interval", 1))
        result = self._get("https://api.kraken.com/0/public/OHLC", {"pair": item["provider_symbol"], "interval": interval})
        if result.get("error"):
            raise RuntimeError("Kraken: " + "; ".join(result["error"]))
        key = next((k for k in result.get("result", {}) if k != "last"), None)
        if key is None:
            raise RuntimeError("Kraken returned no candle series")
        rows = []
        bar_seconds=self._bar_seconds(interval); now=time.time()
        for candle in result["result"][key]:
            stamp, op, hi, lo, close, _vwap, volume, trades = candle[:8]
            if float(stamp)+bar_seconds>now: continue
            rows.append(self._row(item, stamp, float(op), float(hi), float(lo), float(close), float(volume),
                                  trade_count=float(trades)))
        return rows

    @staticmethod
    def _row(item, stamp, op, hi, lo, close, volume, **extras):
        epoch = int(stamp) if isinstance(stamp, (int, float)) else int(pd.Timestamp(stamp).timestamp())
        row = {"date": _utc_stamp(epoch), "symbol": item["symbol"], "market": item["market"],
               "asset_class": item["asset_class"], "open": op, "high": hi, "low": lo, "close": close,
               "volume": volume}
        row.update(extras)
        return row

    def collect_once(self) -> int:
        rows = []
        now = time.monotonic()
        market_now = datetime.now(timezone.utc)
        due = []
        for item in self.instruments:
            if self.stop_file is not None and self.stop_file.exists():
                break
            name = item["symbol"]
            if name in self.broker_symbols:
                continue
            if not _market_session_open(item, market_now):
                continue
            if now < self.next_due.get(name, 0):
                continue
            due.append(item)

        def fetch(item):
            provider = item.get("provider", "yahoo")
            adapter = self.provider_adapters.get(str(provider).lower())
            if adapter is None:
                raise ValueError(f"Unsupported provider {provider!r}; register a MarketDataProviderAdapter")
            return adapter.fetch(item)

        # A broad universe should not turn each minute into a serial chain of
        # HTTP waits. Keep concurrency bounded to avoid overwhelming public APIs.
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(due)))) as pool:
            futures = {pool.submit(fetch, item): item for item in due}
            results = []
            for future, item in futures.items():
                try:
                    results.append((item, future.result(), None))
                except Exception as exc:
                    results.append((item, [], exc))

        for item, fetched, error in results:
            name = item["symbol"]
            if error is None:
                rows.extend(fetched)
                self.failures[name] = 0
                self.next_due[name] = now + max(self.poll_seconds, float(item.get("refresh_seconds", 60)))
            else:
                exc = error
                n = self.failures.get(name, 0) + 1
                self.failures[name] = n
                backoff = min(300.0, max(self.poll_seconds, 5.0) * (2 ** min(n - 1, 5)))
                self.next_due[name] = now + backoff
                event = {"time_utc": datetime.now(timezone.utc).isoformat(), "symbol": name,
                         "provider": item.get("provider", "yahoo"), "retry_seconds": backoff,
                         "error": f"{type(exc).__name__}: {exc}"}
                self.errors_path.parent.mkdir(parents=True, exist_ok=True)
                if self.errors_path.exists() and self.errors_path.stat().st_size >= 2*1024*1024:
                    self.errors_path.write_text("",encoding="utf-8")
                with self.errors_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
                logging.warning("%s failed (%s); retry in %.0fs", name, exc, backoff)
        # Daily history is a small bounded sidecar. Weekly/monthly context is
        # derived from these completed bars, never inferred from a 512-minute
        # live cache or from a still-forming higher-timeframe candle.
        daily_fetched=0
        for name,(item,future) in list(self.daily_futures.items()):
            if not future.done():
                continue
            try:
                daily_fetched+=self.daily_store.upsert(future.result())
                self.daily_failures[name]=0
                self.daily_next_due[name]=now+86_400.0
            except Exception as exc:
                failures=self.daily_failures.get(name,0)+1
                self.daily_failures[name]=failures
                self.daily_next_due[name]=now+min(3600.0,60.0*2**min(failures,6))
                logging.warning("daily history %s failed: %s",name,exc)
            del self.daily_futures[name]
        for item in self.daily_instruments:
            if len(self.daily_futures)>=5:
                break
            name=str(item["symbol"]);provider=str(item.get("provider","yahoo")).lower()
            if name in self.daily_futures or provider not in ("yahoo","kraken") or now<self.daily_next_due.get(name,0.0):
                continue
            daily_item=dict(item)
            if provider=="yahoo":
                daily_item["interval"]="1d"
                daily_item["range"]=("10y" if self.daily_store.needs_long_history(name) else "5d")
            else:
                daily_item["interval"]=1440
            future=self.daily_pool.submit(self.provider_adapters[provider].fetch,daily_item)
            self.daily_futures[name]=(item,future)
        while True:
            try:
                rows.append(self.broker_rows.get_nowait())
            except queue.Empty:
                break
        for row in rows:
            try:
                stamp = pd.Timestamp(row["date"]).timestamp()
                symbol = str(row["symbol"])
                self.latest_completed[symbol] = max(stamp, self.latest_completed.get(symbol, 0.0))
            except (KeyError, ValueError, TypeError):
                continue
        freshness_cutoff = time.time() - 300
        fresh_symbols = sorted(symbol for symbol, stamp in self.latest_completed.items()
                               if stamp >= freshness_cutoff)
        added = self.index.append(rows)
        metrics = {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "output": str(self.output),
                   "configured_instruments": len(self.instruments), "polled_instruments": len(due),
                   "fresh_symbols_5m": fresh_symbols,
                   "fetched_rows": len(rows), "appended_unique_rows": added,
                   "provider_failures": self.failures, "retrying_symbols": sum(v > 0 for v in self.failures.values()),
                   "decision_timeframes": list(TIMEFRAME_NAMES),
                   "daily_history_symbols": self.daily_store.symbol_count(),
                   "daily_history_rows_fetched": daily_fetched,
                   "daily_history_failures": self.daily_failures,
                   "broker_provider": self.broker_provider, "broker_connected": self.broker_status.get("connected", False),
                   "broker_symbols": self.broker_status.get("symbols", 0),
                   "broker_last_message_utc": self.broker_status.get("last_message_utc"),
                   "broker_nxt_subscribed": self.broker_status.get("nxt_subscribed", 0),
                   "broker_subscription_mode":self.broker_status.get("subscription_mode"),
                   "broker_subscription_count":self.broker_status.get("subscription_count",0),
                   "broker_subscription_limit":self.broker_status.get("subscription_limit",200),
                   "broker_nxt_active_symbols": self.broker_status.get("nxt_active_symbols", 0),
                   "broker_nxt_last_message_utc": self.broker_status.get("nxt_last_message_utc"),
                   "broker_nxt_ticks": self.broker_status.get("nxt_ticks", 0),
                   "broker_krx_subscribed": self.broker_status.get("krx_subscribed", 0),
                   "broker_krx_active_symbols": self.broker_status.get("krx_active_symbols", 0),
                   "broker_krx_last_message_utc": self.broker_status.get("krx_last_message_utc"),
                   "broker_krx_ticks": self.broker_status.get("krx_ticks", 0),
                   "broker_error": self.broker_status.get("last_error", "")}
        self.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logging.info("polled=%d fetched=%d appended=%d failures=%d", len(due), len(rows), added,
                     metrics["retrying_symbols"])
        return added

    def run(self, once: bool = False, max_cycles: int | None = None) -> None:
        cycles = 0
        try:
            if self.broker_instruments and not once:
                from .kiwoom_stream import KiwoomRealtimeStream
                stream = KiwoomRealtimeStream(self.runtime_dir, self.broker_instruments,
                    self.broker_stop, self.broker_rows, self.broker_status.update)
                self.broker_thread = threading.Thread(target=stream.run, daemon=True, name="kiwoom-market-ws")
                self.broker_thread.start()
            while not self.stop.is_set():
                if self.stop_file is not None and self.stop_file.exists():
                    break
                self.collect_once()
                cycles += 1
                if once or (max_cycles is not None and cycles >= max_cycles):
                    break
                self.stop.wait(self.poll_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self.broker_stop.set()
            if self.broker_thread is not None:
                self.broker_thread.join(timeout=5)
            tail = []
            while True:
                try:
                    tail.append(self.broker_rows.get_nowait())
                except queue.Empty:
                    break
            if tail:
                self.index.append(tail)
            self.index.close()
            self.daily_pool.shutdown(wait=True,cancel_futures=True)
            self.daily_store.close()
            self.http.close()
