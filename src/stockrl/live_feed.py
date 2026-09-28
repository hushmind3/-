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
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

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
    """Append bars exactly once across process restarts, with a durable key index."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path.with_suffix(self.path.suffix + ".sqlite3")
        self.db = sqlite3.connect(self.db_path, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS seen (symbol TEXT NOT NULL, stamp_ns INTEGER NOT NULL, PRIMARY KEY(symbol,stamp_ns))")
        self.db.commit()
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
        return len(unique)

    def close(self):
        self.db.close()


class LiveMarketCollector:
    def __init__(self, config_path: str | Path, output: str | Path, poll_seconds: float = 15.0,
                 timeout: float = 15.0, log_path: str | Path | None = None,
                 stop_file: str | Path | None = None):
        self.config_path = Path(config_path)
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.instruments = config["instruments"]
        self.output = Path(output)
        self.index = AppendOnlyMarketCSV(output)
        self.poll_seconds = max(1.0, poll_seconds)
        self.timeout = timeout
        self.stop = threading.Event()
        self.stop_file=Path(stop_file) if stop_file else None
        self.errors_path = Path(log_path) if log_path else self.output.with_name("live_feed_errors.jsonl")
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
                with self.errors_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
                logging.warning("%s failed (%s); retry in %.0fs", name, exc, backoff)
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
                   "broker_provider": self.broker_provider, "broker_connected": self.broker_status.get("connected", False),
                   "broker_symbols": self.broker_status.get("symbols", 0),
                   "broker_last_message_utc": self.broker_status.get("last_message_utc"),
                   "broker_nxt_subscribed": self.broker_status.get("nxt_subscribed", 0),
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
            self.http.close()
