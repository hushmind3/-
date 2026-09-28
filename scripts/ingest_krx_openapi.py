"""Resumable downloader for KRX official daily market APIs.

Set KRX_API_KEY to an approved KRX Open API key. Each API is queried by
business date; raw replies are cached so restarts do not repeat completed days.
The normalized OHLCV output follows CommonMarketTrainingLoader's bar schema.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://data-dbg.krx.co.kr/svc/apis"
CATALOG: dict[str, tuple[str, str, str]] = {
    # key: (URL path, asset class, market label)
    "kospi": ("sto/stk_bydd_trd", "equity", "KRX"),
    "kosdaq": ("sto/ksq_bydd_trd", "equity", "KRX"),
    "konex": ("sto/knx_bydd_trd", "equity", "KRX"),
    "etf": ("etp/etf_bydd_trd", "etf", "KRX"),
    "etn": ("etp/etn_bydd_trd", "etn", "KRX"),
    "kospi_index": ("idx/kospi_dd_trd", "index", "KRX"),
    "kosdaq_index": ("idx/kosdaq_dd_trd", "index", "KRX"),
    "krx_index": ("idx/krx_dd_trd", "index", "KRX"),
    "futures": ("drv/fut_bydd_trd", "future", "KRX"),
    "kospi_stock_futures": ("drv/stk_fut_bydd_trd", "future", "KRX"),
    "kosdaq_stock_futures": ("drv/ksq_fut_bydd_trd", "future", "KRX"),
    "options": ("drv/opt_bydd_trd", "option", "KRX"),
    "kospi_stock_options": ("drv/stk_opt_bydd_trd", "option", "KRX"),
    "kosdaq_stock_options": ("drv/ksq_opt_bydd_trd", "option", "KRX"),
}

NUMERIC = ("TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC", "TDD_CLSPRC",
           "ACC_TRDVOL", "ACC_TRDVAL", "MKTCAP", "NAV", "CMPPREVDD_PRC",
           "FLUC_RT", "SETL_PRC", "TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC")


def ymd(s: str) -> date:
    return datetime.strptime(s.replace("-", ""), "%Y%m%d").date()


def number(v: Any) -> float:
    if v is None:
        return 0.0
    s = str(v).strip().replace(",", "").replace("%", "")
    if s in ("", "-", "--"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def data_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    # Current KRX responses use OutBlock_1; accepting a generic items/list
    # keeps the adapter forward compatible with response envelope changes.
    for key in ("OutBlock_1", "OutBlock1", "items", "data", "list"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []


def normalize_row(row: dict[str, Any], asset: str, default_market: str) -> dict[str, Any] | None:
    code = str(row.get("ISU_CD") or row.get("IDX_NM") or row.get("ISU_NM") or "").strip()
    close = number(row.get("TDD_CLSPRC") or row.get("CLSPRC") or row.get("SETL_PRC"))
    if not code or close <= 0:
        return None
    ymd_value = str(row.get("BAS_DD") or "")
    if not re.fullmatch(r"\d{8}", ymd_value):
        return None
    market = str(row.get("MKT_NM") or default_market).strip()
    symbol = code.zfill(6) if asset in ("equity", "etf", "etn") and code.isdigit() else code
    return {
        "date": datetime.strptime(ymd_value, "%Y%m%d").replace(tzinfo=timezone.utc).isoformat(),
        "symbol": symbol,
        "market": "KR",
        "asset_class": asset,
        "open": number(row.get("TDD_OPNPRC") or row.get("OPNPRC") or close),
        "high": number(row.get("TDD_HGPRC") or row.get("HGPRC") or close),
        "low": number(row.get("TDD_LWPRC") or row.get("LWPRC") or close),
        "close": close,
        "volume": number(row.get("ACC_TRDVOL") or row.get("TRDVOL") or row.get("ACC_TRDVOL")),
        "bid": 0.0,
        "ask": 0.0,
        "buy_volume": 0.0,
        "sell_volume": 0.0,
        "trade_count": 0.0,
        "currency": "KRW",
        "trade_value_krw": number(row.get("ACC_TRDVAL") or row.get("TRDVAL")),
        "source": "krx_openapi",
        "source_name": str(row.get("ISU_NM") or row.get("IDX_NM") or ""),
    }


class KRXClient:
    def __init__(self, api_key: str, raw_dir: Path, timeout: float = 30.0):
        if not api_key:
            raise ValueError("Set KRX_API_KEY to an approved KRX Open API key")
        self.api_key = api_key
        self.raw_dir = raw_dir
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"AUTH_KEY": api_key, "Accept": "application/json"})

    def get_day(self, endpoint: str, day: date) -> dict[str, Any]:
        path, _, _ = CATALOG[endpoint]
        url = f"{BASE}/{path}.json"
        response = self.session.get(url, params={"basDd": day.strftime("%Y%m%d")}, timeout=self.timeout)
        if response.status_code in (401, 403):
            raise PermissionError(f"KRX rejected the key ({response.status_code}); check key approval and endpoint subscription")
        response.raise_for_status()
        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(f"KRX returned non-JSON for {endpoint}/{day}: {response.text[:160]}") from exc
        if not isinstance(body, dict):
            raise RuntimeError(f"Unexpected KRX response for {endpoint}/{day}")
        err = str(body.get("RESULT", {}).get("MSG", "")) if isinstance(body.get("RESULT"), dict) else ""
        code = str(body.get("RESULT", {}).get("CODE", "")) if isinstance(body.get("RESULT"), dict) else ""
        if code and code not in ("OK", "INFO-000"):
            raise RuntimeError(f"KRX API {endpoint} error {code}: {err}")
        return body

    def sync(self, endpoints: list[str], start: date, end: date, output: Path,
             delay: float = 0.15, max_days: int = 0) -> dict[str, Any]:
        raw_rows = 0
        fetched_days = 0
        records: dict[tuple[str, str], dict[str, Any]] = {}
        for endpoint in endpoints:
            path, asset, market = CATALOG[endpoint]
            cache = self.raw_dir / endpoint
            cache.mkdir(parents=True, exist_ok=True)
            current = start
            n_days = 0
            while current <= end:
                if current.weekday() < 5:
                    file = cache / f"{current:%Y%m%d}.json"
                    if file.exists():
                        payload = json.loads(file.read_text(encoding="utf-8"))
                    else:
                        payload = self.get_day(endpoint, current)
                        tmp = file.with_suffix(".tmp")
                        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                        tmp.replace(file)
                        fetched_days += 1
                        time.sleep(max(0.0, delay))
                    rows = data_rows(payload)
                    raw_rows += len(rows)
                    for row in rows:
                        normalized = normalize_row(row, asset, market)
                        if normalized:
                            records[(normalized["date"], normalized["symbol"])] = normalized
                    n_days += 1
                    if max_days and n_days >= max_days:
                        break
                current += timedelta(days=1)
            print(f"krx endpoint={endpoint} calendar_days={n_days} fetched={fetched_days} rows_total={len(records):,}", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = ["date", "symbol", "market", "asset_class", "open", "high", "low", "close", "volume",
                  "bid", "ask", "buy_volume", "sell_volume", "trade_count", "currency", "trade_value_krw",
                  "source", "source_name"]
        ordered = sorted(records.values(), key=lambda x: (x["date"], x["market"], x["asset_class"], x["symbol"]))
        tmp_out = output.with_suffix(output.suffix + ".tmp")
        with tmp_out.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(ordered)
        tmp_out.replace(output)
        summary = {"provider": "KRX Open API", "endpoints": endpoints, "start": str(start), "end": str(end),
                   "calendar_days_fetched": fetched_days, "raw_rows_seen": raw_rows,
                   "normalized_rows": len(ordered), "unique_symbols": len({x['symbol'] for x in ordered}),
                   "output": str(output), "raw_cache": str(self.raw_dir),
                   "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
        summary_path = output.with_suffix(".manifest.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2020-01-01", help="inclusive YYYY-MM-DD")
    ap.add_argument("--end", default=date.today().isoformat(), help="inclusive YYYY-MM-DD")
    ap.add_argument("--endpoints", nargs="+", choices=sorted(CATALOG), default=["kospi", "kosdaq", "etf", "kospi_index", "kosdaq_index", "futures", "options"])
    ap.add_argument("--output", type=Path, default=ROOT / "data/external_sources/krx/krx_daily.csv")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data/external_sources/krx/raw")
    ap.add_argument("--delay", type=float, default=0.15)
    ap.add_argument("--max-days", type=int, default=0, help="per-endpoint smoke limit; 0 means full requested range")
    ap.add_argument("--check-auth", action="store_true", help="request the latest one-day KOSPI sample to verify credentials")
    args = ap.parse_args()
    key = os.getenv("KRX_API_KEY", "").strip()
    if not key:
        print("KRX API key missing. Register/apply at https://openapi.krx.co.kr/ then set KRX_API_KEY.", file=sys.stderr)
        return 2
    client = KRXClient(key, args.raw_dir)
    if args.check_auth:
        body = client.get_day("kospi", ymd(args.end))
        print(json.dumps({"auth": "ok", "date": args.end, "rows": len(data_rows(body))}, ensure_ascii=False))
        return 0
    summary = client.sync(args.endpoints, ymd(args.start), ymd(args.end), args.output, args.delay, args.max_days)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
