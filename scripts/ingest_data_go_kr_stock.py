"""Download Financial Services Commission daily stock quotes from data.go.kr.

The API key is read from DATA_GO_KR_STOCK_KEY or a hidden console prompt. It is
never written to disk and request URLs are never printed. Responses are cached
by page so interrupted downloads can resume without repeating completed calls.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo"
DEFAULT_RAW = ROOT / "data/external_sources/data_go_kr_stock/raw"
DEFAULT_OUT = ROOT / "data/external_sources/data_go_kr_stock/kr_stock_daily.csv"
DB_PATH = ROOT / "data/external_sources/data_go_kr_stock/normalize.sqlite3"
FIELDS = ("date", "symbol", "market", "asset_class", "open", "high", "low", "close",
          "volume", "trade_value", "market_cap", "name", "isin")


def extract_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    header = payload.get("response", {}).get("header", {})
    code, message = str(header.get("resultCode", "")), str(header.get("resultMsg", ""))
    if code not in ("00", "0", ""):
        raise RuntimeError(f"API returned resultCode={code} resultMsg={message}")
    body = payload.get("response", {}).get("body", {})
    items = body.get("items", {})
    rows = items.get("item", []) if isinstance(items, dict) else items
    if isinstance(rows, dict):
        rows = [rows]
    return [r for r in rows if isinstance(r, dict)], int(body.get("totalCount", 0) or 0)


def clean(row: dict[str, Any]) -> tuple[Any, ...] | None:
    code = str(row.get("srtnCd", "")).strip()
    date = str(row.get("basDt", "")).strip()
    close = str(row.get("clpr", "")).replace(",", "").strip()
    if not code or len(date) != 8 or not close or close in ("-", "--"):
        return None
    market = str(row.get("mrktCtg", "KRX")).strip() or "KRX"
    return (f"{date[:4]}-{date[4:6]}-{date[6:]}T00:00:00Z", code.zfill(6), market,
            "equity", float(str(row.get("mkp", "0")).replace(",", "") or 0),
            float(str(row.get("hipr", "0")).replace(",", "") or 0),
            float(str(row.get("lopr", "0")).replace(",", "") or 0), float(close),
            float(str(row.get("trqu", "0")).replace(",", "") or 0),
            float(str(row.get("trPrc", "0")).replace(",", "") or 0),
            float(str(row.get("mrktTotAmt", "0")).replace(",", "") or 0),
            str(row.get("itmsNm", "")), str(row.get("isinCd", "")))


def safe_get(session: requests.Session, params: dict[str, Any], timeout: int) -> requests.Response:
    response = session.get(URL, params=params, timeout=timeout)
    if response.status_code >= 400:
        # requests.HTTPError includes the full URL (and serviceKey query value).
        # Never raise it or print the response/request object.
        try:
            detail = response.json()
            header = detail.get("response", {}).get("header", {})
            reason = f" resultCode={header.get('resultCode')} resultMsg={header.get('resultMsg')}"
        except Exception:
            body = re.sub(r"(?i)(serviceKey=)[^&\s\"']+", r"\1<redacted>", response.text[:400])
            reason = " response_body=" + body.replace("\n", " ")
        raise RuntimeError(f"API HTTP status={response.status_code}.{reason}")
    return response


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, help="limit API pages for a smoke run")
    parser.add_argument("--delay", type=float, default=0.12)
    parser.add_argument("--prompt-key", action="store_true", help="read key without echo")
    args = parser.parse_args()
    key = os.environ.get("DATA_GO_KR_STOCK_KEY")
    if not key and args.prompt_key:
        key = getpass.getpass("data.go.kr API key (input hidden): ").strip()
    if not key:
        print("Set DATA_GO_KR_STOCK_KEY or pass --prompt-key.", file=sys.stderr)
        return 2
    if not 1 <= args.page_size <= 1000:
        parser.error("--page-size must be between 1 and 1000")

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("CREATE TABLE IF NOT EXISTS quotes (date TEXT, symbol TEXT, market TEXT, asset_class TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, trade_value REAL, market_cap REAL, name TEXT, isin TEXT, PRIMARY KEY(date,symbol))")
    session = requests.Session()
    session.headers.update({"User-Agent": "stockrl-research-data-ingestor/1.0"})
    first_file = args.raw_dir / "page-0000001.json"
    try:
        if first_file.exists():
            first_payload = json.loads(first_file.read_text(encoding="utf-8"))
        else:
            response = safe_get(session, {"serviceKey": key, "numOfRows": args.page_size,
                                          "pageNo": 1, "resultType": "json"}, timeout=60)
            first_payload = response.json()
            extract_rows(first_payload)
            first_file.write_text(json.dumps(first_payload, ensure_ascii=False), encoding="utf-8")
        _, total = extract_rows(first_payload)
        if total <= 0:
            print("API returned zero totalCount.", file=sys.stderr)
            return 3
        pages = (total + args.page_size - 1) // args.page_size
        last_page = min(pages, args.max_pages) if args.max_pages else pages
        print(f"api_total_rows={total:,} page_size={args.page_size} pages={pages:,} scheduled_pages={last_page:,}", flush=True)

        inserted = 0
        for page in range(1, last_page + 1):
            cached = args.raw_dir / f"page-{page:07d}.json"
            if cached.exists():
                payload = json.loads(cached.read_text(encoding="utf-8"))
            elif page == 1:
                payload = first_payload
            else:
                time.sleep(max(args.delay, 0.0))
                response = safe_get(session, {"serviceKey": key, "numOfRows": args.page_size,
                                              "pageNo": page, "resultType": "json"}, timeout=90)
                payload = response.json()
                cached.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            rows, _ = extract_rows(payload)
            values = [item for row in rows if (item := clean(row)) is not None]
            conn.executemany("INSERT OR IGNORE INTO quotes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", values)
            inserted += conn.total_changes
            if page % 50 == 0 or page == last_page:
                conn.commit()
                print(f"pages_done={page:,}/{last_page:,} page_rows={len(rows):,} newly_inserted_total={inserted:,}", flush=True)
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM quotes").fetchone()[0]
        bounds = conn.execute("SELECT MIN(date),MAX(date),COUNT(DISTINCT symbol) FROM quotes").fetchone()
        with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerows(conn.execute("SELECT date,symbol,market,asset_class,open,high,low,close,volume,trade_value,market_cap,name,isin FROM quotes ORDER BY date,symbol"))
        manifest = {"source": "Financial Services Commission, Korea Listed Stock Quote Information",
                    "source_url": "https://www.data.go.kr/data/15094808/openapi.do",
                    "endpoint_operation": "getStockPriceInfo", "rows": count,
                    "symbols": bounds[2], "date_range_utc": [bounds[0], bounds[1]],
                    "requested_pages": last_page, "total_pages": pages,
                    "output": str(args.output.resolve()), "api_key_saved": False,
                    "note": "Daily delayed data; not real-time."
                    }
        (args.output.parent / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"downloaded_unique_rows={count:,} symbols={bounds[2]:,} date_range={bounds[0]}..{bounds[1]}", flush=True)
        print(f"output={args.output} bytes={args.output.stat().st_size:,}", flush=True)
        return 0 if last_page == pages else 4
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
