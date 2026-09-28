"""Download point-in-time DART filings/financial statement facts.

Requires an OpenDART key in OPENDART_API_KEY. Financial values retain DART's
receipt number; its first eight digits are the filing receipt date, so facts
can be joined to market bars only from their publication date forward.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://opendart.fss.or.kr/api"
REPORTS = {"11011": "annual", "11012": "half", "11013": "q1", "11014": "q3"}


def parse_corp_zip(blob: bytes) -> list[dict[str, str]]:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        xml_name = next(n for n in zf.namelist() if n.lower().endswith(".xml"))
        root = ET.fromstring(zf.read(xml_name))
    out = []
    for node in root.findall("list"):
        row = {child.tag: (child.text or "").strip() for child in node}
        code = row.get("stock_code", "")
        if len(code) == 6 and code.isdigit():
            out.append({"corp_code": row.get("corp_code", ""), "stock_code": code,
                        "corp_name": row.get("corp_name", "")})
    return out


def api_json(session: requests.Session, endpoint: str, params: dict[str, str]) -> dict:
    response = session.get(f"{BASE}/{endpoint}.json", params=params, timeout=45)
    response.raise_for_status()
    result = response.json()
    code = str(result.get("status", ""))
    if code not in ("000", "013"):
        raise RuntimeError(f"OpenDART {endpoint} status={code}: {result.get('message', '')}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=datetime.now().year)
    ap.add_argument("--max-companies", type=int, default=0, help="smoke cap; 0 means all listed companies")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data/external_sources/dart")
    ap.add_argument("--pause", type=float, default=0.05)
    args = ap.parse_args()
    key = os.getenv("OPENDART_API_KEY", "").strip() or os.getenv("DART_API_KEY", "").strip()
    if not key:
        print("OpenDART key missing. Apply at https://opendart.fss.or.kr/ then set OPENDART_API_KEY.", file=sys.stderr)
        return 2
    if args.start_year > args.end_year:
        raise ValueError("start-year must be <= end-year")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    corp_path = args.output_dir / "corp_codes.csv"
    if corp_path.exists():
        import pandas as pd
        corps = pd.read_csv(corp_path, dtype=str).fillna("").to_dict("records")
    else:
        response = session.get(f"{BASE}/corpCode.xml", params={"crtfc_key": key}, timeout=60)
        response.raise_for_status()
        corps = parse_corp_zip(response.content)
        with corp_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["corp_code", "stock_code", "corp_name"])
            w.writeheader(); w.writerows(corps)
    if args.max_companies:
        corps = corps[:args.max_companies]
    out_path = args.output_dir / "financial_facts.csv"
    seen = set()
    rows = []
    if out_path.exists():
        import pandas as pd
        old = pd.read_csv(out_path, dtype=str).fillna("")
        rows = old.to_dict("records")
        seen = {(r.get("corp_code"), r.get("bsns_year"), r.get("reprt_code")) for r in rows}
    failures = []
    requests_made = 0
    for ci, corp in enumerate(corps, 1):
        for year in range(args.start_year, args.end_year + 1):
            for report_code, report_name in REPORTS.items():
                ident = (corp["corp_code"], str(year), report_code)
                if ident in seen:
                    continue
                try:
                    query = {
                        "crtfc_key": key, "corp_code": corp["corp_code"], "bsns_year": str(year),
                        "reprt_code": report_code, "fs_div": "CFS"}
                    result = api_json(session, "fnlttSinglAcntAll", query)
                    statement_rows = result.get("list", [])
                    # Many smaller listed firms publish only separate (OFS)
                    # statements. Fall back rather than silently dropping them.
                    if not statement_rows:
                        query["fs_div"] = "OFS"
                        result = api_json(session, "fnlttSinglAcntAll", query)
                        statement_rows = result.get("list", [])
                    for fact in statement_rows:
                        rcept = str(fact.get("rcept_no", ""))
                        rows.append({"corp_code": corp["corp_code"], "stock_code": corp["stock_code"],
                                     "corp_name": corp["corp_name"], "available_date": rcept[:8],
                                     "rcept_no": rcept, "bsns_year": str(year), "reprt_code": report_code,
                                     "report_name": report_name, "fs_div": fact.get("fs_div", query["fs_div"]),
                                     "sj_div": fact.get("sj_div", ""), "account_id": fact.get("account_id", ""),
                                     "account_nm": fact.get("account_nm", ""),
                                     "thstrm_amount": fact.get("thstrm_amount", ""),
                                     "frmtrm_amount": fact.get("frmtrm_amount", ""),
                                     "bfefrmtrm_amount": fact.get("bfefrmtrm_amount", "")})
                    seen.add(ident)
                except Exception as exc:
                    failures.append({"corp_code": corp["corp_code"], "stock_code": corp["stock_code"],
                                     "year": year, "report_code": report_code, "error": str(exc)[:400]})
                requests_made += 1
                if args.pause:
                    time.sleep(args.pause)
        if ci % 25 == 0:
            print(f"dart companies={ci}/{len(corps)} account_rows={len(rows):,} failures={len(failures)}", flush=True)
            write_rows(out_path, rows)
    write_rows(out_path, rows)
    summary = {"provider": "OpenDART", "start_year": args.start_year, "end_year": args.end_year,
               "companies": len(corps), "requests": requests_made, "fact_rows": len(rows),
               "failures": len(failures), "output": str(out_path),
               "point_in_time_field": "available_date derived from DART receipt number; no filing fact is backdated"}
    (args.output_dir / "ingest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        (args.output_dir / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = ["corp_code", "stock_code", "corp_name", "available_date", "rcept_no", "bsns_year",
              "reprt_code", "report_name", "fs_div", "sj_div", "account_id", "account_nm",
              "thstrm_amount", "frmtrm_amount", "bfefrmtrm_amount"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
