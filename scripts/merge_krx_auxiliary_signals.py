"""Join KRX bars with point-in-time DART, investor, program and short data.

The existing 0.5B input width is kept intact. Its two reserved external-signal
slots are populated with a fundamental composite and a cost/turnover-scaled
flow composite. The full source tables remain separate and are never discarded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.replace("(", "-", regex=False).str.replace(")", "", regex=False), errors="coerce")


def make_fundamental_snapshots(facts: pd.DataFrame) -> pd.DataFrame:
    if facts.empty:
        return pd.DataFrame(columns=["symbol", "available_date", "fundamental_signal"])
    facts = facts.copy()
    facts["symbol"] = facts["stock_code"].astype(str).str.zfill(6)
    facts["available_date"] = pd.to_datetime(facts["available_date"], format="%Y%m%d", errors="coerce", utc=True)
    facts["amount"] = num(facts["thstrm_amount"])
    names = facts["account_id"].fillna("").astype(str) + " " + facts["account_nm"].fillna("").astype(str)
    facts["kind"] = ""
    patterns = {
        "revenue": r"ifrs-full_Revenue|매출액|수익\(매출액\)",
        "operating_income": r"ProfitLossFromOperatingActivities|영업이익",
        "net_income": r"ProfitLoss|당기순이익|분기순이익",
        "assets": r"ifrs-full_Assets|자산총계",
        "liabilities": r"ifrs-full_Liabilities|부채총계",
    }
    for kind, pattern in patterns.items():
        hit = names.str.contains(pattern, case=False, regex=True, na=False)
        facts.loc[hit & facts["kind"].eq(""), "kind"] = kind
    facts = facts[(facts["kind"] != "") & facts["available_date"].notna()]
    facts = facts.sort_values(["symbol", "available_date", "bsns_year", "reprt_code"])
    facts = facts.drop_duplicates(["symbol", "available_date", "kind"], keep="last")
    wide = facts.pivot_table(index=["symbol", "available_date", "bsns_year", "reprt_code"],
                             columns="kind", values="amount", aggfunc="last").reset_index()
    for col in ("revenue", "operating_income", "net_income", "assets", "liabilities"):
        if col not in wide: wide[col] = np.nan
    wide = wide.sort_values(["symbol", "available_date"])
    # Compare like reporting periods (Q1-to-Q1, annual-to-annual), not a
    # quarter against the preceding annual report.
    growth = wide.groupby(["symbol", "reprt_code"], sort=False)["revenue"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    op_margin = wide["operating_income"] / wide["revenue"].abs().replace(0, np.nan)
    net_margin = wide["net_income"] / wide["revenue"].abs().replace(0, np.nan)
    leverage = wide["liabilities"] / wide["assets"].abs().replace(0, np.nan)
    # Robust, bounded composite: growth and profitability help; leverage detracts.
    wide["fundamental_signal"] = (
        .35 * np.tanh(growth.fillna(0).clip(-3, 3)) +
        .30 * np.tanh(op_margin.fillna(0).clip(-2, 2) * 3) +
        .20 * np.tanh(net_margin.fillna(0).clip(-2, 2) * 3) -
        .15 * np.tanh(leverage.fillna(0).clip(0, 10))
    ).clip(-1, 1).astype(np.float32)
    return wide[["symbol", "available_date", "fundamental_signal"]].dropna(subset=["available_date"])


def attach_asof(bars: pd.DataFrame, signals: pd.DataFrame, date_col: str,
                signal_col: str, target_col: str, strictly_prior: bool = False) -> None:
    if signals.empty:
        bars[target_col] = 0.0
        return
    left = bars[["symbol", date_col]].copy()
    left["_row"] = np.arange(len(left))
    left[date_col] = pd.to_datetime(left[date_col], utc=True)
    right = signals[["symbol", "available_date", signal_col]].copy()
    right["symbol"] = right["symbol"].astype(str).str.zfill(6)
    right["available_date"] = pd.to_datetime(right["available_date"], utc=True)
    if strictly_prior:
        # DART values enter only on the first bar strictly after filing day.
        right["available_date"] += pd.Timedelta(days=1)
    left["symbol"] = left["symbol"].astype(str).str.zfill(6)
    result = np.zeros(len(bars), np.float32)
    for sym, group in left.groupby("symbol", sort=False):
        rhs = right[right["symbol"] == sym].sort_values("available_date")
        if rhs.empty: continue
        lhs = group.sort_values(date_col)
        merged = pd.merge_asof(lhs, rhs[["available_date", signal_col]].sort_values("available_date"),
                               left_on=date_col, right_on="available_date", direction="backward")
        result[merged["_row"].to_numpy()] = merged[signal_col].fillna(0).to_numpy(np.float32)
    bars[target_col] = result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", type=Path, default=ROOT / "data/external_sources/krx/krx_daily.csv")
    ap.add_argument("--dart", type=Path, default=ROOT / "data/external_sources/dart/financial_facts.csv")
    ap.add_argument("--flows", type=Path, default=ROOT / "data/external_sources/krx/investor_program_short.csv",
                    help="optional CSV: date,symbol,foreign_net_buy_value,institution_net_buy_value,program_net_buy_value,short_sell_value")
    ap.add_argument("--output", type=Path, default=ROOT / "data/krx_training_daily.csv")
    args = ap.parse_args()
    bars = pd.read_csv(args.bars, dtype={"symbol": str})
    bars["symbol"] = bars["symbol"].astype(str).str.zfill(6)
    bars["date"] = pd.to_datetime(bars["date"], utc=True)
    bars["video_chart_signal"] = 0.0
    bars["video_volume_signal"] = 0.0
    dart_rows = 0
    if args.dart.exists():
        facts = pd.read_csv(args.dart, dtype=str).fillna("")
        dart_rows = len(facts)
        snapshots = make_fundamental_snapshots(facts)
        attach_asof(bars, snapshots, "date", "fundamental_signal", "video_chart_signal", strictly_prior=True)
    flow_rows = 0
    if args.flows.exists():
        flows = pd.read_csv(args.flows, dtype={"symbol": str})
        flow_rows = len(flows)
        flows["symbol"] = flows["symbol"].astype(str).str.zfill(6)
        flows["available_date"] = pd.to_datetime(flows["date"], utc=True)
        turnover = num(bars["trade_value_krw"]) if "trade_value_krw" in bars else bars["close"] * bars["volume"]
        # The ratio is dimensionless and saturates gently to avoid a scale leak.
        money = ["foreign_net_buy_value", "institution_net_buy_value", "program_net_buy_value", "short_sell_value"]
        for col in money:
            flows[col] = num(flows[col]) if col in flows else 0.0
        net = (.45 * flows["foreign_net_buy_value"] + .35 * flows["institution_net_buy_value"] +
               .20 * flows["program_net_buy_value"] - .35 * flows["short_sell_value"])
        flow = flows[["symbol", "available_date"]].copy()
        # Merge each market session's turnover denominator without using future rows.
        flow = flow.merge(bars[["symbol", "date"]].assign(_turnover=turnover).rename(columns={"date": "available_date"}),
                          on=["symbol", "available_date"], how="left")
        flow["flow_signal"] = np.tanh(5 * net / flow["_turnover"].replace(0, np.nan)).fillna(0).clip(-1, 1)
        attach_asof(bars, flow, "date", "flow_signal", "video_volume_signal")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bars["date"] = bars["date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    bars.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary = {"bars": len(bars), "symbols": bars["symbol"].nunique(), "dart_fact_rows": dart_rows,
               "flow_rows": flow_rows, "date_min": str(bars["date"].min()), "date_max": str(bars["date"].max()),
               "fundamental_nonzero": int((bars["video_chart_signal"] != 0).sum()),
               "flow_nonzero": int((bars["video_volume_signal"] != 0).sum()),
               "output": str(args.output), "model_feature_width_unchanged": 17,
               "note": "external fundamentals/flows use the two existing reserved signal slots; raw source tables are retained"}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
