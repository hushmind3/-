"""Run the current candidate through a chronological, cost-aware paper portfolio replay.

No checkpoint is modified. The candidate's own BUY/HOLD/SELL output gates its
cash-inclusive allocation output; execution occurs at the next bar's open.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stockrl.market_training import CommonMarketTrainingLoader, NormalizedBarAdapter, load_market_candidate_checkpoint
from stockrl.global_transformer import ACTION_NAMES, TransformerConfig
from pretrain_portfolio_agent import (Portfolio, instrument_map_for_daily,
    load_economic_panels, make_context, SLIPPAGE_RATE)


def write_json(path: Path, obj: dict) -> None:
    def convert(value):
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, Path): return str(value)
        raise TypeError(f"not JSON serializable: {type(value).__name__}")
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False, default=convert), encoding="utf-8")


def build_open_panel(source: Path, loader, market: dict) -> np.ndarray:
    raw = pd.read_csv(source, usecols=["date", "symbol", "market", "asset_class", "open"])
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    raw["time_key"] = raw["date"].astype("int64") // 1_000_000_000
    raw["instrument"] = raw["market"].astype(str) + "|" + raw["asset_class"].astype(str) + "|" + raw["symbol"].astype(str)
    pivot = raw.pivot_table(index="time_key", columns="instrument", values="open", aggfunc="last")
    aligned = pivot.reindex(index=loader.time_keys, columns=loader.symbol_names).to_numpy(np.float64)
    factor = np.divide(market["prices"], np.asarray(loader.closes, np.float64),
        out=np.ones_like(market["prices"], dtype=np.float64),
        where=np.isfinite(loader.closes) & (np.asarray(loader.closes) > 0))
    opens = aligned * factor
    # If the vendor omitted an open on a bar, execute at that bar's close proxy.
    return np.where(np.isfinite(opens) & (opens > 0), opens, market["prices"])


def run(args) -> dict:
    if args.runtime.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.runtime}")
    if not args.candidate.is_file():
        raise FileNotFoundError(args.candidate)
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this 0.5B candidate replay")

    device = torch.device("cuda:0")
    args.runtime.mkdir(parents=True)
    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.cuda.get_device_name(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    model, payload = load_market_candidate_checkpoint(args.candidate, device)
    if not all(hasattr(model, name) for name in ("portfolio_action", "portfolio_allocation", "portfolio_cash")):
        raise RuntimeError("candidate has no portfolio action/allocation heads")
    cfg = TransformerConfig(**payload["config"])
    if cfg.max_seq_len < args.sequence_length:
        raise ValueError(f"sequence length {args.sequence_length} exceeds checkpoint limit {cfg.max_seq_len}")

    loader = CommonMarketTrainingLoader(args.cache, [args.source], [NormalizedBarAdapter(86400)],
        chunksize=100_000, seed=20260927, train_fraction=.70, validation_fraction=.15,
        context_stale_seconds=5 * 86400)
    manifest = loader.build(force=False)
    combined_map, local_to_model = instrument_map_for_daily(payload["symbol_map"], loader.symbol_map)
    model_ids = np.asarray([local_to_model[i] for i in range(len(loader.symbol_names))], np.int64)
    if model_ids.max(initial=0) >= cfg.max_symbols:
        raise ValueError("historical instrument IDs exceed candidate embedding capacity")
    market = load_economic_panels(args.source, loader)
    market["sell_tax"] = np.where(
        np.asarray([name.split("|", 2)[1] == "equity" for name in loader.symbol_names]),
        args.equity_sell_tax, 0.0)
    open_prices = build_open_panel(args.source, loader, market)
    n = len(loader.symbol_names)
    local_ids = np.arange(n, dtype=np.int64)
    env = Portfolio(n, args.initial_equity, lots=market["lots"], sell_tax=market["sell_tax"],
        commission=args.commission, participation=args.participation)

    lo, hi = map(int, loader.splits[args.split])
    first = max(lo, args.sequence_length - 1)
    stops = np.arange(first, hi - 1, dtype=np.int64)
    if args.max_steps is not None:
        stops = stops[:args.max_steps]
    if not len(stops):
        raise RuntimeError(f"split {args.split} has no replay steps")

    signal_counts = {name: 0 for name in ACTION_NAMES}
    fill_counts = {"BUY": 0, "SELL": 0}
    totals = {key: 0.0 for key in ("fees", "taxes", "spread_cost", "slippage", "turnover", "realized_pnl")}
    cash_min = float("inf")
    identity_error = 0.0
    rows = []
    decisions = []
    model.eval()
    with torch.inference_mode():
        for step, t in enumerate(stops, 1):
            close = market["prices"][t]
            observed = np.asarray(loader.observed[t], bool)
            tradable = np.asarray(market["tradable"], bool)
            active = observed & tradable & np.isfinite(close) & (close > 0)
            if not active.any():
                continue
            adv = market["adv"][t]
            spread = market["spread"][t]
            impact = np.where(np.isfinite(adv), np.minimum(.02, .001 * np.sqrt(
                np.maximum(args.initial_equity, 0.0) / np.maximum(adv, 1.0))), 0.0)
            cost_rate = args.commission + market["sell_tax"] + spread / 2 + SLIPPAGE_RATE + impact
            ctx = make_context(loader, int(t), local_ids, model_ids, env, close,
                cost_rate, adv, device)
            x, sid, mid, aid, mask, context, pstate, astate = ctx
            with torch.autocast("cuda", dtype=torch.float16):
                logits, values, allocation = model(x, sid, mid, aid, mask, context,
                    torch.tensor([3], device=device), pstate, astate, True)
            logits = logits[0].float()
            values = values[0].float()
            allocation = allocation[0].float()
            if not torch.isfinite(logits).all() or not torch.isfinite(values).all() or not torch.isfinite(allocation).all():
                raise FloatingPointError(f"nonfinite model output at {loader.time_keys[t]}")
            if abs(float(allocation.sum().cpu()) - 1.0) > 1e-4:
                raise AssertionError("cash-inclusive allocation does not sum to one")
            actions = logits.argmax(-1).cpu().numpy()
            allocation_np = allocation.cpu().numpy()
            for action in actions[active]:
                signal_counts[ACTION_NAMES[int(action)]] += 1

            mark = np.where(np.isfinite(close) & (close > 0), close, env.last_prices)
            equity_at_decision = max(env.equity(mark), 1e-12)
            current_weight = env.units * mark / equity_at_decision
            # Compute weights at the actual next-open fill price so HOLD means
            # keep the same quantity across the overnight price gap.
            nxt = t + 1
            fill_prices = open_prices[nxt]
            next_close = market["prices"][nxt]
            next_observed = np.asarray(loader.observed[nxt], bool)
            equity_at_fill = max(env.equity(fill_prices), 1e-12)
            fill_weight = env.units * fill_prices / equity_at_fill
            # SELL targets zero, HOLD and unavailable assets preserve their
            # current weights, and BUY may add up to the model's allocation.
            # Any cash-cap scaling applies only to additional BUY exposure;
            # it can never silently turn a HOLD into a sale.
            target = np.zeros_like(current_weight)
            preserve = (~active) | (active & (actions == 1))
            buy_mask = active & (actions == 2)
            target[preserve] = fill_weight[preserve]
            target[buy_mask] = fill_weight[buy_mask]
            preserved_weight = float(target.sum())
            desired_cash = float(np.clip(allocation_np[-1], 0.0, 1.0))
            buy_capacity = max(0.0, 1.0 - desired_cash - preserved_weight)
            buy_additions = np.zeros_like(current_weight)
            asset_allocations = allocation_np[:-1]
            buy_additions[buy_mask] = np.maximum(asset_allocations[buy_mask] - fill_weight[buy_mask], 0.0)
            addition_total = float(buy_additions.sum())
            if addition_total > buy_capacity and addition_total > 0:
                buy_additions *= buy_capacity / addition_total
            target += buy_additions
            target_with_cash = np.r_[target, max(0.0, 1.0 - float(target.sum()))]

            # Decisions use the completed bar t. Paper fills occur at the
            # following bar's open; portfolio value is marked at that close.
            out = env.rebalance_and_mark(fill_prices, next_close, next_observed,
                next_observed, target_with_cash, tradable, adv, spread)
            if env.cash < -1e-8 or not np.isfinite(env.cash) or np.any(env.units < -1e-10):
                raise AssertionError("cash/position ledger invariant failed")
            identity_error = max(identity_error, abs(
                (out["equity_after"] - out["equity_before"]) - out["reward"] * out["equity_before"]))
            if identity_error > 1e-7:
                raise AssertionError("reported reward does not match account equity change")
            fill_counts["BUY"] += out["action_counts"]["BUY"]
            fill_counts["SELL"] += out["action_counts"]["SELL"]
            totals["fees"] += out["fee"]
            totals["taxes"] += out["tax"]
            totals["spread_cost"] += out["spread_cost"]
            totals["slippage"] += out["slippage"]
            totals["turnover"] += out["turnover"]
            totals["realized_pnl"] += out["realized_pnl_delta"]
            cash_min = min(cash_min, env.cash)
            stamp = pd.to_datetime(loader.time_keys[t], unit="s", utc=True).isoformat()
            decisions.extend({"decision_time": stamp, "symbol": loader.symbol_names[j],
                "action": ACTION_NAMES[int(actions[j])], "value": float(values[j].cpu()),
                "p_sell": float(torch.softmax(logits[j], -1)[0].cpu()),
                "p_hold": float(torch.softmax(logits[j], -1)[1].cpu()),
                "p_buy": float(torch.softmax(logits[j], -1)[2].cpu()),
                "target_weight": float(target[j]), "allocation_proposed": float(allocation_np[j])}
                for j in np.flatnonzero(active))
            rows.append({"decision_time": stamp, "fill_time": pd.to_datetime(loader.time_keys[nxt], unit="s", utc=True).isoformat(),
                "equity": out["equity_after"], "cash": out["cash"], "positions": out["positions"],
                "net_return": out["reward"], "buy_fills": out["action_counts"]["BUY"],
                "sell_fills": out["action_counts"]["SELL"], "fees": out["fee"], "taxes": out["tax"],
                "spread_cost": out["spread_cost"], "slippage": out["slippage"]})
            if step % 20 == 0:
                torch.cuda.synchronize(device)
                print(json.dumps({"step": step, "total_steps": len(stops), "cash": env.cash,
                    "equity": out["equity_after"], "positions": out["positions"]}), flush=True)

    final_marks = np.where(np.isfinite(env.last_prices) & (env.last_prices > 0), env.last_prices, 0.0)
    final_holdings_value = float(np.sum(env.units * final_marks))
    final_equity = env.cash + final_holdings_value
    summary = {
        "candidate_checkpoint": str(args.candidate),
        "candidate_sha256": __import__("hashlib").sha256(args.candidate.read_bytes()).hexdigest(),
        "champion_modified": False, "candidate_modified": False, "real_orders_sent": False,
        "source": str(args.source), "source_manifest": manifest,
        "split": args.split, "time_start": rows[0]["decision_time"], "time_end": rows[-1]["fill_time"],
        "steps": len(rows), "universe_in_loader": n,
        "tradeable_symbols": int(np.asarray(market["tradable"], bool).sum()),
        "sequence_length": args.sequence_length, "initial_equity": args.initial_equity,
        "final_cash": env.cash, "final_holdings_value": final_holdings_value,
        "final_equity": final_equity, "net_pnl": final_equity - args.initial_equity,
        "net_return": final_equity / args.initial_equity - 1.0,
        "realized_net_pnl": env.realized_net,
        "model_action_signals": signal_counts, "executed_fills": fill_counts,
        "ending_position_count": int(np.count_nonzero(env.units > 1e-8)),
        "ending_positions": [{"symbol": loader.symbol_names[i], "quantity": float(env.units[i]),
            "average_cost": float(env.average_cost[i]), "mark": float(env.last_prices[i]),
            "market_value": float(env.units[i] * env.last_prices[i]),
            "unrealized_pnl": float(env.units[i] * (env.last_prices[i] - env.average_cost[i]))}
            for i in np.flatnonzero(env.units > 1e-8)],
        "costs": totals, "cash_min": cash_min,
        "ledger_checks": {"cash_never_negative": cash_min >= -1e-8,
            "positions_never_negative": True, "pnl_equity_identity_error_max": identity_error,
            "allocation_sum_checked_each_step": True},
        "fees_per_side": args.commission, "equity_sell_tax": args.equity_sell_tax,
        "slippage_and_spread": "half daily high-low spread proxy + 1bp base slippage + square-root ADV impact",
        "execution_assumption": "decision after bar close; paper fill at next bar open; mark at next bar close",
        "device": torch.cuda.get_device_name(device),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": time.perf_counter() - t0,
    }
    pd.DataFrame(rows).to_csv(args.runtime / "portfolio_ledger.csv", index=False)
    pd.DataFrame(decisions).to_csv(args.runtime / "decisions.csv", index=False)
    write_json(args.runtime / "summary.json", summary)
    print((args.runtime / "summary.json").read_text(encoding="utf-8"), flush=True)
    loader.close()
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", type=Path, default=ROOT / "runtime-global-cuda-final/candidate.pt")
    ap.add_argument("--source", type=Path, default=ROOT / "data/global_market_daily.csv")
    ap.add_argument("--cache", type=Path, default=ROOT / "runtime-global-market-training/shared-portfolio-daily-cache")
    ap.add_argument("--runtime", type=Path, default=ROOT / "runtime-global-market-training/candidate-shadow-test")
    ap.add_argument("--split", choices=("validation", "test"), default="test")
    ap.add_argument("--sequence-length", type=int, default=128)
    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    ap.add_argument("--commission", type=float, default=.0005)
    ap.add_argument("--equity-sell-tax", type=float, default=.001)
    ap.add_argument("--participation", type=float, default=.01)
    ap.add_argument("--max-steps", type=int)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
