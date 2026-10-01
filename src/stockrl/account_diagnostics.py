"""Read-only measurements shared by the two observation ledgers."""
from __future__ import annotations

import math

import numpy as np

from .paper_account import _currency


def valid_bid_ask_count(rows) -> int:
    count = 0
    for row in rows:
        try:
            values = [float(row.get(key) or 0) for key in ("bid", "ask")]
        except (ValueError, TypeError):
            continue
        count += all(math.isfinite(value) and value > 0 for value in values)
    return count


def input_availability(rows):
    rows=list(rows)
    def positive(row,key):
        try:
            value=float(row.get(key) or 0)
            return math.isfinite(value) and value>0
        except (ValueError,TypeError):
            return False
    return {"quote_symbols":len(rows),
        "ohlcv_symbols":sum(all(positive(row,key) for key in ("open","high","low","close")) for row in rows),
        "bid_ask_symbols":valid_bid_ask_count(rows),
        "book_size_symbols":sum(positive(row,"bid_size") and positive(row,"ask_size") for row in rows),
        "directional_volume_symbols":sum(positive(row,"buy_volume") or positive(row,"sell_volume") for row in rows),
        "trade_count_symbols":sum(positive(row,"trade_count") for row in rows)}


def summarize_account(account: dict) -> dict:
    """Compute both ledgers with one formula; never mix KRW and USD."""
    books = {}
    fills = account.get("fills", [])
    for currency, book in account.get("books", {}).items():
        seed = float(book.get("initial_cash", 0))
        cash = float(book.get("cash", 0))
        positions = []
        marks = book.get("marks", {})
        for symbol, position in book.get("positions", {}).items():
            quantity = float(position.get("quantity", 0))
            if not quantity:
                continue
            average = float(position.get("average_cost", 0))
            mark = float(marks.get(symbol, average))
            positions.append({"symbol": symbol, "quantity": quantity,
                "average_cost": average, "mark": mark, "value": quantity * mark,
                "unrealized_pnl": quantity * (mark - average),
                "mark_available": symbol in marks})
        holdings = sum(p["value"] for p in positions)
        unrealized = sum(p["unrealized_pnl"] for p in positions)
        equity = cash + holdings
        realized = float(book.get("realized_pnl", 0))
        components = {key: float(book.get(key, 0))
                      for key in ("fees", "sell_tax", "spread", "slippage")}
        costs = sum(components.values())
        for position in positions:
            position["weight"] = position["value"] / equity if equity > 0 else None
        recent = [f for f in fills if f.get("currency") == currency]
        difference = equity - seed - realized - unrealized
        statistics=dict(book.get("trade_statistics",{}))
        sells=int(statistics.get("sell_count",0)); holds=int(statistics.get("holding_count",0))
        statistics["sell_win_rate"]=statistics.get("sell_wins",0)/sells if sells else None
        statistics["mean_sell_net_pnl"]=statistics.get("sell_realized_sum",0)/sells if sells else None
        statistics["mean_holding_seconds"]=statistics.get("holding_seconds_sum",0)/holds if holds else None
        hours=0.0
        if statistics.get("first_timestamp") and account.get("last_timestamp"):
            try:
                hours=float((np.datetime64(account["last_timestamp"])-np.datetime64(
                    statistics["first_timestamp"]))/np.timedelta64(1,"h"))
            except (TypeError,ValueError):
                pass
        statistics["fills_per_elapsed_hour"]=(statistics.get("buy_count",0)+sells)/hours if hours>0 else None
        books[currency] = {"initial_cash": seed, "cash": cash, "equity": equity,
            "holdings_value": holdings, "net_pnl": equity - seed,
            "net_return_rate": (equity - seed) / seed if seed else None,
            "realized_pnl": realized, "unrealized_pnl": unrealized,
            **components, "costs": costs, "cost_return_rate": costs / seed if seed else None,
            # This is an accounting add-back, not a no-cost counterfactual run.
            "recorded_pnl_plus_costs": equity - seed + costs,
            "cash_ratio": cash / equity if equity > 0 else None,
            "largest_position_weight": max((p["weight"] or 0 for p in positions), default=0),
            "position_count": len(positions), "trade_count": int(book.get("trade_count", 0)),
            "positions": sorted(positions, key=lambda p: -p["value"]),
            "recent_buy_fills": sum(f.get("action") == "BUY" for f in recent),
            "recent_sell_fills": sum(f.get("action") == "SELL" for f in recent),
            "recent_fill_count": len(recent),
            "trade_statistics":statistics,
            "reconciliation_difference": difference,
            "reconciliation_ok": abs(difference) <= max(.01, abs(seed) * 1e-8)}
    return {"last_timestamp": account.get("last_timestamp"), "books": books,
            "pending_orders": len(account.get("pending", {})),
            "recent_fills": fills[-200:], "recent_fill_limit": 200}


def summarize_policy(panel, index, probabilities, actions, allocation,
                     account, submitted_ids, timestamp) -> dict:
    """Measure already computed actions; do not run another model forward."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    weights = None
    if allocation is not None:
        values = np.asarray(allocation, dtype=np.float64)
        if (values.shape == (len(panel.symbols) + 1,) and np.isfinite(values).all()
                and (values >= 0).all() and values.sum() > 0):
            weights = values / values.sum()
    indices = {"KRW": [], "USD": []}
    for j, symbol in enumerate(panel.symbols):
        if not panel.observed[index, j] or symbol not in panel.groups:
            continue
        currency = _currency(*panel.groups[symbol])
        price = float(panel.closes[index, j])
        if currency in indices and math.isfinite(price) and price > 0:
            indices[currency].append(j)
    rows = []
    cash_weights = {}
    for currency, members in indices.items():
        denominator = (weights[-1] + sum(weights[j] for j in members)
                       if weights is not None else 0)
        cash_weights[currency] = (float(weights[-1] / denominator)
                                 if denominator > 0 and members else None)
        for j in members:
            symbol = panel.symbols[j]
            p = probabilities[j]
            held = account.state["books"][currency]["positions"].get(symbol, {})
            rows.append({"symbol": symbol, "currency": currency,
                "action": ("SELL", "HOLD", "BUY")[int(actions[j])],
                "p_sell": float(p[0]), "p_hold": float(p[1]), "p_buy": float(p[2]),
                "top_probability": float(p.max()),
                "entropy": float(-(p * np.log(np.maximum(p, 1e-12))).sum() / math.log(3)),
                "differs_from_argmax": int(actions[j]) != int(p.argmax()),
                "held_quantity": float(held.get("quantity", 0)),
                "allocation_target": float(weights[j] / denominator) if denominator > 0 else None,
                "order_submitted": f"{timestamp}|{symbol}" in submitted_ids})
    return {"timestamp": timestamp, "observed_tradable_symbols": len(rows),
        "action_counts": {a: sum(row["action"] == a for row in rows)
                          for a in ("SELL", "HOLD", "BUY")},
        "submitted_orders": len(submitted_ids),
        "sell_without_position": sum(row["action"] == "SELL" and not row["held_quantity"] for row in rows),
        "sampled_actions_different_from_argmax": sum(row["differs_from_argmax"] for row in rows),
        "mean_top_probability": sum(row["top_probability"] for row in rows) / len(rows) if rows else None,
        "mean_normalized_entropy": sum(row["entropy"] for row in rows) / len(rows) if rows else None,
        "effective_cash_target": cash_weights, "decisions": rows}
