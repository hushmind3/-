"""Persistent broker-free paper account driven by completed market bars.

The model's cash-inclusive allocation head determines order budgets. Fills are
next-bar paper fills and rewards include every configured trading cost.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path


SEED_CASH = {"KRW": 10_000_000.0, "USD": 10_000.0}
KR_SELL_TAX_ASSUMPTION = 0.001  # configurable simulation assumption, not a tax quote


def _currency(market: str, asset: str) -> str | None:
    if asset not in ("equity", "etf"):
        return None
    if market in ("KRX", "KOSDAQ"):
        return "KRW"
    if market in ("US", "NYSE", "NASDAQ", "NYSEARCA", "AMEX"):
        return "USD"
    return None


class PaperAccount:
    def __init__(self, path: Path, fee: float, slippage: float):
        self.path = Path(path)
        self.fee = float(fee)
        self.slippage = float(slippage)
        self.state = {
            "version": 1,
            "last_timestamp": None,
            "pending": {},
            "fills": [],
            "books": {currency: {
                "initial_cash": seed, "cash": seed, "positions": {}, "marks": {},
                "fees": 0.0, "slippage": 0.0, "spread": 0.0,
                "sell_tax": 0.0, "realized_pnl": 0.0, "trade_count": 0,
            } for currency, seed in SEED_CASH.items()},
        }
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved.get("version") != 1 or set(saved.get("books", {})) != set(SEED_CASH):
                raise ValueError("paper account state schema does not match the live ledger")
            self.state = saved

    def _equity(self, currency: str) -> float:
        book = self.state["books"][currency]
        return float(book["cash"] + sum(
            float(position["quantity"]) * float(book["marks"].get(symbol, position["average_cost"]))
            for symbol, position in book["positions"].items()))

    def total_equity(self) -> float:
        return float(sum(self._equity(currency) for currency in SEED_CASH))

    def normalized_equity(self) -> float:
        return float(sum(self._equity(c) / max(float(self.state["books"][c]["initial_cash"]), 1e-9)
                         for c in SEED_CASH))

    def model_inputs(self, panel, index: int):
        """Return per-symbol [N,8] and account [8] state for the portfolio heads."""
        n = len(panel.symbols)
        pstate = [[0.0] * 8 for _ in range(n)]
        total_equity = max(self.total_equity(), 1e-9)
        total_cash = sum(float(self.state["books"][c]["cash"]) for c in SEED_CASH)
        total_positions = total_unreal = 0.0
        for j, symbol in enumerate(panel.symbols):
            if not panel.observed[index, j] and not panel.observed[:index + 1, j].any():
                continue
            market, asset = panel.groups[symbol]
            currency = _currency(market, asset)
            if currency is None:
                continue
            book = self.state["books"][currency]
            price = float(panel.closes[index, j])
            if not math.isfinite(price) or price <= 0:
                price = float(book["marks"].get(symbol, 0.0))
            if price <= 0:
                continue
            book["marks"][symbol] = price
            pos = book["positions"].get(symbol)
            qty = float(pos["quantity"]) if pos else 0.0
            avg = float(pos["average_cost"]) if pos else 0.0
            value = qty * price; unreal = value - qty * avg
            equity = max(self._equity(currency), 1e-9)
            pstate[j] = [1.0 if qty else 0.0, value / equity,
                         float(book["cash"]) / equity, unreal / max(equity, 1.0),
                         qty / max(1.0, float(book["initial_cash"]) / price),
                         float(book["trade_count"]) / 1000.0,
                         float(book["fees"]) / max(equity, 1.0),
                         float(book["sell_tax"]) / max(equity, 1.0)]
            total_positions += value; total_unreal += unreal
        account = [total_cash / total_equity, total_positions / total_equity,
                   total_unreal / total_equity,
                   sum(float(self.state["books"][c]["trade_count"]) for c in SEED_CASH) / 1000.0,
                   sum(float(self.state["books"][c]["fees"]) for c in SEED_CASH) / total_equity,
                   sum(float(self.state["books"][c]["slippage"]) for c in SEED_CASH) / total_equity,
                   sum(float(self.state["books"][c]["spread"]) for c in SEED_CASH) / total_equity,
                   1.0 - total_cash / total_equity]
        return pstate, account

    def _fill(self, symbol: str, currency: str, action: str, price: float,
              spread_rate: float, budget: float, timestamp: str) -> None:
        book = self.state["books"][currency]
        half_spread = max(0.0, min(float(spread_rate) / 2.0, 0.025))
        execution_cost = half_spread + self.slippage
        position = book["positions"].get(symbol)
        if action == "BUY":
            if position or budget <= 0:
                return
            quantity = math.floor(min(float(budget), float(book["cash"])) /
                                  (price * (1.0 + execution_cost) * (1.0 + self.fee)))
            if quantity < 1:
                return
            fill_price = price * (1.0 + execution_cost)
            notional = quantity * fill_price
            fee = notional * self.fee
            if notional + fee > book["cash"] + 1e-8:
                return
            book["cash"] -= notional + fee
            book["positions"][symbol] = {"quantity": quantity,
                                          "average_cost": (notional + fee) / quantity}
            tax = 0.0
            realized = 0.0
        elif action == "SELL":
            if not position:
                return  # cash-only: no naked short sale
            quantity = int(position["quantity"])
            fill_price = price * max(0.0, 1.0 - execution_cost)
            notional = quantity * fill_price
            fee = notional * self.fee
            tax = notional * KR_SELL_TAX_ASSUMPTION if currency == "KRW" else 0.0
            realized = notional - fee - tax - quantity * float(position["average_cost"])
            book["cash"] += notional - fee - tax
            book["realized_pnl"] += realized
            del book["positions"][symbol]
        else:
            return
        book["trade_count"] += 1
        book["fees"] += fee
        book["sell_tax"] += tax
        book["spread"] += quantity * price * half_spread
        book["slippage"] += quantity * price * self.slippage
        self.state["fills"].append({"date": timestamp, "symbol": symbol,
            "currency": currency, "action": action, "quantity": quantity,
            "price": fill_price, "fee": fee, "sell_tax": tax,
            "realized_pnl": realized})
        self.state["fills"] = self.state["fills"][-200:]
        if book["cash"] < -1e-6 or any(p["quantity"] < 0 for p in book["positions"].values()):
            raise AssertionError("paper account spent more cash or sold more shares than it owns")

    def process_bar(self, panel, index: int, enabled: bool) -> None:
        timestamp = str(panel.dates[index])
        if self.state["last_timestamp"] and timestamp <= self.state["last_timestamp"]:
            return
        if not enabled:
            self.state["pending"].clear()
        for j, symbol in enumerate(panel.symbols):
            if not panel.observed[index, j]:
                continue
            market, asset = panel.groups[symbol]
            currency = _currency(market, asset)
            if currency is None:
                continue
            price = float(panel.closes[index, j])
            if not math.isfinite(price) or price <= 0:
                continue
            book = self.state["books"][currency]
            book["marks"][symbol] = price
            order = self.state["pending"].get(symbol)
            if enabled and order and order["date"] < timestamp:
                # The completed bar close is known only now. The prior signal
                # could not trade on its own observation bar.
                spread_bps = max(0.0, float(panel.features[index, j, 7]))
                self._fill(symbol, currency, order["action"], price,
                           spread_bps / 10_000.0, float(order.get("budget", 0.0)), timestamp)
                del self.state["pending"][symbol]
        self.state["last_timestamp"] = timestamp

    def queue_decisions(self, panel, index: int, probabilities, enabled: bool,
                        allocation=None) -> None:
        if not enabled:
            return
        timestamp = str(panel.dates[index])
        buys: dict[str, list[tuple[str, float]]] = {key: [] for key in SEED_CASH}
        for j, symbol in enumerate(panel.symbols):
            if not panel.observed[index, j] or symbol in self.state["pending"]:
                continue
            market, asset = panel.groups[symbol]
            currency = _currency(market, asset)
            if currency is None:
                continue
            action = int(probabilities[j].argmax())
            owned = symbol in self.state["books"][currency]["positions"]
            if action == 0 and owned:
                self.state["pending"][symbol] = {"date": timestamp, "action": "SELL"}
            elif action == 2 and not owned:
                score = (float(allocation[j]) if allocation is not None and len(allocation) > j
                         else float(probabilities[j, 2]))
                buys[currency].append((symbol, max(score, 0.0)))
        for currency, signals in buys.items():
            total = sum(score for _, score in signals)
            cash = float(self.state["books"][currency]["cash"])
            if total <= 0 or cash <= 0:
                continue
            for symbol, score in signals:
                self.state["pending"][symbol] = {
                    "date": timestamp, "action": "BUY", "budget": cash * score / total}

    def snapshot(self) -> dict:
        books = {}
        for currency, book in self.state["books"].items():
            equity = self._equity(currency)
            books[currency] = {**book, "equity": equity,
                "net_pnl": equity - float(book["initial_cash"]),
                "holdings_value": equity - float(book["cash"])}
        return {**self.state, "books": books,
                "execution": "next completed bar close; cash-only equities and ETFs",
                "allocation": "model cash-inclusive portfolio_allocation head; legacy fallback is BUY probability",
                "kr_sell_tax_assumption": KR_SELL_TAX_ASSUMPTION}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)
