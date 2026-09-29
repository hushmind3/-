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
        self.state = self._empty_state()
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved.get("version") != 1 or set(saved.get("books", {})) != set(SEED_CASH):
                raise ValueError("paper account state schema does not match the live ledger")
            self.state = saved

    @staticmethod
    def _empty_state() -> dict:
        return {
            "version": 1,
            "last_timestamp": None,
            "pending": {},
            "fills": [],
            "books": {currency: {
                "initial_cash": seed, "cash": seed, "positions": {}, "marks": {},
                "fees": 0.0, "slippage": 0.0, "spread": 0.0,
                "sell_tax": 0.0, "realized_pnl": 0.0, "symbol_realized_pnl": {}, "trade_count": 0,
            } for currency, seed in SEED_CASH.items()},
        }

    def reset(self) -> None:
        """Reset a dedicated simulation ledger to the shared starting cash."""
        self.state = self._empty_state()
        self.save()

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

    def symbol_net_pnl(self, symbol: str) -> float:
        """Realized plus open-position PnL for one symbol, normalized by seed cash."""
        for book in self.state["books"].values():
            realized = float(book.get("symbol_realized_pnl", {}).get(symbol, 0.0))
            position = book["positions"].get(symbol)
            unrealized = 0.0
            if position:
                mark = float(book["marks"].get(symbol, position["average_cost"]))
                unrealized = int(position["quantity"]) * (mark - float(position["average_cost"]))
            return (realized + unrealized) / max(float(book["initial_cash"]), 1e-9)
        return 0.0

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
            if budget <= 0:
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
            if position:
                old_quantity = int(position["quantity"])
                old_cost = old_quantity * float(position["average_cost"])
                position["quantity"] = old_quantity + quantity
                position["average_cost"] = (old_cost + notional + fee) / (old_quantity + quantity)
            else:
                book["positions"][symbol] = {"quantity": quantity,
                                              "average_cost": (notional + fee) / quantity}
            tax = 0.0
            realized = 0.0
        elif action == "SELL":
            if not position:
                return  # cash-only: no naked short sale
            requested = int(budget) if budget > 0 else int(position["quantity"])
            quantity = min(int(position["quantity"]), requested)
            if quantity <= 0:
                return
            fill_price = price * max(0.0, 1.0 - execution_cost)
            notional = quantity * fill_price
            fee = notional * self.fee
            tax = notional * KR_SELL_TAX_ASSUMPTION if currency == "KRW" else 0.0
            realized = notional - fee - tax - quantity * float(position["average_cost"])
            book["cash"] += notional - fee - tax
            book["realized_pnl"] += realized
            symbol_realized = book.setdefault("symbol_realized_pnl", {})
            symbol_realized[symbol] = float(symbol_realized.get(symbol, 0.0)) + realized
            position["quantity"] = int(position["quantity"]) - quantity
            if position["quantity"] <= 0:
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
        full_weights = None
        if allocation is not None:
            candidate_weights = [float(value) for value in allocation]
            if (len(candidate_weights) == len(panel.symbols) + 1
                    and all(math.isfinite(value) and value >= 0 for value in candidate_weights)):
                weight_total = sum(candidate_weights)
                if weight_total > 0:
                    # The final entry is the model's explicit cash allocation.
                    # Normalize only numerical drift; never renormalize the BUY
                    # subset, since that would silently spend the cash weight.
                    full_weights = [value / weight_total for value in candidate_weights]
        for j, symbol in enumerate(panel.symbols):
            if not panel.observed[index, j] or symbol in self.state["pending"]:
                continue
            market, asset = panel.groups[symbol]
            currency = _currency(market, asset)
            if currency is None:
                continue
            action = int(probabilities[j].argmax())
            book = self.state["books"][currency]
            position = book["positions"].get(symbol)
            current_quantity = int(position["quantity"]) if position else 0
            if full_weights is None:
                if action == 0 and current_quantity:
                    self.state["pending"][symbol] = {"date": timestamp, "action": "SELL"}
                elif action == 2:
                    buys[currency].append((symbol, max(float(probabilities[j, 2]), 0.0)))
                continue
            if action == 1:
                continue
            price = float(panel.closes[index, j])
            equity = max(0.0, self._equity(currency))
            current_weight = current_quantity * price / equity if equity > 0 else 0.0
            model_weight = full_weights[j]
            target_weight = max(current_weight, model_weight) if action == 2 else min(current_weight, model_weight)
            target_quantity = max(0, int(equity * target_weight / price))
            delta = target_quantity - current_quantity
            if delta > 0:
                buys[currency].append((symbol, delta * price))
            elif delta < 0:
                self.state["pending"][symbol] = {"date": timestamp, "action": "SELL",
                                                  "budget": -delta}
        for currency, signals in buys.items():
            cash = float(self.state["books"][currency]["cash"])
            if cash <= 0:
                continue
            valid_signals=[(symbol,score) for symbol,score in signals if score>0]
            if not valid_signals:
                continue
            if full_weights is None:
                total=sum(score for _,score in valid_signals)
                targets=[(symbol,cash*score/total) for symbol,score in valid_signals]
            else:
                # These are positive target-weight deltas. KRW and USD remain
                # separate; proceeds from queued sells are not spent early.
                targets=valid_signals
                target_total=sum(budget for _,budget in targets)
                # Enforce one shared cash cap while preserving the model's
                # relative weights across this currency's BUY orders.
                scale=min(1.0,cash/target_total) if target_total>0 else 0.0
                targets=[(symbol,budget*scale) for symbol,budget in targets]
            for symbol,budget in targets:
                existing=self.state["pending"].get(symbol)
                if existing and existing.get("date")==timestamp and existing.get("action")=="SELL":
                    continue
                self.state["pending"][symbol] = {
                    "date": timestamp, "action": "BUY", "budget": budget}

    def snapshot(self) -> dict:
        books = {}
        for currency, book in self.state["books"].items():
            equity = self._equity(currency)
            books[currency] = {**book, "equity": equity,
                "net_pnl": equity - float(book["initial_cash"]),
                "holdings_value": equity - float(book["cash"])}
        return {**self.state, "books": books,
                "execution": "next completed bar close; cash-only equities and ETFs",
                "allocation": "model cash-inclusive target weights; action-gated partial buys and sells",
                "kr_sell_tax_assumption": KR_SELL_TAX_ASSUMPTION}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)
