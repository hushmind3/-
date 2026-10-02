"""Persistent broker-free paper account driven by completed market bars.

The model's cash-inclusive allocation head determines order budgets. Fills are
next-bar paper fills and rewards include every configured trading cost.
"""
from __future__ import annotations

import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .state_io import atomic_json


SEED_CASH = {"KRW": 10_000_000.0, "USD": 10_000.0}
KR_SELL_TAX_ASSUMPTION = 0.001  # configurable simulation assumption, not a tax quote


def _currency(market: str, asset: str) -> str | None:
    # Explicit cash-only USDT crypto simulation in the USD seed ledger.
    # No margin/futures or implicit conversion of arbitrary crypto quotes.
    if asset == "crypto" and market == "BINANCE_USDT":
        return "USD"
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
            self.state.setdefault("episode_id",uuid.uuid4().hex)

    @staticmethod
    def _empty_state() -> dict:
        return {
            "version": 1,
            "episode_id": uuid.uuid4().hex,
            "last_timestamp": None,
            "pending": {},
            "fills": [],
            "books": {currency: {
                "initial_cash": seed, "cash": seed, "positions": {}, "marks": {},
                "fees": 0.0, "slippage": 0.0, "spread": 0.0,
                "sell_tax": 0.0, "realized_pnl": 0.0, "symbol_realized_pnl": {}, "trade_count": 0,
            } for currency, seed in SEED_CASH.items()},
        }

    @classmethod
    def in_memory(cls, fee: float, slippage: float):
        account=cls.__new__(cls)
        account.path=None
        account.fee=float(fee)
        account.slippage=float(slippage)
        account.state=cls._empty_state()
        return account

    def reset(self) -> None:
        """Reset a dedicated simulation ledger to the shared starting cash."""
        goal=self.state.get("goal")
        self.state = self._empty_state()
        if goal:
            self.configure_goal(goal["target_multiple"],goal["win_bonus_points"])
        self.save()

    def configure_goal(self, target_multiple=10.0, win_bonus_points=100.0):
        """Attach a shared task without changing cash, positions or past returns."""
        target_multiple=float(target_multiple);win_bonus_points=float(win_bonus_points)
        if not math.isfinite(target_multiple) or target_multiple<=1 or not math.isfinite(win_bonus_points) or win_bonus_points<=0:
            raise ValueError("goal target must exceed one; WIN points must be positive")
        self.state.setdefault("episode_id",uuid.uuid4().hex)
        goal=self.state.setdefault("goal",{"target_multiple":target_multiple,
            "win_bonus_points":win_bonus_points,"wins":{}})
        if goal["target_multiple"]!=target_multiple or goal["win_bonus_points"]!=win_bonus_points:
            raise ValueError("changing an active account goal requires an explicit new episode")

    def observe_goal(self, timestamp):
        goal=self.state.get("goal")
        if not goal:
            return
        for currency,book in self.state["books"].items():
            multiple=self._equity(currency)/float(book["initial_cash"])
            if currency not in goal["wins"] and multiple>=goal["target_multiple"]:
                goal["wins"][currency]={"timestamp":str(timestamp),"multiple":multiple,
                    "bonus_points":goal["win_bonus_points"]}

    def goal_points(self):
        return {c:float(self.state.get("goal",{}).get("wins",{}).get(c,{}).get("bonus_points",0.0))
                for c in SEED_CASH}

    def goal_inputs(self):
        goal=self.state.get("goal")
        if not goal:
            return None
        ratios=[self._equity(c)/float(self.state["books"][c]["initial_cash"]) for c in SEED_CASH]
        target=float(goal["target_multiple"])
        return [*ratios,target,*[max(0.0,1.0-r/target) for r in ratios],
                float(len(goal["wins"])<len(SEED_CASH))]

    def goal_summary(self):
        goal=self.state.get("goal")
        if not goal:
            return {"enabled":False}
        target=float(goal["target_multiple"])
        return {"enabled":True,"episode_id":self.state["episode_id"],
            "target_multiple":target,"win_bonus_points":goal["win_bonus_points"],
            "status":"WIN" if len(goal["wins"])==len(SEED_CASH) else "IN_PROGRESS",
            "win_condition":"each_currency_reaches_target_once_in_this_episode",
            "books":{c:{"initial_cash":float(b["initial_cash"]),"equity":self._equity(c),
                "multiple":self._equity(c)/float(b["initial_cash"]),
                "target_equity":float(b["initial_cash"])*target,
                "target_asset_ratio":self._equity(c)/(float(b["initial_cash"])*target),
                "net_return_rate":self._equity(c)/float(b["initial_cash"])-1,
                "progress":min(1.0,max(0.0,(self._equity(c)/float(b["initial_cash"])-1)/(target-1))),
                "win":goal["wins"].get(c)} for c,b in self.state["books"].items()}}

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

    def reward_points(self) -> dict:
        """One net-return percentage point equals one point, per currency."""
        return {c:100.0*(self._equity(c)/float(self.state["books"][c]["initial_cash"])-1.0)
                for c in SEED_CASH}

    def symbol_net_pnl(self, symbol: str) -> float:
        """Realized plus open-position PnL for one symbol, normalized by seed cash."""
        for book in self.state["books"].values():
            if (symbol not in book["positions"] and
                    symbol not in book.get("symbol_realized_pnl", {})):
                continue
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
        for j, symbol in enumerate(panel.symbols):
            seen=(panel.ever_observed[j] if hasattr(panel,"ever_observed")
                  else panel.observed[:index + 1,j].any())
            if not panel.observed[index, j] and not seen:
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
        # KRW and USD are separate paper ledgers. Express each amount in its
        # own book's seed-cash units before combining account features.
        normalized_equity = max(self.normalized_equity(), 1e-9)
        normalized_cash = normalized_positions = normalized_unreal = 0.0
        normalized_fees = normalized_slippage = normalized_spread = 0.0
        trade_count = 0.0
        for book in self.state["books"].values():
            seed = max(float(book["initial_cash"]), 1e-9)
            normalized_cash += float(book["cash"]) / seed
            normalized_positions += sum(
                float(position["quantity"]) * float(book["marks"].get(symbol, position["average_cost"]))
                for symbol, position in book["positions"].items()) / seed
            normalized_unreal += sum(
                float(position["quantity"]) * (
                    float(book["marks"].get(symbol, position["average_cost"]))
                    - float(position["average_cost"]))
                for symbol, position in book["positions"].items()) / seed
            normalized_fees += float(book["fees"]) / seed
            normalized_slippage += float(book["slippage"]) / seed
            normalized_spread += float(book["spread"]) / seed
            trade_count += float(book["trade_count"])
        account = [normalized_cash / normalized_equity,
                   normalized_positions / normalized_equity,
                   normalized_unreal / normalized_equity,
                   trade_count / 1000.0,
                   normalized_fees / normalized_equity,
                   normalized_slippage / normalized_equity,
                   normalized_spread / normalized_equity,
                   normalized_positions / normalized_equity]
        return pstate, account

    def _fill(self, symbol: str, currency: str, action: str, price: float,
              spread_rate: float, budget: float, timestamp: str,
              requested_quantity: int | None = None,
              decision_id: str | None = None,
              order_date: str | None = None) -> dict | None:
        book = self.state["books"][currency]
        half_spread = max(0.0, min(float(spread_rate) / 2.0, 0.025))
        execution_cost = half_spread + self.slippage
        position = book["positions"].get(symbol)
        if action == "BUY":
            if budget <= 0 and requested_quantity is None:
                return None
            unit_cost=price*(1.0+execution_cost)*(1.0+self.fee)
            affordable=math.floor(float(book["cash"])/unit_cost)
            quantity=(min(max(0,int(requested_quantity)),affordable)
                      if requested_quantity is not None else
                      math.floor(min(float(budget),float(book["cash"]))/unit_cost))
            if quantity < 1:
                return None
            fill_price = price * (1.0 + execution_cost)
            notional = quantity * fill_price
            fee = notional * self.fee
            if notional + fee > book["cash"] + 1e-8:
                return None
            book["cash"] -= notional + fee
            if position:
                old_quantity = int(position["quantity"])
                old_cost = old_quantity * float(position["average_cost"])
                position["quantity"] = old_quantity + quantity
                position["average_cost"] = (old_cost + notional + fee) / (old_quantity + quantity)
            else:
                book["positions"][symbol] = {"quantity": quantity,
                    "average_cost": (notional + fee) / quantity,"opened_timestamp":timestamp}
            tax = 0.0
            realized = 0.0
        elif action == "SELL":
            if not position:
                return None  # cash-only: no naked short sale
            requested = int(budget) if budget > 0 else int(position["quantity"])
            quantity = min(int(position["quantity"]), requested)
            if quantity <= 0:
                return None
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
            return None
        book["trade_count"] += 1
        book["fees"] += fee
        book["sell_tax"] += tax
        book["spread"] += quantity * price * half_spread
        book["slippage"] += quantity * price * self.slippage
        statistics=book.setdefault("trade_statistics",{
            "first_timestamp":timestamp,"buy_count":0,"sell_count":0,
            "sell_wins":0,"sell_losses":0,"sell_flat":0,"sell_realized_sum":0.0,
            "holding_count":0,"holding_seconds_sum":0.0,
            "holding_seconds_min":None,"holding_seconds_max":None})
        statistics["last_timestamp"]=timestamp
        statistics["buy_count" if action=="BUY" else "sell_count"]+=1
        holding_seconds=None
        if action=="SELL":
            statistics["sell_wins" if realized>0 else "sell_losses" if realized<0 else "sell_flat"]+=1
            statistics["sell_realized_sum"]+=realized
            if position.get("opened_timestamp"):
                def utc(value):
                    parsed=datetime.fromisoformat(str(value)[:26].replace("Z","+00:00"))
                    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
                try:
                    holding_seconds=max(0,(utc(timestamp)-utc(position["opened_timestamp"])).total_seconds())
                except (ValueError,TypeError):
                    holding_seconds=None
                if holding_seconds is not None:
                    statistics["holding_count"]+=1
                    statistics["holding_seconds_sum"]+=holding_seconds
                    for key,fn in (("holding_seconds_min",min),("holding_seconds_max",max)):
                        statistics[key]=holding_seconds if statistics[key] is None else fn(statistics[key],holding_seconds)
        fill={"date": timestamp, "symbol": symbol,
            "currency": currency, "action": action, "quantity": quantity,
            "price": fill_price, "fee": fee, "sell_tax": tax,
            "realized_pnl": realized, "decision_id": decision_id,
            "order_date": order_date,"holding_seconds":holding_seconds}
        self.state["fills"].append(fill)
        self.state["fills"] = self.state["fills"][-200:]
        if book["cash"] < -1e-6 or any(p["quantity"] < 0 for p in book["positions"].values()):
            raise AssertionError("paper account spent more cash or sold more shares than it owns")
        return fill

    def process_bar(self, panel, index: int, enabled: bool) -> list[dict]:
        timestamp = str(panel.dates[index])
        if self.state["last_timestamp"] and timestamp <= self.state["last_timestamp"]:
            return []
        if not enabled:
            self.state["pending"].clear()
        filled=[]
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
                symbol_pnl_before_fill=self.symbol_net_pnl(symbol)
                fill=self._fill(symbol, currency, order["action"], price,
                           spread_bps / 10_000.0, float(order.get("budget", 0.0)), timestamp,
                           requested_quantity=(int(order["requested_quantity"])
                                               if order.get("requested_quantity") is not None else None),
                           decision_id=order.get("decision_id"),
                           order_date=order.get("date"))
                if fill is not None:
                    fill["symbol_pnl_before_fill"]=symbol_pnl_before_fill
                    filled.append(fill)
                del self.state["pending"][symbol]
        self.state["last_timestamp"] = timestamp
        return filled

    def queue_decisions(self, panel, index: int, probabilities, enabled: bool,
                        allocation=None, actions=None) -> set[str]:
        if not enabled:
            return set()
        timestamp = str(panel.dates[index])
        queued_decisions=set()
        buys: dict[str, list[tuple[str, float]]] = {key: [] for key in SEED_CASH}
        buy_quantities: dict[str,float] = {}
        buy_prices: dict[str,float] = {}
        buy_spreads: dict[str,float] = {}
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
        currency_weights=None
        if full_weights is not None:
            currency_indices={key:[] for key in SEED_CASH}
            for j,symbol in enumerate(panel.symbols):
                if not panel.observed[index,j] or symbol not in panel.groups:
                    continue
                market,asset=panel.groups[symbol]
                currency=_currency(market,asset)
                price=float(panel.closes[index,j])
                if currency is not None and math.isfinite(price) and price>0:
                    currency_indices[currency].append(j)
            cash_weight=full_weights[-1]
            currency_weights=[0.0]*len(panel.symbols)
            for currency,indices in currency_indices.items():
                # Each paper ledger has its own seed cash. Remove weight assigned
                # to other currencies and non-tradable assets before converting
                # the model's ranking into this ledger's target weights.
                denominator=cash_weight+sum(full_weights[j] for j in indices)
                if denominator>0:
                    for j in indices:
                        currency_weights[j]=full_weights[j]/denominator
        for j, symbol in enumerate(panel.symbols):
            if (not panel.observed[index,j] or symbol not in panel.groups
                    or symbol in self.state["pending"]):
                continue
            market, asset = panel.groups[symbol]
            currency = _currency(market, asset)
            if currency is None:
                continue
            action = int(actions[j]) if actions is not None else int(probabilities[j].argmax())
            book = self.state["books"][currency]
            position = book["positions"].get(symbol)
            current_quantity = int(position["quantity"]) if position else 0
            if full_weights is None:
                if action == 0 and current_quantity:
                    decision_id=f"{timestamp}|{symbol}"
                    self.state["pending"][symbol] = {"date": timestamp, "action": "SELL",
                                                     "decision_id": decision_id}
                    queued_decisions.add(decision_id)
                elif action == 2:
                    buys[currency].append((symbol, max(float(probabilities[j, 2]), 0.0)))
                continue
            if action == 1:
                continue
            price = float(panel.closes[index, j])
            equity = max(0.0, self._equity(currency))
            current_weight = current_quantity * price / equity if equity > 0 else 0.0
            model_weight = currency_weights[j]
            if action == 2:
                target_weight = max(current_weight, model_weight)
            else:
                # SELL must reduce an existing position. When the allocation
                # head contradicts SELL by asking to keep/increase its weight,
                # the explicit exit signal takes precedence.
                target_weight = min(current_weight, model_weight)
                if model_weight >= current_weight:
                    target_weight = 0.0
            target_quantity = max(0.0, equity * target_weight / price)
            delta = target_quantity - current_quantity
            if delta > 0:
                buys[currency].append((symbol, delta * price))
                buy_quantities[symbol]=delta
                buy_prices[symbol]=price
                buy_spreads[symbol]=max(0.0,float(panel.features[index,j,7]))/10_000.0
            elif delta < 0:
                # Whole-share accounts cannot realize fractional target sizes.
                target_whole_quantity=max(0,math.floor(target_quantity+0.5))
                if action == 0 and current_quantity > 0:
                    target_whole_quantity=min(target_whole_quantity,current_quantity-1)
                self.state["pending"][symbol] = {"date": timestamp, "action": "SELL",
                                                  "budget": current_quantity-target_whole_quantity,
                                                  "decision_id": f"{timestamp}|{symbol}"}
                queued_decisions.add(f"{timestamp}|{symbol}")
        for currency, signals in buys.items():
            cash = float(self.state["books"][currency]["cash"])
            if cash <= 0:
                continue
            valid_signals=[(symbol,score) for symbol,score in signals if score>0]
            if not valid_signals:
                continue
            if full_weights is None:
                total=sum(score for _,score in valid_signals)
                targets=[(symbol,cash*score/total,None) for symbol,score in valid_signals]
            else:
                # Preserve the model's BUY allocation pool, then round it to
                # whole shares against estimated spread, slippage, and fees.
                # Largest fractional shares get first claim on available cash.
                unit_costs={}
                for symbol,_ in valid_signals:
                    spread=max(0.0,min(buy_spreads[symbol]/2.0,0.025))
                    unit_costs[symbol]=buy_prices[symbol]*(1.0+spread+self.slippage)*(1.0+self.fee)
                target_cost=sum(buy_quantities[symbol]*unit_costs[symbol]
                                for symbol,_ in valid_signals)
                scale=min(1.0,cash/target_cost) if target_cost>0 else 0.0
                base=[]; remaining_cash=cash; remaining_target_cost=target_cost*scale
                for symbol,budget in sorted(valid_signals,key=lambda item:(-item[1],item[0])):
                    unit_cost=unit_costs[symbol]
                    exact=buy_quantities[symbol]*scale
                    wanted=math.floor(exact+1e-12)
                    units=min(wanted,math.floor(remaining_cash/unit_cost))
                    if units:
                        spent=units*unit_cost
                        remaining_cash-=spent
                        remaining_target_cost-=spent
                    base.append([symbol,budget*scale,units,wanted,
                                 exact-math.floor(exact+1e-12),unit_cost])
                for row in sorted(base,key=lambda item:(-item[4],-item[1],item[0])):
                    symbol,scaled_budget,units,wanted,remainder,unit_cost=row
                    if (remainder>1e-12 and units<wanted+1
                            and remaining_cash+1e-8>=unit_cost
                            and remaining_target_cost+1e-8>=unit_cost):
                        row[2]+=1; remaining_cash-=unit_cost; remaining_target_cost-=unit_cost
                targets=[(symbol,scaled_budget,units if units>0 else None)
                         for symbol,scaled_budget,units,_,_,_ in base if units>0]
            for symbol,budget,requested_quantity in targets:
                existing=self.state["pending"].get(symbol)
                if existing and existing.get("date")==timestamp and existing.get("action")=="SELL":
                    continue
                order={"date":timestamp,"action":"BUY","budget":budget}
                order["decision_id"]=f"{timestamp}|{symbol}"
                if requested_quantity is not None:
                    order["requested_quantity"]=requested_quantity
                self.state["pending"][symbol]=order
                queued_decisions.add(order["decision_id"])
        return queued_decisions

    def snapshot(self) -> dict:
        books = {}
        for currency, book in self.state["books"].items():
            equity = self._equity(currency)
            books[currency] = {**book, "equity": equity,
                "net_pnl": equity - float(book["initial_cash"]),
                "holdings_value": equity - float(book["cash"])}
        return {**self.state, "books": books,
                "execution": "next completed bar close; cash-only equities and ETFs",
                "allocation": "per-currency tradable model weights; action-gated whole-share orders with cost-aware cash limits",
                "kr_sell_tax_assumption": KR_SELL_TAX_ASSUMPTION}

    def save(self) -> None:
        if self.path is not None:
            atomic_json(self.snapshot(), self.path)
