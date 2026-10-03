"""TradingMoE output -> the existing paper ledger and delayed reward engine.

No broker API, model loading, account reset or optimizer lives in this bridge.
"""
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np

from .paper_account import PaperAccount, _currency
from .replay_store import GlobalReplayBuffer
from .rewards import _RewardMixin
from .experience import REWARD_VERSION


class TradingMoEPaper(_RewardMixin):
    def __init__(self, state_dir, *, fee=.001, slippage=.0001, credit_seconds=3600):
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        self.paper_account = PaperAccount(state_dir / "paper_account.json", fee, slippage)
        self.replay = GlobalReplayBuffer(journal_path=state_dir / "replay.sqlite3", dual_learning=False)
        self.metrics = {}
        self.horizon_kind, self.horizon_amount = "seconds", int(credit_seconds)
        if self.horizon_amount < 1:
            raise ValueError("reward credit interval must be positive")
        # Existing reward engine uses this pending kind for non-Champion roles.
        # It is isolated in this MoE's own journal, not Candidate's live DB.
        self.pending = self.replay.load_pending("candidate_portfolio")

    def advance(self, panel, index, *, enabled=True):
        fills = self.paper_account.process_bar(panel, index, enabled=enabled)
        self.pending = self._mature_portfolio(self.pending, panel, index, fills,
                                             account=self.paper_account, origin_model="trading_moe")
        self.replay.save_pending_kind("candidate_portfolio", self.pending)
        self.paper_account.save()
        return fills

    def submit(self, result, panel, index, *, paper_executable=False):
        """Explicit paper consent; executable=False remains the live-order guard."""
        if not paper_executable:
            return {"paper_executable":False, "orders":[]}
        output = result["trading_output"]
        stamp = str(panel.dates[index])
        if self.paper_account.state["last_timestamp"] and np.datetime64(self.paper_account.state["last_timestamp"]) > panel.dates[index]:
            raise ValueError("cannot submit a stale decision to a later paper account")
        if np.datetime64(result["as_of"]) != panel.dates[index]:
            raise ValueError("inference as-of must match the decision bar")
        actions_by_symbol = output["actions"]
        weights = output["target_weights"]
        cash = output["cash_weights_by_currency"]
        if set(actions_by_symbol) != set(weights):
            raise ValueError("policy action/weight identities differ")
        for symbol in weights:
            if actions_by_symbol[symbol] not in ("BUY", "HOLD", "SELL"):
                raise ValueError("unknown policy action")
            if not math.isfinite(weights[symbol]) or not 0 <= weights[symbol] <= 1:
                raise ValueError("invalid target weight")
        # A separate allocation vector per currency preserves independent ledgers.
        tradable = {}
        for j, symbol in enumerate(panel.symbols):
            currency = _currency(*panel.groups[symbol])
            if symbol in weights and currency and panel.observed[index,j]:
                tradable.setdefault(currency, []).append(j)
        paper_cash = dict(cash)
        output_currencies = result.get("currencies") or {s:"USD" for s in weights}
        tradable_symbols = {panel.symbols[j] for indices in tradable.values() for j in indices}
        for symbol, weight in weights.items():
            if symbol not in tradable_symbols:
                currency = output_currencies[symbol]
                paper_cash[currency] = paper_cash.get(currency,0) + weight
        for currency, indices in tradable.items():
            value = paper_cash.get(currency)
            if value is None or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid paper cash allocation")
            if abs(sum(weights[panel.symbols[j]] for j in indices) + value - 1) > 1e-5:
                raise ValueError("incomplete per-currency policy allocation")
        pstate, astate = self.paper_account.model_inputs(panel,index)
        queued = set()
        for currency, indices in tradable.items():
            sub = SimpleNamespace(dates=panel.dates, symbols=[panel.symbols[j] for j in indices],
                groups=panel.groups, observed=panel.observed[:,indices], closes=panel.closes[:,indices],
                features=panel.features[:,indices])
            acts = np.asarray([{"SELL":0,"HOLD":1,"BUY":2}[actions_by_symbol[s]] for s in sub.symbols])
            probs = np.eye(3,dtype=np.float32)[acts]
            allocation = [weights[s] for s in sub.symbols] + [paper_cash[currency]]
            queued |= self.paper_account.queue_decisions(sub,index,probs,True,allocation,acts)
        existing = {d["decision_id"] for d in self.pending}
        start = max(0,index-127)
        decision_inputs = None
        for indices in tradable.values():
            for j in indices:
                symbol = panel.symbols[j]
                decision_id = f"{stamp}|{symbol}"
                if decision_id in existing:
                    continue
                if decision_inputs is None:
                    # All symbols in this decision observe the same owned input
                    # block. Outcome processing reads these arrays; it never
                    # mutates them. Keep one copy, independent of future bars.
                    decision_inputs = {
                        "features":panel.features[start:index+1].copy(),
                        "valid_mask":panel.observed[start:index+1].copy(),
                        "symbol_ids":panel.symbol_ids.copy(),
                        "market_ids":panel.market_ids.copy(),
                        "asset_ids":panel.asset_ids.copy(),
                        "portfolio_state":np.asarray(pstate,dtype=np.float32),
                        "account_state":np.asarray(astate,dtype=np.float32),
                    }
                self.pending.append({"timestamp":stamp,"decision_id":decision_id,"symbol":symbol,
                    "symbol_index":j,"input_symbols":list(panel.symbols),
                    "action":{"SELL":0,"HOLD":1,"BUY":2}[actions_by_symbol[symbol]],
                    **decision_inputs,"reward_version":REWARD_VERSION,
                    "entry_price":float(panel.closes[index,j]),"bars_elapsed":0,
                    "equity_before":self.paper_account.normalized_equity(),
                    "symbol_pnl_before":self.paper_account.symbol_net_pnl(symbol),
                    "fill_expected":decision_id in queued,"is_validation":False,
                    "moe_run_id":result.get("run_id")})
        self.replay.save_pending_kind("candidate_portfolio",self.pending)
        self.paper_account.save()
        return {"paper_executable":True,"live_executable":False,"orders":sorted(queued),
                "pending_orders":dict(self.paper_account.state["pending"])}
