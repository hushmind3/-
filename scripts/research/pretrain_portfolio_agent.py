"""Pretrain portfolio selection, sizing and position management by historical replay.

The 0.5B champion is read-only. A new candidate starts from its exact weights,
adds only small portfolio-state adapters, and trains those adapters on the
chronological train split. Validation compares the same shadow portfolio engine
before/after; the test split and promotion path are deliberately untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stockrl.global_transformer import GlobalMarketTransformer, TransformerConfig, load_compatible_state_dict
from stockrl.market_panel import TIME_SCALE_NAMES
from stockrl.market_training import (CommonMarketTrainingLoader, ContextConditionedTransformer,
                                     NormalizedBarAdapter, CONTEXT_FEATURES)
from stockrl.market_panel import GLOBAL_FEATURES
from stockrl.global_transformer import parameter_count
from stockrl.market_training import stable_id

DEFAULT_CHAMPION = ROOT / "runtime-global-cuda-final/champion.pt"
DEFAULT_SOURCE = ROOT / "data/global_market_daily.csv"
DEFAULT_CACHE = ROOT / "runtime-global-market-training/shared-portfolio-daily-cache"
DEFAULT_RUNTIME = ROOT / "runtime-global-market-training/portfolio-pretraining"
SEQ = 128
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0001


class NumpyJSONEncoder(json.JSONEncoder):
    """Serialize NumPy scalar/array metrics emitted by market replays."""
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

# Indicators and yields are observation/context only. The simulator trades
# assets for which the source identifies an executable product.
TRADEABLE_ASSET_CLASSES = {"equity", "currency", "commodity_future", "commodity_etf",
                           "index_future", "index_etf", "bond_etf", "sector_etf"}


def instrument_map_for_daily(old_map: dict, daily_map: dict) -> tuple[dict, dict[int, int]]:
    by_ticker: dict[str, list[int]] = {}
    for key, ix in old_map.items():
        by_ticker.setdefault(key.rsplit("|", 1)[-1], []).append(int(ix))
    combined = dict(old_map)
    daily_map_to_model: dict[int, int] = {}
    next_id = max(map(int, old_map.values())) + 1
    for key, local_id in sorted(daily_map.items(), key=lambda kv: kv[1]):
        ticker = key.rsplit("|", 1)[-1]
        matches = by_ticker.get(ticker, [])
        if len(matches) == 1:
            target = matches[0]
        else:
            # Distinct daily market instruments get dedicated embedding rows.
            target = next_id
            combined[key] = target
            next_id += 1
        daily_map_to_model[int(local_id)] = target
    return combined, daily_map_to_model


def create_candidate(champion_path: Path, candidate_path: Path, daily_map: dict):
    source = torch.load(champion_path, map_location="cpu", weights_only=False)
    old_cfg = TransformerConfig(**source["config"])
    combined_map, local_to_model = instrument_map_for_daily(source["symbol_map"], daily_map)
    cfg = replace(old_cfg, max_symbols=max(old_cfg.max_symbols, len(combined_map)))
    model = ContextConditionedTransformer(GlobalMarketTransformer(cfg).half())
    model.context_policy.float(); model.context_value.float()
    state = model.state_dict()
    old_state = source["state_dict"]
    old_embed = old_state["backbone.symbol_embedding.weight"]
    with torch.no_grad():
        state["backbone.symbol_embedding.weight"][:old_embed.shape[0]].copy_(old_embed)
        if state["backbone.symbol_embedding.weight"].shape[0] > old_embed.shape[0]:
            state["backbone.symbol_embedding.weight"][old_embed.shape[0]:].copy_(
                old_embed.float().mean(0, keepdim=True).half())
    for key, value in old_state.items():
        if key != "backbone.symbol_embedding.weight":
            state[key] = value
    load_compatible_state_dict(model, state, strict=True)
    payload = dict(source)
    payload.update({"config": asdict(cfg), "state_dict": model.state_dict(),
        "symbol_map": combined_map, "context_features": list(CONTEXT_FEATURES),
        "market_context_model": True, "source_champion": str(champion_path),
        "source_champion_sha256": file_sha256(champion_path),
        "portfolio_state_features": ["position_weight", "normalized_quantity",
            "unrealized_return", "unrealized_pnl_fraction", "holding_age", "cash_available"],
        "portfolio_account_features": ["cash_fraction", "equity_over_start",
            "realized_pnl_over_start", "drawdown", "active_position_fraction", "cumulative_turnover"],
        "portfolio_action_head": "3-way policy residual + cash-inclusive per-asset softmax allocation",
        "promotion": "not performed"})
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, candidate_path)
    del source, model
    return payload, local_to_model


def file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_economic_panels(path: Path, loader):
    """Build USD notional, spread proxy and ADV from the real daily source."""
    raw = pd.read_csv(path, dtype={"symbol": "string", "market": "string", "asset_class": "string"})
    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    raw["time_key"] = raw["date"].astype("int64") // 1_000_000_000
    raw["instrument"] = raw["market"].astype(str) + "|" + raw["asset_class"].astype(str) + "|" + raw["symbol"].astype(str)
    t_count, n_count = len(loader.time_keys), len(loader.symbol_names)
    raw_close = np.asarray(loader.closes, np.float64)
    raw_high = raw_close.copy(); raw_low = raw_close.copy()
    volume = np.zeros((t_count, n_count), np.float64)
    time_ids = {int(k): i for i, k in enumerate(loader.time_keys)}
    for row in raw.itertuples(index=False):
        sid = loader.symbol_map.get(str(row.instrument))
        tid = time_ids.get(int(row.time_key))
        if sid is None or tid is None: continue
        volume[tid, sid] = max(float(row.volume or 0), 0.0)
        raw_high[tid, sid] = float(row.high)
        raw_low[tid, sid] = float(row.low)
    by_ticker = {name.rsplit("|", 1)[-1]: ix for ix, name in enumerate(loader.symbol_names)}
    def quote(ticker):
        ix = by_ticker.get(ticker)
        return raw_close[:, ix] if ix is not None else np.full(t_count, np.nan)
    krw_per_usd = quote("KRW=X")
    jpy_per_usd = quote("JPY=X")
    usd_per_eur = quote("EURUSD=X")
    usd_per_gbp = quote("GBPUSD=X")
    factors = np.ones((t_count, n_count), np.float64)
    tradable = np.zeros(n_count, bool)
    lots = np.ones(n_count, np.float64)
    tax = np.zeros(n_count, np.float64)
    tickers = []
    for sid, name in enumerate(loader.symbol_names):
        market, asset, ticker = name.split("|", 2)
        tickers.append(ticker)
        if market == "KR": factors[:, sid] = 1.0 / np.maximum(krw_per_usd, 1e-12)
        elif market in {"DE", "FR", "EU"}: factors[:, sid] = usd_per_eur
        elif market == "UK": factors[:, sid] = usd_per_gbp
        elif market == "JP": factors[:, sid] = 1.0 / np.maximum(jpy_per_usd, 1e-12)
        elif market == "FX":
            first = next((x for x in raw_close[:, sid] if np.isfinite(x) and x > 0), 1.0)
            # Spot FX has no reported volume in this source; preserve its
            # percentage path in normalized notional units.
            factors[:, sid] = 1.0 / first
        else: factors[:, sid] = 1.0
        has_fx = market in {"US", "KR", "UK", "DE", "FR", "EU", "JP", "FX"}
        tradable[sid] = asset in TRADEABLE_ASSET_CLASSES and has_fx
        if asset == "currency": lots[sid] = .01
        if asset == "equity":
            # Explicit, configurable research assumption, charged on sale.
            tax[sid] = .0010
    with np.errstate(invalid="ignore", divide="ignore"):
        prices = raw_close * factors
    # Quotes stay stale on venue holidays, but are never backfilled into the
    # past; observations still control whether an order can execute.
    prices = pd.DataFrame(prices).ffill().to_numpy(np.float64)
    # Market return/accounting panels are deliberately separate from model
    # features, which remain the source's original 17-dimensional schema.
    spread = np.clip(.05 * np.maximum(raw_high - raw_low, 0.0) /
                     np.maximum(raw_close, 1e-12), .0001, .02)
    adv = prices * volume
    for sid, name in enumerate(loader.symbol_names):
        market, asset, ticker = name.split("|", 2)
        if market == "FX" and asset == "currency" and not np.any(volume[:, sid] > 0):
            adv[:, sid] = np.inf
        if not tradable[sid]: adv[:, sid] = 0.0
    observed = np.asarray(loader.observed, bool)
    if not np.isfinite(prices[:, tradable][observed[:, tradable]]).all():
        raise ValueError("tradable daily observations are missing required FX conversion data")
    return {"prices": prices.astype(np.float64), "volume": volume,
            "adv": adv.astype(np.float64), "spread": spread.astype(np.float64),
            "tradable": tradable, "lots": lots, "sell_tax": tax,
            "symbols": tickers}


class Portfolio:
    """Cash account with lots, side costs, sell tax, spread, impact and ADV caps."""
    def __init__(self, n: int, initial_equity: float = 1.0, lots=None,
                 sell_tax=None, commission: float = .0005,
                 participation: float = .01):
        self.initial_equity = float(initial_equity)
        self.cash = float(initial_equity)
        self.units = np.zeros(n, np.float64)
        self.average_cost = np.zeros(n, np.float64)
        self.holding_age = np.zeros(n, np.float64)
        self.realized_net = 0.0
        self.turnover_total = 0.0
        self.peak_equity = float(initial_equity)
        self.last_prices = np.zeros(n, np.float64)
        self.last_fee = 0.0
        self.last_tax = 0.0
        self.last_spread = 0.0
        self.last_slippage = 0.0
        self.last_action_counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
        self.lots = np.ones(n, np.float64) if lots is None else np.asarray(lots, np.float64)
        self.sell_tax = np.zeros(n, np.float64) if sell_tax is None else np.asarray(sell_tax, np.float64)
        self.commission = float(commission)
        self.participation = float(participation)

    def equity(self, marks: np.ndarray) -> float:
        px = np.where(np.isfinite(marks) & (marks > 0), marks, self.last_prices)
        return float(self.cash + np.sum(self.units * px))

    def observe(self, prices: np.ndarray, cost_rate: np.ndarray,
                adv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px = np.where(np.isfinite(prices) & (prices > 0), prices, self.last_prices)
        equity = max(self.equity(px), 1e-12)
        current_value = self.units * px
        weights = current_value / equity
        has = self.units > 1e-12
        unrealized = np.zeros_like(weights)
        unrealized[has] = px[has] / np.maximum(self.average_cost[has], 1e-12) - 1.0
        unrealized_pnl = np.zeros_like(weights)
        unrealized_pnl[has] = self.units[has] * (px[has] - self.average_cost[has]) / equity
        state = np.stack((weights,
            self.units * px / self.initial_equity,
            unrealized,
            unrealized_pnl,
            np.minimum(self.holding_age / 30.0, 1.0),
            np.full_like(weights, max(self.cash, 0.0) / equity),
            np.asarray(cost_rate, np.float64),
            np.clip(np.asarray(adv, np.float64) / max(self.initial_equity, 1e-12), 0.0, 100.0)), axis=-1)
        account = np.asarray((max(self.cash, 0.0) / equity,
            equity / self.initial_equity,
            self.realized_net / self.initial_equity,
            max(0.0, (self.peak_equity - equity) / max(self.peak_equity, 1e-12)),
            float(has.mean()),
            self.turnover_total / self.initial_equity,
            np.log10(max(self.initial_equity, 1.0)) / 8.0,
            float(np.mean(cost_rate))), np.float32)
        return state.astype(np.float32), account

    def rebalance_and_mark(self, prices: np.ndarray, next_prices: np.ndarray,
                           observed: np.ndarray, next_observed: np.ndarray,
                           target_weights: np.ndarray, tradable: np.ndarray,
                           adv: np.ndarray, spread: np.ndarray) -> dict:
        px = np.asarray(prices, np.float64)
        nxt = np.asarray(next_prices, np.float64)
        mark = np.where(np.isfinite(px) & (px > 0), px, self.last_prices)
        equity_before = self.equity(mark)
        active = np.asarray(observed, bool) & np.asarray(tradable, bool) & np.isfinite(mark) & (mark > 0)
        current_values = self.units * mark
        current_weights = current_values / max(equity_before, 1e-12)
        frozen = ~active
        available = max(0.0, 1.0 - float(current_weights[frozen].sum()))
        raw = np.maximum(np.asarray(target_weights[:-1], np.float64), 0.0)
        raw[~active] = 0.0
        cash_raw = max(float(target_weights[-1]), 0.0)
        denom = float(raw.sum() + cash_raw)
        if denom <= 1e-15:
            raw[:] = 0.0; cash_raw = 1.0; denom = 1.0
        desired_weights = current_weights.copy()
        desired_weights[active] = available * raw[active] / denom
        desired_values = desired_weights * equity_before
        delta_value = desired_values - current_values
        self.last_fee = self.last_tax = self.last_spread = self.last_slippage = 0.0
        realized_delta = 0.0
        trades = np.zeros_like(self.units)
        adv = np.asarray(adv, np.float64)
        spread = np.asarray(spread, np.float64)
        # Reductions execute first and their proceeds are the only new buying power.
        sells = np.flatnonzero(active & (delta_value < -1e-12))
        for i in sells:
            units = min(self.units[i], -delta_value[i] / mark[i])
            units = np.floor(units / self.lots[i]) * self.lots[i]
            cap = self.participation * adv[i] if np.isfinite(adv[i]) else np.inf
            units = min(units, cap / mark[i])
            units = np.floor(units / self.lots[i]) * self.lots[i]
            if units <= 0: continue
            mid_notional = units * mark[i]
            impact = min(.02, .001 * math.sqrt(mid_notional / max(adv[i], 1.0))) if np.isfinite(adv[i]) else 0.0
            half_spread = max(0.0, spread[i] / 2.0)
            slip_rate = SLIPPAGE_RATE + impact
            fill = mark[i] * (1.0 - min(.05, half_spread + slip_rate))
            fee = units * fill * self.commission
            tax = units * fill * self.sell_tax[i]
            self.cash += units * fill - fee - tax
            realized = units * (fill - self.average_cost[i]) - fee - tax
            realized_delta += realized; self.realized_net += realized
            self.units[i] -= units; trades[i] -= units
            self.last_fee += fee; self.last_tax += tax
            self.last_spread += mid_notional * half_spread
            self.last_slippage += mid_notional * slip_rate
            if self.units[i] <= 1e-12:
                self.units[i] = 0.0; self.average_cost[i] = 0.0; self.holding_age[i] = 0.0

        buys = np.flatnonzero(active & (delta_value > 1e-12))
        requested = np.zeros_like(self.units)
        required_cash = 0.0
        for i in buys:
            units = np.floor((delta_value[i] / mark[i]) / self.lots[i]) * self.lots[i]
            cap = self.participation * adv[i] if np.isfinite(adv[i]) else np.inf
            requested[i] = np.floor(min(units, cap / mark[i]) / self.lots[i]) * self.lots[i]
            notional = requested[i] * mark[i]
            impact = min(.02, .001 * math.sqrt(notional / max(adv[i], 1.0))) if np.isfinite(adv[i]) else 0.0
            one_way = min(.05, spread[i] / 2.0 + SLIPPAGE_RATE + impact)
            required_cash += notional * (1.0 + one_way) * (1.0 + self.commission)
        scale = min(1.0, max(self.cash, 0.0) / max(required_cash, 1e-12))
        for i in buys:
            units = np.floor((requested[i] * scale) / self.lots[i]) * self.lots[i]
            if units <= 0: continue
            mid_notional = units * mark[i]
            impact = min(.02, .001 * math.sqrt(mid_notional / max(adv[i], 1.0))) if np.isfinite(adv[i]) else 0.0
            half_spread = max(0.0, spread[i] / 2.0)
            slip_rate = SLIPPAGE_RATE + impact
            fill = mark[i] * (1.0 + min(.05, half_spread + slip_rate))
            gross_fill = units * fill
            fee = gross_fill * self.commission
            old_units, old_basis = self.units[i], self.average_cost[i]
            self.cash -= gross_fill + fee
            self.units[i] += units
            self.average_cost[i] = (old_units * old_basis + gross_fill + fee) / self.units[i]
            trades[i] += units
            self.last_fee += fee
            self.last_spread += mid_notional * half_spread
            self.last_slippage += mid_notional * slip_rate

        # A bounded correction absorbs only floating point dust; large negative
        # cash indicates a real duplicate-spend bug and must fail the run.
        if self.cash < -1e-9:
            raise AssertionError(f"negative cash after rebalance: {self.cash}")
        self.cash = max(self.cash, 0.0)
        self.turnover_total += float(np.sum(np.abs(trades * mark)))
        self.holding_age[self.units > 1e-12] += 1.0
        self.holding_age[self.units <= 1e-12] = 0.0
        realized_ret = np.zeros_like(mark)
        has_ret = (np.asarray(next_observed, bool) & np.isfinite(nxt) & (nxt > 0) &
                   np.isfinite(mark) & (mark > 0))
        realized_ret[has_ret] = nxt[has_ret] / mark[has_ret] - 1.0
        # Assets closed tomorrow keep their last valid mark; newly stale marks
        # produce zero return until the venue supplies a fresh close.
        next_mark = np.where(np.isfinite(nxt) & (nxt > 0), nxt, mark)
        equity_after = self.equity(next_mark)
        actual_reward = (equity_after - equity_before) / max(equity_before, 1e-12)
        self.last_prices = next_mark.copy()
        self.peak_equity = max(self.peak_equity, equity_after)
        self.last_action_counts = {
            "BUY": int(np.count_nonzero(trades > 1e-12)),
            "SELL": int(np.count_nonzero(trades < -1e-12)),
            "HOLD": int(np.count_nonzero(active & (np.abs(trades) <= 1e-12) & (self.units > 1e-12))),
        }
        return {"equity_before": equity_before, "equity_after": equity_after,
            "reward": actual_reward, "realized_pnl_delta": realized_delta,
            "fee": self.last_fee, "tax": self.last_tax,
            "spread_cost": self.last_spread, "slippage": self.last_slippage,
            "turnover": float(np.sum(np.abs(trades * mark))),
            "positions": int(np.count_nonzero(self.units > 1e-8)),
            "cash": self.cash, "action_counts": dict(self.last_action_counts),
            "next_returns": realized_ret, "target_weights": desired_weights}


def smoke_environment() -> dict:
    env = Portfolio(3, 1.0, lots=np.full(3, .001), sell_tax=np.array([.001, 0., 0.]),
                    commission=.0005, participation=.5)
    p0 = np.array([10., 20., 30.]); p1 = np.array([11., 20., 30.])
    active = np.ones(3, bool); tradable = np.ones(3, bool)
    def tick(target, prices, future):
        equity0 = env.equity(prices)
        out = env.rebalance_and_mark(prices, future, active, active, np.asarray(target), tradable,
                                     np.full(3, np.inf), np.full(3, .002))
        if abs((out["equity_after"] - out["equity_before"]) / out["equity_before"] - out["reward"]) > 1e-10:
            raise AssertionError("reported reward differs from portfolio equity change")
        if env.cash < -1e-9 or not np.isfinite(env.cash): raise AssertionError("cash accounting invalid")
        return out
    buy = tick([.4, 0., 0., .6], p0, p1)
    if env.units[0] <= 0 or buy["fee"] <= 0 or buy["slippage"] <= 0: raise AssertionError("buy/cost test failed")
    # After the first mark, use the exact drifted weight so this is a true hold.
    hold_weight = env.units[0] * p1[0] / env.equity(p1)
    hold = tick([hold_weight, 0., 0., 1.0 - hold_weight], p1, p1)
    if hold["turnover"] > 1e-8: raise AssertionError("hold generated unexplained turnover")
    replace_out = tick([0., .3, 0., .7], p1, p1)
    if env.units[0] > 1e-10 or env.units[1] <= 0 or replace_out["action_counts"]["SELL"] == 0 or replace_out["action_counts"]["BUY"] == 0:
        raise AssertionError("position replace test failed")
    close = tick([0., 0., 0., 1.], p1, p1)
    if np.count_nonzero(env.units) or close["positions"]: raise AssertionError("liquidation test failed")
    return {"buy_cost": buy["fee"] + buy["spread_cost"] + buy["slippage"],
            "sell_tax": replace_out["tax"], "hold_turnover": hold["turnover"],
            "replace_actions": replace_out["action_counts"], "closed_positions": close["positions"],
            "cash_never_negative": True, "reward_matches_equity_delta": True}


def make_context(loader, end: int, local_ids: np.ndarray, map_ids: np.ndarray,
                 env: Portfolio, prices: np.ndarray, cost_rate: np.ndarray,
                 adv: np.ndarray, device):
    tix = np.arange(end - SEQ + 1, end + 1, dtype=np.int64)
    x = np.asarray(loader.features[tix[:, None], local_ids[None, :]], np.float16)[None]
    mask = np.asarray(loader.observed[tix[:, None], local_ids[None, :]], bool)[None]
    ctx = np.asarray(loader.context[tix], np.float32)[None]
    pstate, astate = env.observe(prices[local_ids], cost_rate[local_ids], adv[local_ids])
    return (torch.as_tensor(x, device=device),
            torch.as_tensor(map_ids[None], dtype=torch.long, device=device),
            torch.as_tensor(np.asarray(loader.market_ids[local_ids], np.int64)[None], dtype=torch.long, device=device),
            torch.as_tensor(np.asarray(loader.asset_ids[local_ids], np.int64)[None], dtype=torch.long, device=device),
            torch.as_tensor(mask, dtype=torch.bool, device=device),
            torch.as_tensor(ctx, dtype=torch.float32, device=device),
            torch.as_tensor(pstate[None], dtype=torch.float32, device=device),
            torch.as_tensor(astate[None], dtype=torch.float32, device=device))


def mask_allocations(allocation: torch.Tensor, active: torch.Tensor,
                     current_weight: torch.Tensor) -> torch.Tensor:
    n = allocation.shape[-1] - 1
    scores = torch.log(allocation[0].clamp_min(1e-12))
    scores[:n] = scores[:n].masked_fill(~active, -1e9)
    dist = torch.softmax(scores, dim=-1)
    frozen_weight = current_weight[~active].sum()
    available = (1.0 - frozen_weight).clamp(0.0, 1.0)
    out = current_weight.clone()
    out[active] = dist[:n][active] * available
    cash = dist[n] * available
    return torch.cat((out, cash[None]))


def run_episode(model, loader, market, local_ids, model_ids, tradable, split, train,
                optimizer=None, max_steps=None, seed=0, episode_id=0,
                initial_equity=10_000.0, commission=.0005, participation=.01,
                device=torch.device("cuda:0")) -> dict:
    lo, hi = map(int, loader.splits[split])
    start = max(lo, SEQ - 1)
    stops = np.arange(start, hi - 1, dtype=np.int64)
    if max_steps is not None and len(stops) > max_steps:
        # A contiguous chronological smoke slice preserves the environment's
        # position and price path while keeping the first check short.
        stops = stops[:max_steps]
    if not len(stops): raise RuntimeError(f"split {split} has no replay steps")
    env = Portfolio(len(local_ids), initial_equity,
        lots=market["lots"][local_ids], sell_tax=market["sell_tax"][local_ids],
        commission=commission, participation=participation)
    start_t = time.perf_counter()
    metrics = {"steps": 0, "net_return": 0.0, "fees": 0.0, "taxes": 0.0,
               "spread_cost": 0.0, "slippage": 0.0,
               "turnover": 0.0, "buy": 0, "sell": 0, "hold": 0,
               "position_count_sum": 0, "cash_min": float("inf"),
               "pnl_equity_identity_error_max": 0.0, "loss": 0.0}
    model.train(train)
    for day_ix, t in enumerate(stops):
        prices = market["prices"][t]
        future = market["prices"][t + 1]
        observed = np.asarray(loader.observed[t], bool)
        next_obs = np.asarray(loader.observed[t + 1], bool)
        adv = market["adv"][t]
        spread = market["spread"][t]
        impact_est = np.where(np.isfinite(adv), np.minimum(.02, .001 * np.sqrt(
            np.maximum(initial_equity, 0.0) / np.maximum(adv, 1.0))), 0.0)
        cost_rate = commission + market["sell_tax"] + spread / 2.0 + SLIPPAGE_RATE + impact_est
        active_np = observed[local_ids] & tradable
        if not active_np.any(): continue
        current_values = env.units * np.where(np.isfinite(prices[local_ids]), prices[local_ids], env.last_prices)
        current_weight = current_values / max(env.equity(prices[local_ids]), 1e-12)
        args = make_context(loader, int(t), local_ids, model_ids, env, prices,
                            cost_rate, adv, device)
        x, sid, mid, aid, mask, ctx, pstate, astate = args
        active = torch.as_tensor(active_np, device=device, dtype=torch.bool)
        current_t = torch.as_tensor(current_weight, dtype=torch.float32, device=device)
        if train:
            with torch.autocast("cuda", dtype=torch.float16):
                logits, values, alloc = model(x, sid, mid, aid, mask, ctx,
                    torch.tensor([3], device=device), pstate, astate, True)
            target = mask_allocations(alloc, active, current_t)
            ret_np = np.zeros(len(local_ids), np.float32)
            good = next_obs[local_ids] & (prices[local_ids] > 0) & (future[local_ids] > 0)
            ret_np[good] = future[local_ids][good] / prices[local_ids][good] - 1.0
            ret = torch.as_tensor(ret_np, dtype=torch.float32, device=device)
            delta_weight = target[:-1] - current_t
            base_cost = (commission + market["spread"][t, local_ids] / 2.0 +
                         SLIPPAGE_RATE + impact_est[local_ids])
            base_cost_t = torch.as_tensor(base_cost, dtype=torch.float32, device=device)
            tax_t = torch.as_tensor(market["sell_tax"][local_ids], dtype=torch.float32, device=device)
            cost = torch.sum(torch.relu(delta_weight) * base_cost_t +
                             torch.relu(-delta_weight) * (base_cost_t + tax_t))
            surrogate = torch.sum(target[:-1] * ret) - cost
            loss = -surrogate
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
            metrics["loss"] += float(loss.detach().cpu())
            target_np = target.detach().cpu().numpy()
        else:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                logits, values, alloc = model(x, sid, mid, aid, mask, ctx,
                    torch.tensor([3], device=device), pstate, astate, True)
                target = mask_allocations(alloc, active, current_t)
            target_np = target.cpu().numpy()
        out = env.rebalance_and_mark(prices[local_ids], future[local_ids],
            observed[local_ids], next_obs[local_ids], target_np, tradable,
            adv[local_ids], spread[local_ids])
        if not math.isclose(out["equity_after"] - out["equity_before"],
                            out["reward"] * out["equity_before"], rel_tol=1e-8, abs_tol=1e-9):
            raise AssertionError("portfolio reward and actual account PnL disagree")
        metrics["steps"] += 1
        metrics["net_return"] += out["reward"]
        metrics["fees"] += out["fee"]
        metrics["taxes"] += out["tax"]
        metrics["spread_cost"] += out["spread_cost"]
        metrics["slippage"] += out["slippage"]
        metrics["turnover"] += out["turnover"]
        metrics["buy"] += out["action_counts"]["BUY"]
        metrics["sell"] += out["action_counts"]["SELL"]
        metrics["hold"] += out["action_counts"]["HOLD"]
        metrics["position_count_sum"] += out["positions"]
        metrics["cash_min"] = min(metrics["cash_min"], out["cash"])
        metrics["pnl_equity_identity_error_max"] = max(metrics["pnl_equity_identity_error_max"],
             abs((out["equity_after"] - out["equity_before"]) - out["reward"] * out["equity_before"]))
    elapsed = time.perf_counter() - start_t
    metrics.update({"split": split, "episode": episode_id, "initial_equity": initial_equity,
                    "final_equity": env.equity(env.last_prices),
                    "net_return_compounded": env.equity(env.last_prices) / initial_equity - 1.0,
                    "avg_positions": metrics["position_count_sum"] / max(metrics["steps"], 1),
                    "avg_net_return_per_step": metrics["net_return"] / max(metrics["steps"], 1),
                    "mean_loss": metrics["loss"] / max(metrics["steps"], 1),
                    "seconds": elapsed, "steps_per_second": metrics["steps"] / max(elapsed, 1e-9),
                    "total_realized_net_pnl": env.realized_net,
                    "cash_nonnegative": metrics["cash_min"] >= -1e-9})
    return metrics


def run_joint_training(model, loader, market, local_ids, model_ids, tradable,
                       seed_sizes, optimizer, seed, commission_range=(.0002, .001),
                       participation_range=(.005, .02), max_steps=None,
                       device=torch.device("cuda:0")) -> list[dict]:
    """Train capital tiers together so one tier cannot overwrite another.

    Every date, each account contributes a separate realized cost-aware
    objective. Their losses are averaged into one optimizer update, then all
    portfolios execute that date's allocations. The policy remains free to
    choose assets or cash; no HOLD/cash target is prescribed.
    """
    lo, hi = map(int, loader.splits["train"])
    stops = np.arange(max(lo, SEQ - 1), hi - 1, dtype=np.int64)
    if max_steps is not None:
        stops = stops[:max_steps]
    if not len(stops):
        raise RuntimeError("train split has no joint replay steps")
    rng = np.random.default_rng(seed)
    scenarios = []
    for ix, capital in enumerate(seed_sizes):
        commission = float(rng.uniform(*commission_range))
        participation = float(rng.uniform(*participation_range))
        env = Portfolio(len(local_ids), capital, lots=market["lots"][local_ids],
            sell_tax=market["sell_tax"][local_ids], commission=commission,
            participation=participation)
        metrics = {"steps": 0, "fees": 0.0, "taxes": 0.0, "spread_cost": 0.0,
            "slippage": 0.0, "turnover": 0.0, "buy": 0, "sell": 0, "hold": 0,
            "position_count_sum": 0, "cash_min": float("inf"), "loss": 0.0,
            "pnl_equity_identity_error_max": 0.0}
        scenarios.append({"capital": float(capital), "commission": commission,
            "participation": participation, "env": env, "metrics": metrics})

    start_t = time.perf_counter()
    model.train(True)
    for t in stops:
        prices = market["prices"][t]
        future = market["prices"][t + 1]
        observed = np.asarray(loader.observed[t], bool)
        next_obs = np.asarray(loader.observed[t + 1], bool)
        spread = market["spread"][t, local_ids]
        active_np = observed[local_ids] & tradable
        if not active_np.any():
            continue
        losses, predictions = [], []
        for scenario in scenarios:
            env = scenario["env"]
            commission = scenario["commission"]
            adv = market["adv"][t, local_ids]
            impact_est = np.where(np.isfinite(adv), np.minimum(.02, .001 * np.sqrt(
                np.maximum(scenario["capital"], 0.0) / np.maximum(adv, 1.0))), 0.0)
            cost_rate = (commission + market["sell_tax"][local_ids] + spread / 2.0 +
                         SLIPPAGE_RATE + impact_est)
            mark = np.where(np.isfinite(prices[local_ids]), prices[local_ids], env.last_prices)
            current_weight = env.units * mark / max(env.equity(prices[local_ids]), 1e-12)
            args = make_context(loader, int(t), local_ids, model_ids, env, prices,
                                cost_rate, market["adv"][t], device)
            x, sid, mid, aid, mask, ctx, pstate, astate = args
            with torch.autocast("cuda", dtype=torch.float16):
                logits, values, alloc = model(x, sid, mid, aid, mask, ctx,
                    torch.tensor([3], device=device), pstate, astate, True)
            active = torch.as_tensor(active_np, device=device, dtype=torch.bool)
            current_t = torch.as_tensor(current_weight, dtype=torch.float32, device=device)
            target = mask_allocations(alloc, active, current_t)
            ret_np = np.zeros(len(local_ids), np.float32)
            good = next_obs[local_ids] & (prices[local_ids] > 0) & (future[local_ids] > 0)
            ret_np[good] = future[local_ids][good] / prices[local_ids][good] - 1.0
            ret = torch.as_tensor(ret_np, dtype=torch.float32, device=device)
            delta_weight = target[:-1] - current_t
            base_cost = (scenario["commission"] + spread / 2.0 + SLIPPAGE_RATE + impact_est)
            base_cost_t = torch.as_tensor(base_cost, dtype=torch.float32, device=device)
            tax_t = torch.as_tensor(market["sell_tax"][local_ids], dtype=torch.float32, device=device)
            costs = torch.sum(torch.relu(delta_weight) * base_cost_t +
                              torch.relu(-delta_weight) * (base_cost_t + tax_t))
            losses.append(-(torch.sum(target[:-1] * ret) - costs))
            predictions.append((scenario, target.detach().cpu().numpy(), impact_est))

        # One update per date, with all seed sizes represented in the batch.
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        for scenario, target_np, _impact_est in predictions:
            env, metrics = scenario["env"], scenario["metrics"]
            out = env.rebalance_and_mark(prices[local_ids], future[local_ids],
                observed[local_ids], next_obs[local_ids], target_np, tradable,
                market["adv"][t, local_ids], spread)
            if not math.isclose(out["equity_after"] - out["equity_before"],
                    out["reward"] * out["equity_before"], rel_tol=1e-8, abs_tol=1e-9):
                raise AssertionError("joint replay reward and account PnL disagree")
            metrics["steps"] += 1
            metrics["loss"] += float(loss.detach().cpu()) / len(scenarios)
            for source_key, dest_key in (("fee", "fees"), ("tax", "taxes"),
                    ("spread_cost", "spread_cost"), ("slippage", "slippage"),
                    ("turnover", "turnover")):
                metrics[dest_key] += out[source_key]
            metrics["buy"] += out["action_counts"]["BUY"]
            metrics["sell"] += out["action_counts"]["SELL"]
            metrics["hold"] += out["action_counts"]["HOLD"]
            metrics["position_count_sum"] += out["positions"]
            metrics["cash_min"] = min(metrics["cash_min"], out["cash"])
            metrics["pnl_equity_identity_error_max"] = max(
                metrics["pnl_equity_identity_error_max"],
                abs((out["equity_after"] - out["equity_before"]) -
                    out["reward"] * out["equity_before"]))

    elapsed = time.perf_counter() - start_t
    results = []
    for ix, scenario in enumerate(scenarios):
        env, metrics = scenario["env"], scenario["metrics"]
        start_equity = scenario["capital"]
        metrics.update({"split": "train", "episode": 0, "initial_equity": start_equity,
            "final_equity": env.equity(env.last_prices),
            "net_return_compounded": env.equity(env.last_prices) / start_equity - 1.0,
            "avg_positions": metrics["position_count_sum"] / max(metrics["steps"], 1),
            "avg_net_return_per_step": (env.equity(env.last_prices) / start_equity - 1.0) /
                                       max(metrics["steps"], 1),
            "mean_loss": metrics["loss"] / max(metrics["steps"], 1),
            "seconds": elapsed, "steps_per_second": metrics["steps"] / max(elapsed, 1e-9),
            "total_realized_net_pnl": env.realized_net,
            "cash_nonnegative": metrics["cash_min"] >= -1e-9,
            "capital_tier_index": ix, "commission_per_side": scenario["commission"],
            "participation_limit": scenario["participation"]})
        results.append(metrics)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed-sizes-usd", type=str, default="1000,10000,1000000")
    ap.add_argument("--smoke-steps", type=int, default=24)
    ap.add_argument("--learning-rate", type=float, default=0.001)
    ap.add_argument("--equity-sell-tax", type=float, default=.001)
    ap.add_argument("--validation-commission", type=float, default=.0005)
    ap.add_argument("--validation-participation", type=float, default=.01)
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()
    if args.runtime.exists(): raise FileExistsError(f"Refusing to overwrite {args.runtime}")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required for the 0.5B portfolio candidate")
    args.runtime.mkdir(parents=True)
    device = torch.device("cuda:0")
    loader = CommonMarketTrainingLoader(args.cache, [args.source], [NormalizedBarAdapter(86400)],
        chunksize=100_000, seed=args.seed, train_fraction=.70,
        validation_fraction=.15, context_stale_seconds=5 * 86400)
    manifest = loader.build(force=False)
    payload0 = torch.load(args.champion, map_location="cpu", weights_only=False)
    combined_map, local_to_model = instrument_map_for_daily(payload0["symbol_map"], loader.symbol_map)
    model_ids = np.asarray([local_to_model[i] for i in range(len(loader.symbol_names))], np.int64)
    market = load_economic_panels(args.source, loader)
    market["sell_tax"][...] = np.where(
        np.asarray([name.split("|", 2)[1] == "equity" for name in loader.symbol_names]),
        args.equity_sell_tax, 0.0)
    tradable = market["tradable"]
    if not tradable.any(): raise RuntimeError("daily panel has no tradable instruments")
    candidate_path = args.runtime / "candidate.pt"
    del payload0
    candidate_payload, _ = create_candidate(args.champion, candidate_path, loader.symbol_map)
    # Use the same tax profile in the account and in the model-visible cost state.
    market["sell_tax"] = np.where(
        np.asarray([name.split("|", 2)[1] == "equity" for name in loader.symbol_names]),
        args.equity_sell_tax, 0.0)
    cfg = TransformerConfig(**candidate_payload["config"])
    model = ContextConditionedTransformer(GlobalMarketTransformer(cfg).half())
    model.context_policy.float(); model.context_value.float()
    load_compatible_state_dict(model, candidate_payload["state_dict"], strict=True)
    # Replay training updates only the portfolio selectors. The 0.5B market
    # representation and existing action/value heads remain the exact champion.
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for module in (model.portfolio_action, model.portfolio_allocation, model.portfolio_cash):
        for parameter in module.parameters(): parameter.requires_grad_(True)
    model.freeze_backbone = True
    model.eval(); model.to(device)
    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(train_params, lr=args.learning_rate, weight_decay=1e-4)

    env_smoke = smoke_environment()
    ids = np.arange(len(loader.symbol_names))
    seed_sizes = [float(x.strip()) for x in args.seed_sizes_usd.split(",") if x.strip()]
    if not seed_sizes or any(x <= 0 for x in seed_sizes): raise ValueError("seed sizes must be positive")
    # Exercise the real multi-capital optimizer before expensive validation.
    # Restore the 3k adapter parameters afterward so this diagnostic is not
    # counted as training data or allowed to bias the actual replay run.
    trainable_snapshot = {name: parameter.detach().clone()
        for name, parameter in model.named_parameters() if parameter.requires_grad}
    joint_smoke = run_joint_training(model, loader, market, ids, model_ids, tradable,
        seed_sizes, optimizer, seed=args.seed, max_steps=2, device=device)
    if len(joint_smoke) != len(seed_sizes) or any(x["steps"] != 2 for x in joint_smoke):
        raise AssertionError("joint capital-tier update smoke test failed")
    if sum(x["buy"] for x in joint_smoke) <= 0:
        raise AssertionError("joint training smoke test failed to exercise portfolio buys")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in trainable_snapshot:
                parameter.copy_(trainable_snapshot[name])
    optimizer = torch.optim.AdamW(train_params, lr=args.learning_rate, weight_decay=1e-4)
    val_before = {}
    for capital in seed_sizes:
        val_before[str(int(capital))] = run_episode(model, loader, market, ids, model_ids,
            tradable, "validation", False, max_steps=None, seed=args.seed,
            initial_equity=capital, commission=args.validation_commission,
            participation=args.validation_participation, device=device)
    smoke_model = run_episode(model, loader, market, ids, model_ids,
        tradable, "train", False, max_steps=args.smoke_steps, seed=args.seed,
        initial_equity=seed_sizes[0], commission=args.validation_commission,
        participation=args.validation_participation, device=device)
    if smoke_model["buy"] <= 0 or smoke_model["avg_positions"] <= 0:
        raise AssertionError("model smoke replay did not select or hold any instruments")
    if not smoke_model["cash_nonnegative"] or smoke_model["pnl_equity_identity_error_max"] > 1e-9:
        raise AssertionError("smoke replay account checks failed")
    report = {"event": "portfolio_smoke_test_passed", "source": str(args.source),
        "source_manifest": manifest, "champion_sha256": candidate_payload["source_champion_sha256"],
        "candidate": str(candidate_path), "instrument_count": len(loader.symbol_names),
        "tradable_instrument_count": int(tradable.sum()),
        "nontradable_context_count": int((~tradable).sum()),
        "trading_rule": "cash-backed long positions; per-instrument lots; no leverage/shorting",
        "seed_sizes_usd": seed_sizes,
        "commission_profile": {"training_range_per_side": [0.0002, 0.001],
                                "validation_per_side": args.validation_commission},
        "equity_sell_tax_assumption": args.equity_sell_tax,
        "slippage_model": "half-spread proxy from daily high-low + base slippage + square-root order/ADV impact",
        "volume_limit": "per-order participation cap against observed dollar ADV; FX volume absent uses no ADV cap",
        "fx_conversion": "KRW, JPY, EUR, GBP converted using source FX series; FX-pair instruments use normalized price path",
        "futures_model_limit": "unlevered notional proxy; contract multipliers/margins unavailable in source",
        "environment_checks": env_smoke, "model_smoke": smoke_model,
        "joint_capital_training_smoke": joint_smoke,
        "validation_before": val_before, "student_parameters": parameter_count(model.backbone),
        "trainable_portfolio_parameters": sum(p.numel() for p in train_params),
        "test_accessed": False, "promotion": "not performed"}
    (args.runtime / "smoke-result.json").write_text(json.dumps(report, indent=2, allow_nan=False, cls=NumpyJSONEncoder), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False, cls=NumpyJSONEncoder), flush=True)

    # Train chronological historical self-play. Each date feeds actual cash,
    # positions, average cost, open PnL, realized PnL and costs back as state.
    epoch_results = []
    for epoch in range(args.epochs):
        tier_results = run_joint_training(model, loader, market, ids, model_ids,
            tradable, seed_sizes, optimizer, seed=args.seed + epoch * 101,
            max_steps=None, device=device)
        for result in tier_results:
            result["epoch"] = epoch + 1
            epoch_results.append(result)
            with (args.runtime / "training.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "portfolio_replay_epoch", **result}, allow_nan=False, cls=NumpyJSONEncoder) + "\n")
            print(json.dumps({"event": "portfolio_replay_epoch", **result}, cls=NumpyJSONEncoder), flush=True)
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        payload.update({"state_dict": state, "portfolio_pretrain_epoch": epoch + 1,
                       "portfolio_training_log": str(args.runtime / "training.jsonl"),
                       "portfolio_update_mode": "joint_capital_tiers_one_optimizer_update_per_date",
                       "candidate_decision": "awaiting_validation"})
        torch.save(payload, candidate_path)
        del payload

    val_after = {}
    for capital in seed_sizes:
        val_after[str(int(capital))] = run_episode(model, loader, market, ids, model_ids,
            tradable, "validation", False, max_steps=None, seed=args.seed,
            initial_equity=capital, commission=args.validation_commission,
            participation=args.validation_participation, device=device)
    # Reload from disk and run a CUDA inference check with explicit account state.
    reloaded = ContextConditionedTransformer(GlobalMarketTransformer(cfg).half())
    reloaded.context_policy.float(); reloaded.context_value.float()
    ck = torch.load(candidate_path, map_location="cpu", weights_only=False)
    load_compatible_state_dict(reloaded, ck["state_dict"], strict=True)
    reloaded.freeze_backbone = True; reloaded.eval().to(device)
    t = int(loader.splits["validation"][0]) + SEQ - 1
    sample_env = Portfolio(len(ids), seed_sizes[0], lots=market["lots"],
        sell_tax=market["sell_tax"], commission=args.validation_commission,
        participation=args.validation_participation)
    state_prices = market["prices"][t]
    sample_cost = args.validation_commission + market["sell_tax"] + market["spread"][t] / 2 + SLIPPAGE_RATE
    sample = make_context(loader, t, ids, model_ids, sample_env, state_prices,
                          sample_cost, market["adv"][t], device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        logits, values, allocation = reloaded(*sample[:6], torch.tensor([3], device=device),
                                               sample[6], sample[7], True)
    if not torch.isfinite(logits).all() or not torch.isfinite(values).all() or not torch.isfinite(allocation).all():
        raise AssertionError("reloaded CUDA portfolio inference returned nonfinite outputs")
    allocation_sum = float(allocation.sum().cpu())
    if abs(allocation_sum - 1.0) > 1e-5: raise AssertionError("cash-inclusive allocation does not sum to one")
    final = {"event": "portfolio_pretraining_complete", "candidate": str(candidate_path),
        "champion_sha256": candidate_payload["source_champion_sha256"],
        "epochs": args.epochs, "train_epochs": epoch_results,
        "validation_before": val_before, "validation_after": val_after,
        "validation_net_return_delta_by_seed": {k: val_after[k]["net_return_compounded"] - val_before[k]["net_return_compounded"] for k in val_after},
        "validation_improved_by_seed": {k: val_after[k]["net_return_compounded"] > val_before[k]["net_return_compounded"] for k in val_after},
        "candidate_cuda_reload": True, "inference_logits_shape": list(logits.shape),
        "allocation_shape_including_cash": list(allocation.shape),
        "allocation_sum": allocation_sum,
        "trained_modules": ["portfolio_action", "portfolio_allocation", "portfolio_cash"],
        "market_backbone_changed": False, "test_accessed": False,
        "candidate_decision": ("eligible_for_manual_review" if all(
            val_after[k]["net_return_compounded"] > val_before[k]["net_return_compounded"]
            for k in val_after) else "rejected_validation"),
        "tool_authority_design": "autonomy execution gate is external to BUY/HOLD/SELL policy output",
        "promotion": "not performed; existing champion unchanged"}
    (args.runtime / "result.json").write_text(json.dumps(final, indent=2, allow_nan=False, cls=NumpyJSONEncoder), encoding="utf-8")
    with (args.runtime / "training.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(final, allow_nan=False, cls=NumpyJSONEncoder) + "\n")
    print(json.dumps(final, indent=2, allow_nan=False, cls=NumpyJSONEncoder), flush=True)
    loader.close()


if __name__ == "__main__":
    main()
