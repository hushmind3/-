"""Shared chronological market panels and feature/context contracts."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
if TYPE_CHECKING:
    import torch

GLOBAL_FEATURES = (
    "ret1", "ret5", "ret20", "range", "volume_z", "rsi", "volatility",
    "spread_bps", "book_imbalance", "trade_imbalance", "trade_intensity",
    "implied_volatility", "open_interest_change", "yield_change",
    "days_to_expiry", "video_chart_signal", "video_volume_signal",
)

ACTION_NAMES = ("SELL", "HOLD", "BUY")

TIME_SCALE_NAMES = ("seconds", "minutes", "hours", "days")

MARKET_CONTEXT_FEATURES = (
    "universe_coverage", "recently_active_share", "advance_share", "decline_share",
    "current_mean_return_pct", "current_return_vol_pct", "recent_mean_return_pct",
    "recent_median_return_pct", "recent_return_vol_pct", "current_log_volume",
    "current_trade_imbalance", "recent_median_spread_pct", "recent_p90_spread_pct",
    "recent_mean_book_imbalance", "top10_volume_share", "recent_median_volatility_pct",
)

def stable_id(value: object, modulo: int) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo

def _feature_panel(frame: pd.DataFrame, training_compatible: bool = False) -> pd.DataFrame:
    required = {"date", "symbol", "market", "asset_class", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing: raise ValueError(f"global CSV missing columns: {sorted(missing)}")
    df = frame.copy()
    # Intraday feeds may use ISO timestamps. Convert to UTC-naive timestamps
    # without truncating to a day; daily historical inputs remain at midnight.
    df["date"] = pd.to_datetime(df.date, errors="raise", utc=True).dt.tz_convert(None)
    df = df.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.close.isna().any() or (df.close <= 0).any(): raise ValueError("close must be positive numeric data")
    built = []
    for symbol, g in df.groupby("symbol", sort=False):
        g = g.copy(); close = g.close.astype(float); ret = close.pct_change()
        logvol = np.log1p(g.volume.clip(lower=0))
        vmean = logvol.rolling(20, min_periods=2).mean()
        vstd = logvol.rolling(20, min_periods=2).std()
        delta = close.diff(); gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = -delta.clip(upper=0).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        g["ret1"] = ret; g["ret5"] = close.pct_change(5); g["ret20"] = close.pct_change(20)
        g["range"] = (g.high - g.low) / close
        g["volume_z"] = (logvol-vmean)/vstd.replace(0, np.nan)
        g["rsi"] = (100-100/(1+rs)).fillna(50)/100
        g["volatility"] = ret.rolling(20, min_periods=2).std()
        opt = pd.to_numeric(g.get("open_interest", pd.Series(np.nan,index=g.index)), errors="coerce")
        g["open_interest_change"] = opt.pct_change(fill_method=None)
        bid = pd.to_numeric(g.get("bid", pd.Series(np.nan,index=g.index)), errors="coerce")
        ask = pd.to_numeric(g.get("ask", pd.Series(np.nan,index=g.index)), errors="coerce")
        g["spread_bps"] = (ask-bid)/((ask+bid)/2)*10000
        bs = pd.to_numeric(g.get("bid_size", pd.Series(np.nan,index=g.index)), errors="coerce")
        ass = pd.to_numeric(g.get("ask_size", pd.Series(np.nan,index=g.index)), errors="coerce")
        g["book_imbalance"] = (bs-ass)/(bs+ass)
        bv = pd.to_numeric(g.get("buy_volume", pd.Series(np.nan,index=g.index)), errors="coerce")
        sv = pd.to_numeric(g.get("sell_volume", pd.Series(np.nan,index=g.index)), errors="coerce")
        g["trade_imbalance"] = (bv-sv)/(bv+sv)
        tc = pd.to_numeric(g.get("trade_count", pd.Series(np.nan,index=g.index)), errors="coerce")
        g["trade_intensity"] = tc/(tc.rolling(20,min_periods=2).mean()+1e-8)-1
        if training_compatible:
            # Mirror CommonMarketTrainingLoader._fill_arrays: rolling returns,
            # volume z-score against prior bars, bounded RSI/volatility, and
            # trade intensity against prior observations. The same ordered
            # 17-vector is therefore produced for training and live bars.
            prior_logvol = logvol.shift(1)
            prior_mean = prior_logvol.rolling(20, min_periods=2).mean()
            prior_std = prior_logvol.rolling(20, min_periods=2).std(ddof=1)
            delta = close.diff()
            gains = delta.clip(lower=0).rolling(14, min_periods=1).mean()
            losses = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
            rsi = gains / (gains + losses).replace(0, np.nan)
            g["ret1"] = ret
            g["ret5"] = close.pct_change(5)
            g["ret20"] = close.pct_change(20)
            g["range"] = (g.high - g.low) / close
            g["volume_z"] = (logvol - prior_mean) / prior_std.replace(0, np.nan)
            g["rsi"] = rsi.fillna(.5)
            g["volatility"] = ret.rolling(20, min_periods=2).std(ddof=1)
            g["trade_intensity"] = tc / (tc.shift(1).rolling(20, min_periods=1).mean() + 1e-8) - 1
        for name in ("implied_volatility", "yield_change", "days_to_expiry",
                     "video_chart_signal", "video_volume_signal"):
            g[name] = pd.to_numeric(g.get(name, pd.Series(0.0,index=g.index)), errors="coerce")
        # Standardize selected long-scale fields using past-only rolling stats.
        # Unprovided fields remain zero and are tracked via availability mask.
        for name in GLOBAL_FEATURES:
            if name not in g: g[name] = 0.0
        g[list(GLOBAL_FEATURES)] = g[list(GLOBAL_FEATURES)].replace([np.inf,-np.inf],np.nan).fillna(0).clip(-10,10)
        g["observed"] = True
        built.append(g)
    return pd.concat(built, ignore_index=True)

class GlobalMarketPanel:
    """Chronologically aligned panel; absent bars are forward-filled only."""
    def __init__(self, path: str | Path, max_symbols: int = 8192,
                 symbol_map: dict[str, int] | None = None,
                 context_stale_seconds: int = 60,
                 recent_timestamps: int | None = None,
                 active_stale_seconds: int | None = None,
                 raw_frame=None):
        self.source_path = Path(path)
        raw = pd.read_csv(path) if raw_frame is None else raw_frame.copy()
        self.recent_cutoff = None
        if recent_timestamps is not None and recent_timestamps > 0 and len(raw):
            stamps = pd.to_datetime(raw["date"], errors="raise", utc=True)
            unique_stamps = stamps.dropna().drop_duplicates().sort_values()
            if len(unique_stamps) > recent_timestamps:
                cutoff = unique_stamps.iloc[-recent_timestamps]
                self.recent_cutoff = cutoff.tz_localize(None).to_datetime64()
                # Retain the active window for every symbol. Keep only enough
                # older reference bars to calculate rolling features and carry
                # a last quote across closed sessions.
                reference = ~raw["asset_class"].astype(str).str.casefold().eq("equity")
                recent = raw.loc[stamps >= cutoff]
                carry = raw.loc[(stamps < cutoff) & reference].groupby("symbol",sort=False).tail(20)
                raw = pd.concat((recent,carry),ignore_index=True).sort_values(
                    ["date","symbol"],kind="stable").reset_index(drop=True)
        df = _feature_panel(raw, training_compatible=symbol_map is not None)
        context_frame = df
        if active_stale_seconds is not None and len(df):
            cutoff = df.date.max() - pd.Timedelta(seconds=active_stale_seconds)
            active = set(df.groupby("symbol").date.max().loc[lambda x: x >= cutoff].index)
            df = df[df.symbol.isin(active)].copy()
        self.frame = df
        self._multiscale_builder = None
        self.symbol_map = dict(symbol_map or {})
        self.uses_market_context = bool(symbol_map)
        self.market_context = None
        self._map_symbol_ids: dict[str, int] = {}
        self._map_market_ids: dict[str, int] = {}
        self._map_asset_ids: dict[str, int] = {}
        self.unmatched_symbols: list[str] = []
        if symbol_map:
            canonical_by_ticker: dict[str, list[tuple[str, int]]] = {}
            canonical_by_identity: dict[str, tuple[str, int]] = {}
            for qualified, sid in symbol_map.items():
                parts = qualified.split("|", 2)
                if len(parts) != 3:
                    continue
                identity = f"{parts[0].upper()}|{parts[1].casefold()}|{parts[2].upper()}"
                canonical_by_identity[identity] = (qualified, int(sid))
                canonical_by_ticker.setdefault(parts[2].upper(), []).append((qualified, int(sid)))
            allowed_us = {"US", "NASDAQ", "NYSE", "NYSEARCA", "AMEX", "NASDAQGS", "NASDAQGM", "NASDAQCM"}
            market_aliases = {"KRX": "KR", "KOSDAQ": "KR", "JAPAN": "JP", "HONGKONG": "HK",
                              "GERMANY": "DE", "CME": "US", "COMEX": "US", "NYMEX": "US",
                              "US_TREASURY": "US"}
            asset_aliases = {"yield": "treasury_yield", "indexfuture": "index_future",
                             "commodityfuture": "commodity_future"}
            symbol_rows = []
            unmatched_symbols = []
            for symbol, group in df.groupby("symbol", sort=True):
                last = group.iloc[-1]
                market = str(last.market).upper()
                market = market_aliases.get(market, market)
                asset = str(last.asset_class).casefold().replace("_", "")
                asset = asset_aliases.get(asset, asset)
                identity = f"{market}|{asset}|{str(symbol).upper()}"
                direct_pair = canonical_by_identity.get(identity)
                qualified, direct = direct_pair if direct_pair else ("", None)
                if direct is None and market in allowed_us:
                    candidates = canonical_by_ticker.get(str(symbol).upper(), [])
                    if len(candidates) == 1:
                        qualified, direct = candidates[0]
                if direct is None:
                    unmatched_symbols.append(str(symbol))
                    continue
                parts = qualified.split("|", 2)
                if len(parts) != 3:
                    continue
                symbol_rows.append((str(symbol), int(direct), stable_id(parts[0], 64), stable_id(parts[1], 32)))
            self.symbols = [row[0] for row in symbol_rows]
            self._map_symbol_ids = {row[0]: row[1] for row in symbol_rows}
            self._map_market_ids = {row[0]: row[2] for row in symbol_rows}
            self._map_asset_ids = {row[0]: row[3] for row in symbol_rows}
            self.unmatched_symbols = sorted(unmatched_symbols)
            if not self.symbols:
                raise ValueError("no live instruments match the candidate's trained symbol map")
            # Keep the training-era minimum width for small live universes,
            # while allowing every matched trained instrument when the live
            # market has grown beyond 128 symbols. The model's symbol axis is
            # dynamic; its learned symbol IDs remain bounded by max_symbols.
            inference_width = max(128, len(symbol_rows))
            ids_in_use = {row[1] for row in symbol_rows}
            by_id = sorted(((int(sid), name) for name, sid in symbol_map.items()))
            for sid, qualified in by_id:
                if len(symbol_rows) >= inference_width:
                    break
                if sid in ids_in_use:
                    continue
                parts = qualified.split("|", 2)
                pad_name = f"__PAD__{sid}__{qualified}"
                symbol_rows.append((pad_name, sid, stable_id(parts[0], 64), stable_id(parts[1], 32)))
                ids_in_use.add(sid)
            self.symbols = [row[0] for row in symbol_rows]
            self._map_symbol_ids = {row[0]: row[1] for row in symbol_rows}
            self._map_market_ids = {row[0]: row[2] for row in symbol_rows}
            self._map_asset_ids = {row[0]: row[3] for row in symbol_rows}
            # Inference/backtest tensors include only instruments with trained
            # symbol IDs. Other feed symbols remain out of the action set.
            active_symbols = {row[0] for row in symbol_rows if not row[0].startswith("__PAD__")}
            df = df[df.symbol.astype(str).isin(active_symbols)].copy()
            context_dates = np.sort(context_frame.date.unique())
            context = self._build_market_context(context_frame, symbol_map, context_stale_seconds)
            live_dates = np.sort(df.date.unique())
            self.market_context = context[np.searchsorted(context_dates, live_dates)]
        else:
            self.symbols = sorted(df.symbol.astype(str).unique())
        self.dates = np.array(sorted(df.date.unique()), dtype="datetime64[ns]")
        self.frame = df
        t, n, f = len(self.dates), len(self.symbols), len(GLOBAL_FEATURES)
        self.features = np.zeros((t,n,f),np.float32)
        self.closes = np.full((t,n),np.nan,np.float64)
        self.observed = np.zeros((t,n),bool)
        self.symbol_ids = np.array([self._map_symbol_ids[s] if self.uses_market_context else stable_id(s,max_symbols)
                                    for s in self.symbols],np.int64)
        self.market_ids = np.zeros(n,np.int64); self.asset_ids = np.zeros(n,np.int64)
        self.groups = {}
        date_ix = {d:i for i,d in enumerate(self.dates)}; sym_ix = {s:i for i,s in enumerate(self.symbols)}
        for _, row in df.iterrows():
            i,j=date_ix[row.date.to_datetime64()],sym_ix[str(row.symbol)]
            self.features[i,j]=row[list(GLOBAL_FEATURES)].to_numpy(np.float32)
            self.closes[i,j]=float(row.close); self.observed[i,j]=True
            if self.uses_market_context:
                self.market_ids[j]=self._map_market_ids[str(row.symbol)]
                self.asset_ids[j]=self._map_asset_ids[str(row.symbol)]
            else:
                self.market_ids[j]=stable_id(row.market,64); self.asset_ids[j]=stable_id(row.asset_class,32)
            self.groups[str(row.symbol)]=(str(row.market),str(row.asset_class))
        # Point-in-time forward fill closes/features per instrument; mask still
        # marks actual exchange observations so stale prices never earn reward.
        for j in range(n):
            for i in range(1,t):
                if not self.observed[i,j]:
                    self.features[i,j]=self.features[i-1,j]
                    self.closes[i,j]=self.closes[i-1,j]

    def multiscale_at(self, index: int) -> np.ndarray:
        """Completed 1/3/5/15/60-minute and day/week/month context as of a bar."""
        from .multiscale import MultiscaleFeatures
        if self._multiscale_builder is None:
            self._multiscale_builder = MultiscaleFeatures(
                self.frame, self.symbols,
                self.source_path.with_name("timeframes.sqlite3"), self.dates[-1])
        return self._multiscale_builder.at(self.dates[index])

    def _build_market_context(self, df: pd.DataFrame, symbol_map: dict[str, int],
                              stale_seconds: int) -> np.ndarray:
        n_universe = len(symbol_map)
        t = len(np.sort(df.date.unique()))
        out = np.zeros((t, len(MARKET_CONTEXT_FEATURES)), dtype=np.float32)
        ticker_to_id: dict[str, int] = {}
        for qualified, sid in symbol_map.items():
            ticker_to_id[qualified.rsplit("|", 1)[-1].upper()] = int(sid)
        seen = np.zeros(n_universe, dtype=bool)
        last_seen = np.zeros(n_universe, dtype=np.int64)
        state_return = np.zeros(n_universe, dtype=np.float32)
        state_spread = np.zeros(n_universe, dtype=np.float32)
        state_book = np.zeros(n_universe, dtype=np.float32)
        state_vol = np.zeros(n_universe, dtype=np.float32)
        dates = np.sort(df.date.unique())
        # Group once instead of rescanning the complete source frame for every
        # timestamp. This keeps live panel refresh cost close to O(rows).
        for ti, (date, group) in enumerate(df.groupby("date", sort=True)):
            ids, returns, volumes, net_flows = [], [], [], []
            ts = int(pd.Timestamp(date).timestamp())
            for row in group.itertuples(index=False):
                sid = ticker_to_id.get(str(row.symbol).upper())
                if sid is None:
                    continue
                ids.append(sid)
                ret = float(np.nan_to_num(row.ret1, nan=0.0))
                volume = max(float(np.nan_to_num(row.volume, nan=0.0)), 0.0)
                buy = max(float(np.nan_to_num(getattr(row, "buy_volume", 0.0), nan=0.0)), 0.0)
                sell = max(float(np.nan_to_num(getattr(row, "sell_volume", 0.0), nan=0.0)), 0.0)
                returns.append(ret); volumes.append(volume); net_flows.append(buy-sell)
                seen[sid] = True; last_seen[sid] = ts; state_return[sid] = ret
                state_spread[sid] = float(np.nan_to_num(row.spread_bps, nan=0.0))
                state_book[sid] = float(np.nan_to_num(row.book_imbalance, nan=0.0))
                state_vol[sid] = float(np.nan_to_num(row.volatility, nan=0.0))
            if not ids:
                continue
            current_returns = np.asarray(returns, dtype=np.float64)
            current_volumes = np.asarray(volumes, dtype=np.float64)
            current_flows = np.asarray(net_flows, dtype=np.float64)
            fresh = seen & ((ts-last_seen) <= int(stale_seconds))
            fresh_r = state_return[fresh].astype(np.float64)
            fresh_spread = state_spread[fresh].astype(np.float64)
            fresh_book = state_book[fresh].astype(np.float64)
            total_volume = float(current_volumes.sum())
            top10 = float(np.sort(current_volumes)[-10:].sum()/max(total_volume, 1.0))
            out[ti] = np.asarray((
                seen.mean(), fresh.mean(), float((current_returns>0).mean()), float((current_returns<0).mean()),
                float(current_returns.mean()*100), float(current_returns.std()*100),
                float(fresh_r.mean()*100) if len(fresh_r) else 0.0,
                float(np.median(fresh_r)*100) if len(fresh_r) else 0.0,
                float(fresh_r.std()*100) if len(fresh_r) else 0.0,
                np.log1p(total_volume)/20.0, float(current_flows.sum())/max(total_volume, 1.0),
                float(np.median(fresh_spread))/100.0 if len(fresh_spread) else 0.0,
                float(np.quantile(fresh_spread, .9))/100.0 if len(fresh_spread) else 0.0,
                float(fresh_book.mean()) if len(fresh_book) else 0.0, top10,
                float(np.median(state_vol[fresh]))*100 if fresh.any() else 0.0,
            ), dtype=np.float32)
        np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -10, 10, out=out)
        return out

    def window(self, end: int, length: int, include_context: bool = False) -> tuple[torch.Tensor,...]:
        import torch
        start=max(0,end-length+1); sl=slice(start,end+1)
        x=self.features[sl]; mask=self.observed[sl]
        if len(x)<length:
            pad=length-len(x); x=np.pad(x,((pad,0),(0,0),(0,0))); mask=np.pad(mask,((pad,0),(0,0)))
        result = (torch.from_numpy(x[None].copy()), torch.from_numpy(self.symbol_ids[None].copy()),
                torch.from_numpy(self.market_ids[None].copy()), torch.from_numpy(self.asset_ids[None].copy()),
                torch.from_numpy(mask[None].copy()))
        if include_context:
            if self.market_context is None:
                raise ValueError("market_context is required by this checkpoint; load panel with its symbol_map")
            context = self.market_context[sl]
            if len(context) < length:
                context = np.pad(context, ((length-len(context),0),(0,0)))
            result = result + (torch.from_numpy(context[None].copy()),)
        return result

    def return_to(self, start: int, end: int, symbol: int) -> float:
        a,b=self.closes[start,symbol],self.closes[end,symbol]
        if not self.observed[start,symbol] or not self.observed[end,symbol] or not np.isfinite(a*b) or a<=0:
            return 0.0
        return float(b/a-1)

    def daily_history_at(self,index):
        self.multiscale_at(index)
        return self._multiscale_builder.daily_at(self.dates[index])

    def daily_history_status_at(self,index):
        self.multiscale_at(index)
        return self._multiscale_builder.daily_status_at(self.dates[index])
