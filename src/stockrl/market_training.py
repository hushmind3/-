"""Shared, bounded-memory sequence loader for multi-market training.

Source adapters normalize provider/exchange-specific CSVs into one event schema.
The loader builds a disk-backed chronological panel, full-universe context, and
coverage-balanced symbol groups without materializing all symbols on the GPU.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .global_transformer import (GLOBAL_FEATURES, GlobalMarketTransformer,
                                 TransformerConfig, stable_id, load_compatible_state_dict)


CONTEXT_FEATURES = (
    "universe_coverage", "recently_active_share", "advance_share", "decline_share",
    "current_mean_return_pct", "current_return_vol_pct", "recent_mean_return_pct",
    "recent_median_return_pct", "recent_return_vol_pct", "current_log_volume",
    "current_trade_imbalance", "recent_median_spread_pct", "recent_p90_spread_pct",
    "recent_mean_book_imbalance", "top10_volume_share", "recent_median_volatility_pct",
)
CANONICAL_COLUMNS = (
    "time_key", "event_time_ns", "instrument", "symbol", "market", "asset_class",
    "close", "volume", "bid", "ask", "bid_size", "ask_size", "buy_volume",
    "sell_volume", "trade_count", "implied_volatility", "open_interest",
    "open_interest_change", "yield_change", "days_to_expiry", "video_chart_signal",
    "video_volume_signal", "currency",
)


class MarketSourceAdapter:
    """Adapter contract: turn one source chunk into canonical market events."""
    name = "base"

    def __init__(self, timeframe_seconds: int = 1):
        if timeframe_seconds < 1:
            raise ValueError("timeframe_seconds must be >= 1")
        self.timeframe_seconds = int(timeframe_seconds)

    def iter_chunks(self, path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
        raise NotImplementedError

    def normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def _finish(frame: pd.DataFrame, source: str) -> pd.DataFrame:
        out = frame.copy()
        for name in ("volume", "bid", "ask", "bid_size", "ask_size", "buy_volume", "sell_volume", "trade_count"):
            if name not in out:
                out[name] = 0.0
            out[name] = pd.to_numeric(out[name], errors="coerce").fillna(0.0)
        for name in ("implied_volatility", "open_interest", "open_interest_change", "yield_change",
                     "days_to_expiry", "video_chart_signal", "video_volume_signal"):
            if name not in out:
                out[name] = 0.0
            out[name] = pd.to_numeric(out[name], errors="coerce").fillna(0.0)
        if "currency" not in out:
            out["currency"] = "UNKNOWN"
        out["currency"] = out["currency"].fillna("UNKNOWN").astype(str)
        for name in ("open", "high", "low", "close"):
            if name not in out:
                out[name] = out["close"]
            out[name] = pd.to_numeric(out[name], errors="coerce")
        if out["close"].isna().any() or (out["close"] <= 0).any():
            raise ValueError(f"{source}: close must be positive numeric data")
        if "market" not in out: out["market"] = "UNKNOWN"
        if "asset_class" not in out: out["asset_class"] = "UNKNOWN"
        out["market"] = out["market"].fillna("UNKNOWN").astype(str)
        out["asset_class"] = out["asset_class"].fillna("UNKNOWN").astype(str)
        out["symbol"] = out["symbol"].astype(str)
        out["instrument"] = out["market"] + "|" + out["asset_class"] + "|" + out["symbol"]
        return out[list(CANONICAL_COLUMNS) + ["open", "high", "low"]]


class NormalizedBarAdapter(MarketSourceAdapter):
    """Adapter for UTC OHLCV CSVs with date/timestamp, symbol and market fields."""
    name = "normalized_bars"

    def iter_chunks(self, path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
        yield from pd.read_csv(path, chunksize=chunksize, dtype={"symbol": "string"})

    def normalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        date_col = "timestamp_utc" if "timestamp_utc" in frame else "date"
        if date_col not in frame or "symbol" not in frame:
            raise ValueError("bar source needs timestamp_utc/date and symbol columns")
        # Exchange event timestamps may mix whole seconds and fractional
        # seconds while remaining valid ISO-8601 UTC values.
        dt = pd.to_datetime(frame[date_col], utc=True, errors="raise", format="mixed")
        ns = dt.astype("int64").to_numpy()
        sec = ns // 1_000_000_000
        out = frame.copy()
        out["event_time_ns"] = ns
        out["time_key"] = (sec // self.timeframe_seconds) * self.timeframe_seconds
        out["market"] = out.get("market", pd.Series("UNKNOWN", index=out.index)).fillna("UNKNOWN")
        out["asset_class"] = out.get("asset_class", pd.Series("UNKNOWN", index=out.index)).fillna("UNKNOWN")
        return self._finish(out, self.name)


class ITCHSnapshotAdapter(NormalizedBarAdapter):
    """Adapter for the corrected Nasdaq ITCH snapshot CSV, not raw ITCH bytes."""
    name = "nasdaq_itch_snapshot"


@dataclass
class MarketBatch:
    features: torch.Tensor          # [B, sequence, sampled symbols, 17]
    symbol_ids: torch.Tensor        # [B, sampled symbols], unique direct IDs
    market_ids: torch.Tensor       # [B, sampled symbols]
    asset_ids: torch.Tensor        # [B, sampled symbols]
    valid_mask: torch.Tensor       # [B, sequence, sampled symbols]
    market_context: torch.Tensor  # [B, sequence, context_features]
    target_return: torch.Tensor   # [B, sampled symbols]
    target_valid: torch.Tensor   # [B, sampled symbols]
    end_indices: np.ndarray
    target_indices: np.ndarray
    sampled_ids: list[np.ndarray]


class ContextConditionedTransformer(nn.Module):
    """Global Transformer plus a tiny trainable context residual.

    The base 0.5B architecture remains unchanged. A small adapter lets
    whole-universe context affect policy/value outputs; the backbone can be
    trained with checkpointed activations or frozen for constrained devices.
    """
    def __init__(self, backbone: nn.Module, context_size: int = len(CONTEXT_FEATURES)):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = False
        self.activation_checkpointing = False
        self.context_policy = nn.Linear(context_size, 3, bias=False)
        self.context_value = nn.Linear(context_size, 1, bias=False)
        self.portfolio_action = nn.Sequential(
            nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 3))
        self.portfolio_allocation = nn.Sequential(
            nn.Linear(20, 64), nn.GELU(), nn.Linear(64, 1))
        self.portfolio_cash = nn.Sequential(
            nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.context_policy.weight)
        nn.init.zeros_(self.context_value.weight)
        nn.init.zeros_(self.portfolio_action[-1].weight)
        nn.init.zeros_(self.portfolio_action[-1].bias)
        nn.init.zeros_(self.portfolio_allocation[-1].weight)
        nn.init.zeros_(self.portfolio_allocation[-1].bias)
        nn.init.zeros_(self.portfolio_cash[-1].weight)
        nn.init.zeros_(self.portfolio_cash[-1].bias)

    def forward(self, features: torch.Tensor, symbol_ids: torch.Tensor,
                market_ids: torch.Tensor, asset_ids: torch.Tensor,
                valid_mask: torch.Tensor, market_context: torch.Tensor,
                time_scale_ids: torch.Tensor | None = None,
                portfolio_state: torch.Tensor | None = None,
                account_state: torch.Tensor | None = None,
                return_allocation: bool = False):
        if self.activation_checkpointing and self.training:
            logits, values = self._checkpointed_backbone(
                features, symbol_ids, market_ids, asset_ids, valid_mask, time_scale_ids)
        elif self.freeze_backbone:
            with torch.no_grad():
                logits, values = self.backbone(features, symbol_ids, market_ids, asset_ids, valid_mask, time_scale_ids)
        else:
            logits, values = self.backbone(features, symbol_ids, market_ids, asset_ids, valid_mask, time_scale_ids)
        context = market_context.float()
        # Retain both the recent global state and its sequence-level average;
        # the full-universe context is available at every timestamp in window.
        policy_context = self.context_policy(context)
        value_context = self.context_value(context)
        policy_delta = (policy_context[:, -1] + policy_context.mean(dim=1)).to(dtype=logits.dtype)
        value_delta = (value_context[:, -1] + value_context.mean(dim=1)).to(dtype=values.dtype)
        logits = logits + policy_delta[:, None, :]
        values = values + value_delta[:, None, 0]
        if portfolio_state is None and account_state is None:
            if return_allocation:
                raise ValueError("portfolio_state and account_state are required for allocation output")
            return logits, values
        if portfolio_state is None or account_state is None:
            raise ValueError("portfolio_state and account_state must be supplied together")
        if portfolio_state.shape != (*logits.shape[:2], 8) or account_state.shape != (logits.shape[0], 8):
            raise ValueError("portfolio inputs must have shapes [batch,symbols,8] and [batch,8]")
        per_symbol_state = torch.cat((portfolio_state.float(),
                                      account_state.float()[:, None, :].expand(-1, logits.shape[1], -1)), dim=-1)
        logits = logits + self.portfolio_action(per_symbol_state).to(logits.dtype)
        alloc_input = torch.cat((logits.float(), values.float()[..., None], per_symbol_state), dim=-1)
        asset_scores = self.portfolio_allocation(alloc_input).squeeze(-1)
        cash_score = self.portfolio_cash(account_state.float()).squeeze(-1)
        allocation = torch.softmax(torch.cat((asset_scores, cash_score[:, None]), dim=-1), dim=-1)
        if return_allocation:
            return logits, values, allocation
        return logits, values

    def _checkpointed_backbone(self, features, symbol_ids, market_ids, asset_ids, valid_mask,
                               time_scale_ids=None):
        """Same blocks/layout as the base forward, recomputed for bounded VRAM."""
        base = self.backbone
        b, t, n, _ = features.shape
        if t > base.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len={base.cfg.max_seq_len}")
        x = base.input_proj(features)
        x = x + base.symbol_embedding(symbol_ids)[:, None] + base.market_embedding(market_ids)[:, None]
        x = x + base.asset_embedding(asset_ids)[:, None]
        x = x + base.time_embedding(torch.arange(t, device=x.device))[None, :, None]
        if time_scale_ids is not None:
            scale_ids = torch.as_tensor(time_scale_ids, device=x.device, dtype=torch.long)
            if scale_ids.ndim == 0: scale_ids = scale_ids.expand(b)
            if scale_ids.shape != (b,): raise ValueError(f"time_scale_ids must be scalar or shape [{b}]")
            x = x + base.time_scale_embedding(scale_ids)[:, None, None, :]
        valid = valid_mask.bool()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        for i, block in enumerate(base.blocks):
            if i % 2 == 0:
                z = x.permute(0, 2, 1, 3).reshape(b*n, t, -1)
                mask = (~valid.permute(0, 2, 1)).reshape(b*n, t)
                all_pad = mask.all(-1)
                if all_pad.any(): mask[all_pad, 0] = False
                z = checkpoint(block, z, mask, use_reentrant=False)
                x = z.reshape(b, n, t, -1).permute(0, 2, 1, 3)
            else:
                z = x.reshape(b*t, n, -1)
                mask = (~valid).reshape(b*t, n)
                all_pad = mask.all(-1)
                if all_pad.any(): mask[all_pad, 0] = False
                z = checkpoint(block, z, mask, use_reentrant=False)
                x = z.reshape(b, t, n, -1)
            x = x * valid[..., None]
        x = base.final_norm(x[:, -1])
        return base.policy_head(x), base.value_head(x).squeeze(-1)


def load_market_candidate_checkpoint(path: str | Path,
                                     device: str | torch.device = "cpu"):
    """Reload an isolated common-loader candidate without touching champion."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = TransformerConfig(**payload["config"])
    backbone = GlobalMarketTransformer(config).half()
    model = ContextConditionedTransformer(backbone)
    model.context_policy.float(); model.context_value.float()
    load_compatible_state_dict(model, payload["state_dict"], strict=True)
    model.to(torch.device(device)).eval()
    return model, payload


class CoverageSampler:
    """Mostly least-seen sampling, with a small volume-weighted component."""
    def __init__(self, n_symbols: int, volume_weights: np.ndarray, seed: int = 7,
                 coverage_fraction: float = 0.8):
        self.n_symbols = int(n_symbols)
        self.rng = np.random.default_rng(seed)
        self.exposure = np.zeros(n_symbols, dtype=np.uint64)
        self.volume_weights = np.asarray(volume_weights, dtype=np.float64)
        if len(self.volume_weights) != n_symbols:
            raise ValueError("volume weight count does not match symbol map")
        self.coverage_fraction = float(coverage_fraction)

    def choose(self, count: int) -> np.ndarray:
        if not 0 < count <= self.n_symbols:
            raise ValueError(f"symbols_per_sample must be in [1,{self.n_symbols}]")
        weighted_n = min(count - 1, int(round(count * (1.0 - self.coverage_fraction))))
        fair_n = count - weighted_n
        # Random tie-breaking among the currently least-exposed symbols.
        order = np.lexsort((self.rng.random(self.n_symbols), self.exposure))
        fair = order[:fair_n]
        remaining = np.ones(self.n_symbols, dtype=bool)
        remaining[fair] = False
        if weighted_n:
            pool = np.flatnonzero(remaining)
            weights = self.volume_weights[pool]
            weighted = self.rng.choice(pool, size=weighted_n, replace=False, p=weights / weights.sum())
            chosen = np.concatenate((fair, weighted))
        else:
            chosen = fair
        self.exposure[chosen] += 1
        return chosen.astype(np.int64)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, exposure=self.exposure)
        tmp.replace(path)


class CommonMarketTrainingLoader:
    """Build and sample a shared chronological market panel from source adapters."""
    schema_version = 3

    def __init__(self, cache_dir: str | Path, source_paths: Sequence[str | Path],
                 adapters: Sequence[MarketSourceAdapter] | None = None,
                 chunksize: int = 500_000, seed: int = 7,
                 train_fraction: float = .70, validation_fraction: float = .15,
                 context_stale_seconds: int = 60):
        self.cache_dir = Path(cache_dir)
        self.source_paths = [Path(p) for p in source_paths]
        if not self.source_paths:
            raise ValueError("at least one market source is required")
        self.adapters = list(adapters or [NormalizedBarAdapter() for _ in self.source_paths])
        if len(self.adapters) != len(self.source_paths):
            raise ValueError("one adapter is required for every input source")
        self.chunksize = int(chunksize)
        self.seed = int(seed)
        self.train_fraction = float(train_fraction)
        self.validation_fraction = float(validation_fraction)
        self.context_stale_seconds = int(context_stale_seconds)
        self._sample_batches_since_save = 0
        if not (0 < self.train_fraction < 1 and 0 < self.validation_fraction < 1 and
                self.train_fraction + self.validation_fraction < 1):
            raise ValueError("train and validation fractions must leave a nonempty test split")
        self.manifest_path = self.cache_dir / "manifest.json"

    def _source_manifest(self) -> list[dict]:
        rows=[]
        for path,adapter in zip(self.source_paths,self.adapters):
            stat=path.stat()
            rows.append({"path":str(path.resolve()),"adapter":adapter.name,
                         "timeframe_seconds":adapter.timeframe_seconds,
                         "bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns})
        return rows

    def _chunks(self, source_ix: int) -> Iterator[pd.DataFrame]:
        path, adapter = self.source_paths[source_ix], self.adapters[source_ix]
        for raw in adapter.iter_chunks(path, self.chunksize):
            normalized = adapter.normalize(raw)
            if len(normalized):
                yield normalized

    def build(self, force: bool = False) -> dict:
        if self.manifest_path.exists() and not force:
            meta = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if (meta.get("schema_version") == self.schema_version and
                    meta.get("sources") == self._source_manifest()):
                self._open(meta)
                return meta
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Do not let a previous complete manifest make an interrupted rebuild
        # look valid after its memory-mapped arrays have been truncated.
        self.manifest_path.unlink(missing_ok=True)
        (self.cache_dir / "coverage_exposure.npz").unlink(missing_ok=True)
        symbols: set[str] = set()
        times: set[int] = set()
        rows_seen = 0
        print("market_loader_pass=1 scan all sources for symbol map and time split", flush=True)
        for si in range(len(self.source_paths)):
            for frame in self._chunks(si):
                symbols.update(frame.instrument.unique().tolist())
                times.update(int(x) for x in frame.time_key.unique())
                rows_seen += len(frame)
                if rows_seen and rows_seen % 5_000_000 < self.chunksize:
                    print(f"scanned_rows={rows_seen:,} symbols={len(symbols):,} time_keys={len(times):,}", flush=True)
        symbol_names = sorted(symbols)
        time_keys = np.asarray(sorted(times), dtype=np.int64)
        if not len(symbol_names) or not len(time_keys):
            raise ValueError("market sources contain no usable observations")
        symbol_ids = {name: ix for ix, name in enumerate(symbol_names)}
        time_ids = {int(key): ix for ix, key in enumerate(time_keys)}
        n_times, n_symbols = len(time_keys), len(symbol_names)
        n_features = len(GLOBAL_FEATURES)
        estimate = n_times * n_symbols * (n_features * 2 + 4 + 1)
        train_cut = int(n_times * self.train_fraction)
        validation_cut = int(n_times * (self.train_fraction + self.validation_fraction))
        planned_shapes = {str(k): [1, 128, k, n_features]
                          for k in (64, 128, 256) if k <= n_symbols}
        print(f"shared_market_panel rows={rows_seen:,} time_rows={n_times:,} instruments={n_symbols:,} "
              f"planned_training_shapes={json.dumps(planned_shapes)} "
              f"split_indices=train[0,{train_cut}) validation[{train_cut},{validation_cut}) "
              f"test[{validation_cut},{n_times}) disk_estimate_bytes={estimate:,}", flush=True)
        self._create_arrays(n_times, n_symbols, n_features)
        market_names = {name: name.split("|", 1)[0] for name in symbol_names}
        asset_names = {name: name.split("|", 2)[1] for name in symbol_names}
        market_ids = np.asarray([stable_id(market_names[s], 64) for s in symbol_names], np.int64)
        asset_ids = np.asarray([stable_id(asset_names[s], 32) for s in symbol_names], np.int64)
        np.save(self.cache_dir / "time_keys.npy", time_keys, allow_pickle=False)
        np.save(self.cache_dir / "market_ids.npy", market_ids, allow_pickle=False)
        np.save(self.cache_dir / "asset_ids.npy", asset_ids, allow_pickle=False)
        symbol_map_path = self.cache_dir / "symbol_map.json"
        symbol_map_path.write_text(json.dumps({s: symbol_ids[s] for s in symbol_names}, indent=2), encoding="utf-8")

        print(f"market_loader_pass=2 build shared features/context rows={rows_seen:,} "
              f"times={n_times:,} symbols={n_symbols:,} disk_estimate_bytes={estimate:,}", flush=True)
        training_volumes = self._fill_arrays(symbol_ids, time_ids, time_keys, market_ids, asset_ids, train_cut)
        weights = np.sqrt(np.maximum(training_volumes, 0.0) + 1.0)
        # Use training-only volume weights to avoid looking at validation/test.
        weights = weights / weights.sum()
        self.coverage = CoverageSampler(n_symbols, weights, self.seed)
        np.savez_compressed(self.cache_dir / "symbol_weights.npz", weights=weights)
        train_end = train_cut
        valid_end = validation_cut
        meta = {
            "schema_version": self.schema_version,
            "sources": self._source_manifest(),
            "rows": rows_seen, "times": n_times, "symbols": n_symbols,
            "feature_names": list(GLOBAL_FEATURES), "context_names": list(CONTEXT_FEATURES),
            "train_fraction": self.train_fraction, "validation_fraction": self.validation_fraction,
            "time_splits": {"train": [0, train_end], "validation": [train_end, valid_end],
                            "test": [valid_end, n_times]},
            "memory_mapped_bytes_estimate": estimate,
            "symbol_map": str(symbol_map_path), "cache_dir": str(self.cache_dir),
            "timestamps_utc": [pd.to_datetime(int(time_keys[0]), unit="s", utc=True).isoformat(),
                                pd.to_datetime(int(time_keys[-1]), unit="s", utc=True).isoformat()],
        }
        self.manifest_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._open(meta)
        return meta

    def _create_arrays(self, t: int, n: int, f: int) -> None:
        for filename, dtype, shape in (
            ("features.f16", np.float16, (t, n, f)),
            ("closes.f32", np.float32, (t, n)),
            ("observed.u8", np.uint8, (t, n)),
            ("global_context.f16", np.float16, (t, len(CONTEXT_FEATURES))),
        ):
            path = self.cache_dir / filename
            with path.open("wb") as stream:
                stream.truncate(int(np.prod(shape)) * np.dtype(dtype).itemsize)
        self.features = np.memmap(self.cache_dir / "features.f16", dtype=np.float16, mode="r+", shape=(t,n,f))
        self.closes = np.memmap(self.cache_dir / "closes.f32", dtype=np.float32, mode="r+", shape=(t,n))
        self.observed = np.memmap(self.cache_dir / "observed.u8", dtype=np.uint8, mode="r+", shape=(t,n))
        self.context = np.memmap(self.cache_dir / "global_context.f16", dtype=np.float16, mode="r+", shape=(t,len(CONTEXT_FEATURES)))

    def _open(self, meta: dict) -> None:
        n, t, f = int(meta["symbols"]), int(meta["times"]), len(GLOBAL_FEATURES)
        self.features = np.memmap(self.cache_dir / "features.f16", dtype=np.float16, mode="r", shape=(t,n,f))
        self.closes = np.memmap(self.cache_dir / "closes.f32", dtype=np.float32, mode="r", shape=(t,n))
        self.observed = np.memmap(self.cache_dir / "observed.u8", dtype=np.uint8, mode="r", shape=(t,n))
        self.context = np.memmap(self.cache_dir / "global_context.f16", dtype=np.float16, mode="r", shape=(t,len(CONTEXT_FEATURES)))
        self.time_keys = np.load(self.cache_dir / "time_keys.npy", mmap_mode="r", allow_pickle=False)
        self.symbol_map = json.loads((self.cache_dir / "symbol_map.json").read_text(encoding="utf-8"))
        self.symbol_names = [name for name, _ in sorted(self.symbol_map.items(), key=lambda kv: kv[1])]
        self.market_ids = np.load(self.cache_dir / "market_ids.npy", mmap_mode="r", allow_pickle=False)
        self.asset_ids = np.load(self.cache_dir / "asset_ids.npy", mmap_mode="r", allow_pickle=False)
        weights = np.load(self.cache_dir / "symbol_weights.npz", allow_pickle=False)["weights"]
        exposure_path = self.cache_dir / "coverage_exposure.npz"
        self.coverage = CoverageSampler(n, weights, self.seed)
        if exposure_path.exists():
            with np.load(exposure_path, allow_pickle=False) as state:
                self.coverage.exposure[:] = state["exposure"]
        self.splits = meta["time_splits"]

    def _time_groups(self, source_ix: int) -> Iterator[tuple[int, pd.DataFrame]]:
        carry = None
        last = None
        for chunk in self._chunks(source_ix):
            if last is not None and int(chunk.time_key.iloc[0]) < last:
                raise ValueError(f"{self.source_paths[source_ix]} is not chronologically sorted")
            last = int(chunk.time_key.iloc[-1])
            if carry is not None:
                chunk = pd.concat((carry, chunk), ignore_index=True)
            keys = chunk.time_key.to_numpy(np.int64)
            if not len(keys):
                continue
            boundary = np.flatnonzero(keys[1:] != keys[:-1]) + 1
            starts = np.r_[0, boundary]
            ends = np.r_[boundary, len(chunk)]
            # Carry the final group since another row with the same key may be
            # the first row of the next CSV chunk.
            complete_end = len(starts) - 1
            for j in range(complete_end):
                a, b = int(starts[j]), int(ends[j])
                yield int(keys[a]), chunk.iloc[a:b]
            carry = chunk.iloc[int(starts[-1]):].copy()
        if carry is not None and len(carry):
            yield int(carry.time_key.iloc[0]), carry

    def _merged_time_groups(self) -> Iterator[tuple[int, pd.DataFrame]]:
        streams = [iter(self._time_groups(i)) for i in range(len(self.source_paths))]
        heap: list[tuple[int,int,pd.DataFrame]] = []
        for ix, stream in enumerate(streams):
            try:
                key, frame = next(stream); heapq.heappush(heap, (key, ix, frame))
            except StopIteration:
                pass
        while heap:
            key = heap[0][0]; frames = []
            while heap and heap[0][0] == key:
                _, ix, frame = heapq.heappop(heap); frames.append(frame)
                try:
                    next_key, next_frame = next(streams[ix])
                    heapq.heappush(heap, (next_key, ix, next_frame))
                except StopIteration:
                    pass
            yield key, pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    def _fill_arrays(self, symbol_ids: dict[str,int], time_ids: dict[int,int], time_keys: np.ndarray,
                     market_ids: np.ndarray, asset_ids: np.ndarray, training_end: int) -> np.ndarray:
        n, f = len(symbol_ids), len(GLOBAL_FEATURES)
        state_features = np.zeros((n,f), np.float32)
        state_close = np.full(n, np.nan, np.float32)
        state_spread = np.zeros(n, np.float32)
        state_book_imb = np.zeros(n, np.float32)
        state_return = np.zeros(n, np.float32)
        state_volatility = np.zeros(n, np.float32)
        last_seen = np.full(n, -10**12, np.int64)
        price_history = [deque(maxlen=21) for _ in range(n)]
        return_history = [deque(maxlen=20) for _ in range(n)]
        log_volume_history = [deque(maxlen=20) for _ in range(n)]
        trade_history = [deque(maxlen=20) for _ in range(n)]
        previous_volume = np.zeros(n, np.float32)
        previous_open_interest = np.zeros(n, np.float32)
        training_volumes = np.zeros(n, np.float64)
        last_written = -1
        for time_key, group in self._merged_time_groups():
            ti = time_ids[int(time_key)]
            if ti <= last_written:
                raise ValueError("merged source timeline is not strictly chronological")
            last_written = ti
            current_ids = []
            current_returns = []
            current_volumes = []
            current_net_flows = []
            # Collapse multiple venue/provider rows for an instrument in this
            # global time bucket; latest quote, summed flow/volume.
            group = group.sort_values(["instrument", "event_time_ns"], kind="mergesort")
            agg = group.groupby("instrument", sort=False).agg(
                event_time_ns=("event_time_ns", "last"), close=("close", "last"),
                volume=("volume", "sum"), bid=("bid", "last"), ask=("ask", "last"),
                bid_size=("bid_size", "last"), ask_size=("ask_size", "last"),
                buy_volume=("buy_volume", "sum"), sell_volume=("sell_volume", "sum"),
                trade_count=("trade_count", "sum"), high=("high", "max"), low=("low", "min"),
                implied_volatility=("implied_volatility", "last"),
                open_interest=("open_interest", "last"),
                open_interest_change=("open_interest_change", "last"),
                yield_change=("yield_change", "last"), days_to_expiry=("days_to_expiry", "last"),
                video_chart_signal=("video_chart_signal", "last"),
                video_volume_signal=("video_volume_signal", "last"),
            )
            # Named tuples avoid constructing a pandas Series for every
            # instrument-time row (the ITCH source contains tens of millions).
            for row in agg.reset_index().itertuples(index=False):
                sid = symbol_ids[str(row.instrument)]
                close = float(row.close)
                old_close = float(state_close[sid])
                ret = close / old_close - 1.0 if np.isfinite(old_close) and old_close > 0 else 0.0
                ph = price_history[sid]
                if len(ph) >= 5 and ph[-5] > 0: ret5 = close / ph[-5] - 1.0
                else: ret5 = 0.0
                if len(ph) >= 20 and ph[-20] > 0: ret20 = close / ph[-20] - 1.0
                else: ret20 = 0.0
                ph.append(close)
                rh = return_history[sid]; rh.append(ret)
                vol = max(float(row.volume), 0.0); lv = math.log1p(vol)
                vh = log_volume_history[sid]
                volume_z = 0.0 if len(vh) < 2 else (lv - float(np.mean(vh))) / max(float(np.std(vh, ddof=1)), 1e-8)
                vh.append(lv)
                deltas = np.diff(np.asarray(ph, np.float64))[-14:]
                gains = np.maximum(deltas, 0); losses = -np.minimum(deltas, 0)
                gain = float(gains.mean()) if len(gains) else 0.0
                loss = float(losses.mean()) if len(losses) else 0.0
                rsi = .5 if gain + loss == 0 else gain / (gain + loss)
                ret_vol = float(np.std(rh, ddof=1)) if len(rh) > 1 else 0.0
                bid, ask = float(row.bid), float(row.ask)
                mid = max((bid + ask) / 2.0, 1e-12)
                spread_bps = (ask - bid) / mid * 10_000.0 if bid > 0 and ask > 0 else 0.0
                bs, ass = float(row.bid_size), float(row.ask_size)
                book_imb = (bs - ass) / (bs + ass) if bs + ass > 0 else 0.0
                buy, sell = float(row.buy_volume), float(row.sell_volume)
                trade_imb = (buy - sell) / (buy + sell) if buy + sell > 0 else 0.0
                tc = float(row.trade_count); th = trade_history[sid]
                intensity = tc / (float(np.mean(th)) + 1e-8) - 1.0 if th else 0.0
                th.append(tc)
                oi = float(row.open_interest)
                oi_delta = float(row.open_interest_change)
                if oi_delta == 0.0 and previous_open_interest[sid] > 0 and oi > 0:
                    oi_delta = oi / previous_open_interest[sid] - 1.0
                if oi > 0:
                    previous_open_interest[sid] = oi
                features = np.asarray((ret,ret5,ret20,
                    max(float(row.high)-float(row.low),0.0)/max(close,1e-12),volume_z,rsi,ret_vol,
                    spread_bps,book_imb,trade_imb,intensity,float(row.implied_volatility),oi_delta,
                    float(row.yield_change),float(row.days_to_expiry),float(row.video_chart_signal),
                    float(row.video_volume_signal)),np.float32)
                np.nan_to_num(features,copy=False,nan=0.0,posinf=0.0,neginf=0.0)
                np.clip(features,-10,10,out=features)
                state_features[sid] = features
                state_close[sid] = close
                state_spread[sid] = spread_bps
                state_book_imb[sid] = book_imb
                state_return[sid] = ret
                state_volatility[sid] = ret_vol
                last_seen[sid] = ti
                previous_volume[sid] = vol
                current_ids.append(sid); current_returns.append(ret); current_volumes.append(vol)
                current_net_flows.append(buy - sell)
            if not current_ids:
                continue
            current_ids_a = np.asarray(current_ids, np.int64)
            cr = np.asarray(current_returns, np.float64)
            cv = np.asarray(current_volumes, np.float64)
            cf = np.asarray(current_net_flows, np.float64)
            if ti < training_end:
                np.add.at(training_volumes, current_ids_a, cv)
            seen = np.isfinite(state_close)
            fresh = seen & ((ti - last_seen) <= self.context_stale_seconds)
            freshr = state_return[fresh].astype(np.float64)
            fresh_spread = state_spread[fresh].astype(np.float64)
            fresh_imb = state_book_imb[fresh].astype(np.float64)
            total_volume = float(cv.sum())
            total_net = float(cf.sum())
            top10 = float(np.sort(cv)[-10:].sum() / max(total_volume,1.0)) if len(cv) else 0.0
            ctx = np.asarray((
                seen.mean(), fresh.mean(), float((cr>0).mean()), float((cr<0).mean()),
                float(cr.mean()*100), float(cr.std()*100),
                float(freshr.mean()*100) if len(freshr) else 0.0,
                float(np.median(freshr)*100) if len(freshr) else 0.0,
                float(freshr.std()*100) if len(freshr) else 0.0,
                math.log1p(total_volume)/20.0,
                total_net/max(float(cv.sum()),1.0),
                float(np.median(fresh_spread))/100.0 if len(fresh_spread) else 0.0,
                float(np.quantile(fresh_spread,.9))/100.0 if len(fresh_spread) else 0.0,
                float(fresh_imb.mean()) if len(fresh_imb) else 0.0,
                top10,
                float(np.median(state_volatility[fresh]))*100 if fresh.any() else 0.0,
            ),np.float32)
            np.clip(np.nan_to_num(ctx,nan=0.0,posinf=0.0,neginf=0.0),-10,10,out=ctx)
            self.features[ti] = state_features.astype(np.float16)
            self.closes[ti] = state_close
            self.observed[ti] = 0
            self.observed[ti,current_ids_a] = 1
            self.context[ti] = ctx.astype(np.float16)
            if ti and ti % 1000 == 0:
                print(f"built_time_rows={ti:,}/{len(time_keys):,} updated_symbols={len(current_ids_a):,}",flush=True)
        for arr in (self.features,self.closes,self.observed,self.context):
            arr.flush()
        return training_volumes

    def save_coverage(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.cache_dir / "coverage_exposure.npz", exposure=self.coverage.exposure)

    def sample(self, batch_size: int = 1, symbols_per_sample: int = 128,
               sequence_length: int = 128, split: str = "train", horizon_steps: int = 1,
               horizon_seconds: int | None = None,
               device: str | torch.device = "cpu",
               fixed_end_indices: Sequence[int] | None = None) -> MarketBatch:
        if sequence_length > 128:
            raise ValueError("sequence_length cannot exceed the checkpoint maximum of 128")
        if horizon_seconds is None and horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        if split not in self.splits:
            raise ValueError(f"unknown split {split!r}")
        lo, hi = map(int,self.splits[split])
        first_end = lo + sequence_length - 1
        if first_end >= hi:
            raise ValueError(f"split {split} is shorter than the requested sequence")
        possible = np.arange(first_end, hi, dtype=np.int64)
        if horizon_seconds is None:
            possible = possible[possible + horizon_steps < hi]
            targets = possible + horizon_steps
        else:
            if horizon_seconds < 1:
                raise ValueError("horizon_seconds must be >= 1")
            target_times = np.asarray(self.time_keys[possible], dtype=np.int64) + int(horizon_seconds)
            targets = np.searchsorted(self.time_keys, target_times, side="left").astype(np.int64)
            eligible = targets < hi
            possible, targets = possible[eligible], targets[eligible]
        if not len(possible):
            raise ValueError(f"split {split} has no sequence with a target inside its time boundary")
        if fixed_end_indices is not None:
            selected = np.asarray(fixed_end_indices, dtype=np.int64).reshape(-1)
            if len(selected) != batch_size or not np.isin(selected, possible).all():
                raise ValueError("end_indices must contain one eligible timestamp per batch item")
            target_lookup = {int(e): int(t) for e, t in zip(possible, targets)}
        x_rows=[]; mask_rows=[]; ctx_rows=[]; target_rows=[]; target_masks=[]; end_indices=[]; target_indices=[]; sampled=[]
        for bi in range(batch_size):
            if fixed_end_indices is None:
                pick = int(self.coverage.rng.integers(0, len(possible)))
                end = int(possible[pick])
                target_end = int(targets[pick])
            else:
                end = int(selected[bi])
                target_end = target_lookup[end]
            ids = self.coverage.choose(symbols_per_sample)
            ti = np.arange(end-sequence_length+1,end+1,dtype=np.int64)
            x_rows.append(np.asarray(self.features[ti[:,None],ids[None,:],:],dtype=np.float16))
            mask_rows.append(np.asarray(self.observed[ti[:,None],ids[None,:]],dtype=np.bool_))
            ctx_rows.append(np.asarray(self.context[ti],dtype=np.float16))
            c0=np.asarray(self.closes[end,ids],dtype=np.float32)
            c1=np.asarray(self.closes[target_end,ids],dtype=np.float32)
            valid=np.isfinite(c0)&np.isfinite(c1)&(c0>0)&(c1>0)
            # Require a real observation at both endpoints; stale forward fills
            # remain context only and cannot manufacture a training reward.
            valid &= np.asarray(self.observed[end,ids],dtype=bool)
            valid &= np.asarray(self.observed[target_end,ids],dtype=bool)
            ret=np.zeros(symbols_per_sample,np.float32)
            ret[valid]=c1[valid]/c0[valid]-1.0
            target_rows.append(ret); target_masks.append(valid)
            end_indices.append(end); target_indices.append(target_end); sampled.append(ids)
        dev=torch.device(device)
        batch=MarketBatch(
            torch.as_tensor(np.stack(x_rows),device=dev),
            torch.as_tensor(np.stack(sampled),device=dev,dtype=torch.long),
            torch.as_tensor(np.stack([self.market_ids[x] for x in sampled]),device=dev,dtype=torch.long),
            torch.as_tensor(np.stack([self.asset_ids[x] for x in sampled]),device=dev,dtype=torch.long),
            torch.as_tensor(np.stack(mask_rows),device=dev,dtype=torch.bool),
            torch.as_tensor(np.stack(ctx_rows),device=dev),
            torch.as_tensor(np.stack(target_rows),device=dev),
            torch.as_tensor(np.stack(target_masks),device=dev,dtype=torch.bool),
            np.asarray(end_indices,np.int64),np.asarray(target_indices,np.int64),sampled)
        self._sample_batches_since_save += batch_size
        if self._sample_batches_since_save >= 100:
            self.save_coverage()
            self._sample_batches_since_save = 0
        return batch

    def split_report(self) -> dict:
        out={}
        for name,(lo,hi) in self.splits.items():
            lo,hi=int(lo),int(hi)
            out[name]={"indices":[lo,hi],"time_keys":[int(self.time_keys[lo]),int(self.time_keys[hi-1])],
                       "utc":[pd.to_datetime(int(self.time_keys[lo]),unit="s",utc=True).isoformat(),
                              pd.to_datetime(int(self.time_keys[hi-1]),unit="s",utc=True).isoformat()]}
        return out

    def coverage_report(self) -> dict:
        c=self.coverage.exposure
        return {"symbols":int(len(c)),"sampled_symbols":int((c>0).sum()),"min_exposure":int(c.min()),
                "mean_exposure":float(c.mean()),"p10_exposure":float(np.quantile(c,.1)),
                "median_exposure":float(np.median(c)),"p90_exposure":float(np.quantile(c,.9)),
                "max_exposure":int(c.max()),
                "zero_exposure":int((c==0).sum()),"exposure_by_symbol":{
                    name:int(c[ix]) for ix,name in enumerate(self.symbol_names)}}

    def close(self) -> None:
        """Release Windows file mappings so cache directories can be moved/cleaned."""
        for name in ("features", "closes", "observed", "context", "time_keys",
                     "market_ids", "asset_ids"):
            value = getattr(self, name, None)
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()
            if hasattr(self, name):
                delattr(self, name)
