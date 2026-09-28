"""Train a scale-aware isolated candidate from FinCast outputs and net outcomes.

Four scales are represented explicitly: 30 seconds, 5 minutes, 1 hour, and
5 trading days. Intraday scales use corrected ITCH snapshots; the daily scale
uses the project's real global daily OHLCV panel. This script never writes the
existing champion or prior distillation runtimes, and never reads test labels.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import distill_fincast_offline as base
from stockrl.global_transformer import (GlobalMarketTransformer, TransformerConfig,
                                        TIME_SCALE_NAMES, load_compatible_state_dict,
                                        parameter_count)
from stockrl.market_training import (CommonMarketTrainingLoader, ContextConditionedTransformer,
                                     ITCHSnapshotAdapter, NormalizedBarAdapter,
                                     load_market_candidate_checkpoint)
from bench_market_training_loader import batch_tensors, utility

DEFAULT_RUNTIME = ROOT / "runtime-global-market-training/fincast-temporal-scale-distillation"
DEFAULT_DAILY = ROOT / "data/global_market_daily.csv"
SCALES = (
    {"id": 0, "name": "seconds_30", "source": "itch", "stride": 1, "horizon": 30, "teacher_h": 29, "freq": 0},
    {"id": 1, "name": "minutes_5", "source": "itch", "stride": 10, "horizon": 300, "teacher_h": 29, "freq": 0},
    {"id": 2, "name": "hours_1", "source": "itch", "stride": 60, "horizon": 3600, "teacher_h": 59, "freq": 0},
    {"id": 3, "name": "days_5_sessions", "source": "daily", "stride": 1, "horizon": 5, "teacher_h": 4, "freq": 0},
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, allow_nan=False, default=str) + "\n")


def scale_symbol_map(original: dict[str, int], daily: dict[str, int]) -> tuple[dict[str, int], dict[int, int]]:
    """Share IDs by unique ticker; append daily-only instruments deterministically."""
    by_ticker: dict[str, list[int]] = {}
    for name, ix in original.items():
        by_ticker.setdefault(name.rsplit("|", 1)[-1], []).append(int(ix))
    combined = dict(original)
    next_id = max(map(int, original.values()), default=-1) + 1
    daily_to_model: dict[int, int] = {}
    for name, local_id in sorted(daily.items(), key=lambda x: x[1]):
        ticker = name.rsplit("|", 1)[-1]
        matches = by_ticker.get(ticker, [])
        if len(matches) == 1:
            mapped = matches[0]
        else:
            if name not in combined:
                combined[name] = next_id
                next_id += 1
            mapped = combined[name]
        daily_to_model[int(local_id)] = int(mapped)
    return combined, daily_to_model


def make_candidate(champion_path: Path, candidate_path: Path, daily_map: dict[str, int]) -> tuple[dict, dict[int, int]]:
    source = torch.load(champion_path, map_location="cpu", weights_only=False)
    old_cfg = TransformerConfig(**source["config"])
    combined_map, daily_id_map = scale_symbol_map(source["symbol_map"], daily_map)
    new_cfg = replace(old_cfg, max_symbols=max(old_cfg.max_symbols, len(combined_map)))
    model = ContextConditionedTransformer(GlobalMarketTransformer(new_cfg).half())
    model.context_policy.float(); model.context_value.float()
    current = model.state_dict()
    old = source["state_dict"]
    for key, value in old.items():
        if key == "backbone.symbol_embedding.weight":
            current[key][:value.shape[0]].copy_(value)
            if current[key].shape[0] > value.shape[0]:
                current[key][value.shape[0]:].copy_(value.float().mean(0, keepdim=True).half())
        else:
            current[key] = value
    load_compatible_state_dict(model, current, strict=True)
    output = dict(source)
    output.update({"config": asdict(new_cfg), "state_dict": model.state_dict(),
                   "symbol_map": combined_map,
                   "source_champion_sha256": sha256(champion_path),
                   "temporal_scale_names": list(TIME_SCALE_NAMES),
                   "temporal_scale_training": "initialized; awaiting FinCast distillation"})
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, candidate_path)
    del source, model
    return output, daily_id_map


def get_loader(path: Path, cache: Path, adapter, seed: int):
    obj = CommonMarketTrainingLoader(cache, [path], [adapter], chunksize=250_000,
                                     seed=seed, train_fraction=.70,
                                     validation_fraction=.15,
                                     context_stale_seconds=60)
    return obj, obj.build(force=False)


def choose_ends(loader, split: str, count: int, seq: int, stride: int,
                horizon: int, daily: bool, seed: int) -> np.ndarray:
    lo, hi = map(int, loader.splits[split])
    # Validation/test observations may use the chronologically earlier history
    # as context; only their future targets must stay inside their own split.
    starts = max((seq - 1) * stride, lo if split == "train" else 0)
    candidates = np.arange(starts, hi, dtype=np.int64)
    if daily:
        good = candidates + horizon < hi
        candidates = candidates[good]
    else:
        targets = np.searchsorted(loader.time_keys,
                                  np.asarray(loader.time_keys[candidates], np.int64) + horizon,
                                  side="left")
        candidates = candidates[targets < hi]
    if not len(candidates):
        raise RuntimeError(f"no eligible {split} examples at horizon={horizon}")
    rng = np.random.default_rng(seed)
    take = min(count, len(candidates))
    return np.sort(rng.choice(candidates, take, replace=False))


def assemble_examples(loader, split: str, count: int, seq: int, stride: int,
                      horizon: int, daily: bool, seed: int,
                      daily_id_map: dict[int, int] | None = None) -> list[dict]:
    ends = choose_ends(loader, split, count, seq, stride, horizon, daily, seed)
    n_symbols = min(128, len(loader.symbol_names))
    original_exposure = loader.coverage.exposure.copy()
    rng0 = loader.coverage.rng
    loader.coverage.rng = np.random.default_rng(seed + 1)
    out: list[dict] = []
    for end in ends:
        local_ids = loader.coverage.choose(n_symbols)
        if daily:
            target_ix = int(end) + horizon
        else:
            target_ix = int(np.searchsorted(loader.time_keys,
                             int(loader.time_keys[end]) + horizon, side="left"))
        close_now = np.asarray(loader.closes[end, local_ids], np.float32)
        close_future = np.asarray(loader.closes[target_ix, local_ids], np.float32)
        observed_now = np.asarray(loader.observed[end, local_ids], bool)
        observed_future = np.asarray(loader.observed[target_ix, local_ids], bool)
        valid = observed_now & observed_future & np.isfinite(close_now) & np.isfinite(close_future) & (close_now > 0)
        returns = np.zeros(n_symbols, np.float32)
        returns[valid] = close_future[valid] / close_now[valid] - 1.0
        tix = int(end) - (seq - 1) * stride + np.arange(seq, dtype=np.int64) * stride
        series_by_symbol = []
        teacher_valid = np.zeros(n_symbols, bool)
        for j in range(n_symbols):
            series = np.asarray(loader.closes[tix, local_ids[j]], np.float32)
            good = np.isfinite(series) & (series > 0)
            if valid[j] and good.sum() >= 32:
                series_by_symbol.append((j, series[np.flatnonzero(good)[0]:].astype(np.float32), float(close_now[j])))
                teacher_valid[j] = True
        if daily_id_map is None:
            model_ids = local_ids.copy()
        else:
            model_ids = np.asarray([daily_id_map[int(x)] for x in local_ids], np.int64)
        out.append({"end": int(end), "local_ids": local_ids, "model_ids": model_ids,
                    "returns": returns, "valid": valid, "teacher_valid": teacher_valid,
                    "teacher_probs": np.full((n_symbols, 3), 1 / 3, np.float32),
                    "teacher_value": np.zeros(n_symbols, np.float32),
                    "teacher_q10": np.zeros(n_symbols, np.float32),
                    "teacher_q50": np.zeros(n_symbols, np.float32),
                    "teacher_q90": np.zeros(n_symbols, np.float32),
                    "series": series_by_symbol, "daily": daily, "stride": stride})
    loader.coverage.exposure[:] = original_exposure
    loader.coverage.rng = rng0
    return out


def forecast_cache(teacher, examples: list[dict], scale: dict, seq: int,
                   batch_size: int, cost: float, device: torch.device) -> dict:
    records = [(i, j, s, last) for i, ex in enumerate(examples)
               for j, s, last in ex["series"]]
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for off in range(0, len(records), batch_size):
        group = records[off:off + batch_size]
        _, pred = teacher.forecast([x[2] for x in group], freq=[scale["freq"]] * len(group),
                                   forecast_context_len=seq, normalize=False)
        horizon_ix = min(scale["teacher_h"], pred.shape[1] - 1)
        q = pred[:, horizon_ix, :]
        last = np.asarray([x[3] for x in group], np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            lo, med, hi = q[:, 1].astype(np.float64) / last - 1.0, q[:, 5].astype(np.float64) / last - 1.0, q[:, 9].astype(np.float64) / last - 1.0
        good = np.isfinite(lo) & np.isfinite(med) & np.isfinite(hi)
        lo, med, hi = np.clip(lo, -.25, .25), np.clip(med, -.25, .25), np.clip(hi, -.25, .25)
        probs, value = base.quantile_action_targets(lo, med, hi, cost)
        for k, (i, j, _, _) in enumerate(group):
            if not good[k]:
                examples[i]["teacher_valid"][j] = False
                continue
            ex = examples[i]
            ex["teacher_probs"][j] = probs[k]
            ex["teacher_value"][j] = value[k]
            ex["teacher_q10"][j], ex["teacher_q50"][j], ex["teacher_q90"][j] = lo[k], med[k], hi[k]
        if off == 0 or off + len(group) == len(records) or off // batch_size % 50 == 0:
            torch.cuda.synchronize(device)
            print(f"teacher_scale={scale['name']} series={min(off+len(group),len(records))}/{len(records)} elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return {"series": len(records), "seconds": time.perf_counter() - t0,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device))}


def save_scale_cache(path: Path, examples: list[dict], meta: dict):
    arrays = {key: np.stack([x[key] for x in examples]) for key in
              ("local_ids", "model_ids", "returns", "valid", "teacher_valid", "teacher_probs",
               "teacher_value", "teacher_q10", "teacher_q50", "teacher_q90")}
    arrays["end_indices"] = np.asarray([x["end"] for x in examples], np.int64)
    with path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")


def make_batch(loader, row: dict, scale_id: int, device: torch.device):
    end, ids = int(row["end"]), row["local_ids"]
    daily = row["daily"]; stride = row["stride"]
    seq = 128
    tix = end - (seq - 1) * stride + np.arange(seq, dtype=np.int64) * stride
    feat = np.asarray(loader.features[tix[:, None], ids[None, :]], np.float32).copy()
    close = np.asarray(loader.closes[tix[:, None], ids[None, :]], np.float32)
    # Recompute the return-derived channels in units of the selected bar scale.
    ratio = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0
    ret = np.zeros_like(close); ret[1:] = ratio
    feat[..., 0] = ret
    feat[..., 1] = np.concatenate((np.zeros_like(ret[:5]), close[5:] / np.maximum(close[:-5], 1e-12) - 1.0), axis=0)
    feat[..., 2] = np.concatenate((np.zeros_like(ret[:20]), close[20:] / np.maximum(close[:-20], 1e-12) - 1.0), axis=0)
    for k in range(seq):
        a = max(0, k - 13); r = ret[a:k + 1]
        gain = np.maximum(r, 0).mean(axis=0); loss = np.maximum(-r, 0).mean(axis=0)
        feat[k, :, 5] = np.where(gain + loss > 0, gain / np.maximum(gain + loss, 1e-12), .5)
        feat[k, :, 6] = ret[max(0, k - 19):k + 1].std(axis=0)
    valid = np.asarray(loader.observed[tix[:, None], ids[None, :]], bool)
    context = np.asarray(loader.context[tix], np.float32)
    model_ids = row["model_ids"]
    batch = base.MarketBatch(
        features=torch.as_tensor(feat[None], dtype=torch.float16, device=device),
        symbol_ids=torch.as_tensor(model_ids[None], dtype=torch.long, device=device),
        market_ids=torch.as_tensor(np.asarray(loader.market_ids[ids], np.int64)[None], device=device),
        asset_ids=torch.as_tensor(np.asarray(loader.asset_ids[ids], np.int64)[None], device=device),
        valid_mask=torch.as_tensor(valid[None], dtype=torch.bool, device=device),
        market_context=torch.as_tensor(context[None], dtype=torch.float32, device=device),
        target_return=torch.as_tensor(row["returns"][None], dtype=torch.float32, device=device),
        target_valid=torch.as_tensor(row["valid"][None], dtype=torch.bool, device=device),
        end_indices=np.asarray([end], np.int64), target_indices=np.asarray([0], np.int64),
        sampled_ids=[ids])
    return batch


def row_from_cache(cache: dict, ix: int, daily: bool, stride: int) -> dict:
    ids = cache.get("local_ids", cache.get("symbol_ids"))[ix]
    teacher_valid = cache.get("teacher_valid")
    if teacher_valid is None:
        probabilities = cache["teacher_probs"][ix]
        teacher_valid = cache["valid"][ix] & (np.max(np.abs(probabilities - 1.0 / 3.0), axis=-1) > 1e-5)
    else:
        teacher_valid = teacher_valid[ix]
    return {"end": int(cache["end_indices"][ix]), "local_ids": ids,
            "model_ids": cache.get("model_ids", cache.get("symbol_ids"))[ix],
            "returns": cache.get("returns", cache.get("target_return"))[ix],
            "valid": cache["valid"][ix], "teacher_valid": teacher_valid,
            "teacher_probs": cache["teacher_probs"][ix], "teacher_value": cache["teacher_value"][ix],
            "daily": daily, "stride": stride}


def evaluate(model, datasets, loaders, device):
    report = {}
    model.eval()
    with torch.inference_mode():
        for scale, rows in datasets.items():
            loader = loaders[scale]
            total = count = 0
            acts = np.zeros(3, np.int64)
            for row in rows:
                batch = make_batch(loader, row, scale, device)
                x, sid, mid, aid, mask, ctx, ret, valid = batch_tensors(batch, device)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits, _ = model(x, sid, mid, aid, mask, ctx,
                                      torch.tensor([scale], device=device))
                valid = valid[0] & mask[0, -1]
                if not valid.any(): continue
                _, pnl = utility(logits[0], ret[0], valid)
                decision = logits[0].float().argmax(-1)
                total += float(pnl.gather(1, decision[:, None]).squeeze(1)[valid].sum().cpu())
                count += int(valid.sum())
                for i in range(3): acts[i] += int(((decision == i) & valid).sum())
            report[TIME_SCALE_NAMES[scale]] = {"net_return_sum": total,
                "valid_decisions": count, "mean_net_return": total / max(count, 1),
                "actions": {"SELL": int(acts[0]), "HOLD": int(acts[1]), "BUY": int(acts[2])}}
    model.train()
    return report


def save_candidate(path, model, payload, optimizer, step):
    out = dict(payload)
    state = {}
    for key, value in model.state_dict().items():
        value = value.detach().cpu()
        if key.startswith("backbone.") and value.is_floating_point(): value = value.half()
        state[key] = value
    out.update({"state_dict": state, "optimizer_state": optimizer.state_dict(),
                "distillation_step": step})
    tmp = path.with_suffix(".pt.tmp"); torch.save(out, tmp); tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", type=Path, default=base.DEFAULT_TEACHER)
    ap.add_argument("--itch", type=Path, default=base.DEFAULT_SOURCE)
    ap.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    ap.add_argument("--champion", type=Path, default=base.DEFAULT_CHAMPION)
    ap.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    ap.add_argument("--itch-cache", type=Path, default=base.DEFAULT_LOADER_CACHE)
    ap.add_argument("--examples-per-scale", type=int, default=128)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--teacher-batch-size", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--validation-examples", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=0.0001)
    ap.add_argument("--policy-distill-weight", type=float, default=0.15)
    ap.add_argument("--value-distill-weight", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--round-trip-cost", type=float, default=0.0022)
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required for this 0.5B candidate run")
    if args.runtime.exists(): raise FileExistsError(f"Refusing to overwrite existing runtime: {args.runtime}")
    args.runtime.mkdir(parents=True)
    device = torch.device("cuda:0")
    old_cache_dir = ROOT / "runtime-global-market-training/fincast-distillation"
    old_short_path = old_cache_dir / "teacher_outputs_train.npz"
    old_short_meta_path = old_cache_dir / "teacher_outputs_train.json"
    if not old_short_path.exists() or not old_short_meta_path.exists():
        raise FileNotFoundError("verified 30-second FinCast cache is missing")
    short_meta = json.loads(old_short_meta_path.read_text(encoding="utf-8"))
    if short_meta.get("sequence_length") != 128 or short_meta.get("horizon_steps") != 30 or short_meta.get("test_accessed"):
        raise RuntimeError("existing short-scale cache does not match the required train-only 30s contract")

    itch_loader, itch_manifest = get_loader(args.itch, args.itch_cache, ITCHSnapshotAdapter(1), args.seed)
    daily_cache_dir = args.runtime / "daily-market-cache"
    daily_loader, daily_manifest = get_loader(args.daily, daily_cache_dir, NormalizedBarAdapter(86400), args.seed + 1)
    if itch_manifest["symbols"] != 8694:
        raise RuntimeError(f"expected corrected ITCH universe of 8,694; found {itch_manifest['symbols']}")
    initial_payload, daily_id_map = make_candidate(args.champion, args.runtime / "candidate.pt", daily_loader.symbol_map)
    champion_hash = sha256(args.champion)
    if initial_payload["source_champion_sha256"] != champion_hash:
        raise RuntimeError("candidate does not derive from the unchanged champion")

    datasets: dict[int, list[dict]] = {}
    loaders: dict[int, CommonMarketTrainingLoader] = {}
    cache_metas = {}
    # Scale 0 reuses the already generated and verified train-only 30s cache.
    with np.load(old_short_path, allow_pickle=False) as z:
        short_arrays = {k: z[k] for k in z.files}
    datasets[0] = [row_from_cache(short_arrays, i, False, 1) for i in range(len(short_arrays["end_indices"]))]
    loaders[0] = itch_loader
    cache_metas[0] = {"cache": str(old_short_path), "source": str(args.itch),
                      "source_sha256": short_meta["source_sha256"],
                      "examples": len(datasets[0]), "series": int(short_meta["valid_teacher_series"]),
                      "cached_teacher_reused": True}

    train_spec = {1: (itch_loader, "minutes_5_train.npz", False),
                  2: (itch_loader, "hours_1_train.npz", False),
                  3: (daily_loader, "days_5_sessions_train.npz", True)}
    val_spec = {0: (itch_loader, False), 1: (itch_loader, False),
                2: (itch_loader, False), 3: (daily_loader, True)}
    train_rows: dict[int, list[dict]] = {}
    for scale_id in (1, 2, 3):
        spec = SCALES[scale_id]; loader, filename, is_daily = train_spec[scale_id]
        id_map = daily_id_map if is_daily else None
        examples = assemble_examples(loader, "train", args.examples_per_scale, 128,
            spec["stride"], spec["horizon"], is_daily, args.seed + scale_id, id_map)
        train_rows[scale_id] = examples
        loaders[scale_id] = loader

    # Generate all scale targets with the teacher resident alone, then release it.
    teacher_args = argparse.Namespace(teacher=args.teacher, sequence_length=128,
        horizon_steps=60, teacher_batch_size=args.teacher_batch_size)
    teacher = base.load_teacher(teacher_args)
    teacher_params = sum(p.numel() for p in teacher._model.parameters())
    teacher_reports = {}
    for scale_id in (1, 2, 3):
        spec = SCALES[scale_id]
        teacher_reports[spec["name"]] = forecast_cache(teacher, train_rows[scale_id], spec,
            128, args.teacher_batch_size, args.round_trip_cost, device)
        filename = train_spec[scale_id][1]
        cache_path = args.runtime / filename
        source = args.daily if spec["source"] == "daily" else args.itch
        meta = {"scale": spec, "teacher": "Vincent05R/FinCast", "teacher_sha256": short_meta["teacher_sha256"],
                "source": str(source), "source_sha256": sha256(source),
                "split": "train only", "sequence_length": 128,
                "examples": len(train_rows[scale_id]),
                "valid_teacher_series": teacher_reports[spec["name"]]["series"],
                "teacher_report": teacher_reports[spec["name"]],
                "teacher_parameters": teacher_params, "test_accessed": False}
        save_scale_cache(cache_path, train_rows[scale_id], meta)
        cache_metas[scale_id] = {"cache": str(cache_path), **meta}
        datasets[scale_id] = train_rows[scale_id]
    del teacher
    torch.cuda.empty_cache()
    import gc; gc.collect()

    validation: dict[int, list[dict]] = {}
    for scale_id, (loader, is_daily) in val_spec.items():
        spec = SCALES[scale_id]
        validation[scale_id] = assemble_examples(loader, "validation", args.validation_examples,
            128, spec["stride"], spec["horizon"], is_daily,
            args.seed + 100 + scale_id, daily_id_map if is_daily else None)

    candidate_path = args.runtime / "candidate.pt"
    model, payload = load_market_candidate_checkpoint(candidate_path, "cpu")
    model.backbone.float(); model.context_policy.float(); model.context_value.float()
    model.activation_checkpointing = True
    model.to(device).train()
    before = evaluate(model, validation, loaders, device)
    payload.update({"temporal_scale_names": list(TIME_SCALE_NAMES),
        "temporal_scale_definitions": [dict(x) for x in SCALES],
        "temporal_scale_training": "FinCast forecast distillation + realized net-PnL rehearsal",
        "source_champion_sha256": champion_hash,
        "daily_source_sha256": sha256(args.daily),
        "itch_source_sha256": short_meta["source_sha256"],
        "test_split_accessed": False, "validation_before": before,
        "student_parameters": parameter_count(model.backbone),
        "teacher_parameters": teacher_params})
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    save_candidate(candidate_path, model, payload, optimizer, 0)
    report = {"event": "temporal_distillation_started", "runtime": str(args.runtime),
        "champion": str(args.champion), "champion_sha256": champion_hash,
        "candidate": str(candidate_path), "scales": [dict(x) for x in SCALES],
        "train_caches": cache_metas, "validation_before": before,
        "teacher_parameters": teacher_params, "student_parameters": parameter_count(model.backbone),
        "test_accessed": False, "device": torch.cuda.get_device_name(device)}
    save_jsonl(args.runtime / "distillation.jsonl", report)
    print(json.dumps(report, indent=2, allow_nan=False, default=str), flush=True)

    total_loss = 0.0; updates = 0; step_t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed + 999)
    order = np.resize(np.arange(4, dtype=np.int64), args.steps)
    rng.shuffle(order)
    torch.cuda.reset_peak_memory_stats(device)
    for step_ix, scale_id in enumerate(order, 1):
        choices = datasets[int(scale_id)]
        ex_ix = int(rng.integers(len(choices)))
        row = choices[ex_ix]
        batch = make_batch(loaders[int(scale_id)], row, int(scale_id), device)
        x, sid, mid, aid, mask, context, realized, valid = batch_tensors(batch, device)
        valid = valid[0] & mask[0, -1]
        tvalid = torch.as_tensor(row["teacher_valid"], device=device) & valid
        if not valid.any(): continue
        # BF16 has the same exponent range as FP32 and avoids the overflow
        # observed in long-horizon realized-return updates on 8 GB Ampere GPUs.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, values = model(x, sid, mid, aid, mask, context,
                                   torch.tensor([int(scale_id)], device=device))
        policy_probs = torch.softmax(logits[0].float(), dim=-1)
        _, pnl_matrix = utility(logits[0], realized[0], valid)
        expected_pnl = (policy_probs * pnl_matrix).sum(-1)
        loss = -expected_pnl[valid].mean() + .5 * F.smooth_l1_loss(values[0, valid].float(), expected_pnl[valid].detach())
        if tvalid.any():
            tp = torch.as_tensor(row["teacher_probs"], device=device)
            tv = torch.as_tensor(row["teacher_value"], device=device)
            temp = args.temperature
            kl = F.kl_div(F.log_softmax(logits[0, tvalid].float() / temp, dim=-1),
                          tp[tvalid], reduction="batchmean") * temp * temp
            vloss = F.smooth_l1_loss(values[0, tvalid].float(), tv[tvalid])
            loss = loss + args.policy_distill_weight * kl + args.value_distill_weight * vloss
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        total_loss += float(loss.detach().cpu()); updates += 1
        if step_ix % args.checkpoint_every == 0 or step_ix == args.steps:
            torch.cuda.synchronize(device)
            event = {"event": "temporal_distillation_progress", "step": step_ix,
                "updates": updates, "last_scale": TIME_SCALE_NAMES[int(scale_id)],
                "mean_loss": total_loss / max(updates, 1),
                "steps_per_second": step_ix / max(time.perf_counter() - step_t0, 1e-9),
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                "device": torch.cuda.get_device_name(device)}
            save_candidate(candidate_path, model, payload, optimizer, step_ix)
            save_jsonl(args.runtime / "distillation.jsonl", event)
            print(json.dumps(event), flush=True)

    after = evaluate(model, validation, loaders, device)
    torch.cuda.synchronize(device)
    final = {"event": "temporal_distillation_complete", "candidate": str(candidate_path),
        "source_champion_sha256": champion_hash, "teacher_sha256": short_meta["teacher_sha256"],
        "teacher_parameters": teacher_params, "student_parameters": parameter_count(model.backbone),
        "steps": args.steps, "optimizer_updates": updates,
        "mean_loss": total_loss / max(updates, 1),
        "elapsed_seconds": time.perf_counter() - step_t0,
        "seconds_per_update": (time.perf_counter() - step_t0) / max(updates, 1),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "validation_before_by_scale": before, "validation_after_by_scale": after,
        "time_scale_embedding_changed": not torch.equal(
            payload["state_dict"].get("backbone.time_scale_embedding.weight"),
            model.state_dict()["backbone.time_scale_embedding.weight"].detach().cpu().half()),
        "test_accessed": False, "promotion": "not performed; champion untouched"}
    payload["validation_after_by_scale"] = after
    payload["distillation_step"] = args.steps
    save_candidate(candidate_path, model, payload, optimizer, args.steps)
    (args.runtime / "latest-result.json").write_text(json.dumps(final, indent=2, allow_nan=False), encoding="utf-8")
    save_jsonl(args.runtime / "distillation.jsonl", final)
    print(json.dumps(final, indent=2, allow_nan=False), flush=True)
    itch_loader.close(); daily_loader.close()


if __name__ == "__main__":
    main()
