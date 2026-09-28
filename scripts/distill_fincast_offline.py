"""Offline FinCast forecast distillation into the existing global market agent.

FinCast runs first and writes an immutable teacher-output cache. It is unloaded
before the student is placed on CUDA, so the ~1B teacher and ~0.5B student are
never resident on the GPU together. Only training-split windows are used to
create teacher targets or update the student; validation is reserved for the
before/after net-PnL comparison and test is never opened.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "models/teachers/FinCast-fts/src"))

from stockrl.global_transformer import TransformerConfig, parameter_count  # noqa: E402
from stockrl.market_training import (  # noqa: E402
    CONTEXT_FEATURES, CommonMarketTrainingLoader, ContextConditionedTransformer,
    ITCHSnapshotAdapter, MarketBatch, load_market_candidate_checkpoint,
)
from bench_market_training_loader import (  # noqa: E402
    DEFAULT_CHAMPION, DEFAULT_SOURCE, batch_tensors, utility,
)


DEFAULT_TEACHER = ROOT / "models/teachers/FinCast-1B/v1.pth"
DEFAULT_SOURCE_CODE = ROOT / "models/teachers/FinCast-fts"
DEFAULT_RUNTIME = ROOT / "runtime-global-market-training/fincast-distillation"
DEFAULT_LOADER_CACHE = ROOT / "runtime-global-market-training/shared-market-cache"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_json(path: Path, obj: dict, append: bool = False) -> None:
    with path.open("a" if append else "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, allow_nan=False, default=str) + "\n")


def eligible_times(loader, split: str, sequence_length: int, horizon_seconds: int) -> np.ndarray:
    lo, hi = map(int, loader.splits[split])
    possible = np.arange(lo + sequence_length - 1, hi, dtype=np.int64)
    target = np.searchsorted(
        loader.time_keys,
        np.asarray(loader.time_keys[possible], dtype=np.int64) + horizon_seconds,
        side="left",
    )
    return possible[target < hi]


def open_loader(args):
    loader = CommonMarketTrainingLoader(
        args.loader_cache, [args.source], [ITCHSnapshotAdapter(1)],
        chunksize=250_000, seed=2026, train_fraction=.70,
        validation_fraction=.15, context_stale_seconds=60,
    )
    # Existing canonical panel is reused; this distillation must not reparse
    # the 6.8 GB source or alter its data/splits.
    manifest = loader.build(force=False)
    return loader, manifest


def load_teacher(args):
    try:
        from ffm import FFM, FFmHparams
    except Exception as exc:
        raise RuntimeError(
            "FinCast imports failed. Install the isolated inference dependencies: "
            "beartype einx einops CoLT5-attention utilsforecast"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("FinCast distillation was requested on CUDA, but CUDA is unavailable")
    # Official FinCast configuration: 50 decoder layers, d=1280, 16 heads,
    # 4 experts with top-2 routing, 32-point patches, and 9 quantiles.
    hparams = FFmHparams(
        context_len=args.sequence_length,
        horizon_len=args.horizon_steps,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=50,
        num_heads=16,
        model_dims=1280,
        per_core_batch_size=args.teacher_batch_size,
        backend="gpu",
        quantiles=tuple(i / 10 for i in range(1, 10)),
        use_positional_embedding=False,
        num_experts=4,
        gating_top_n=2,
        point_forecast_mode="median",
    )
    teacher = FFM(hparams=hparams, checkpoint=str(args.teacher), loading_mode=0)
    teacher.model_eval_mode()
    return teacher


def quantile_action_targets(low: np.ndarray, median: np.ndarray, high: np.ndarray,
                            cost: float) -> tuple[np.ndarray, np.ndarray]:
    """Translate FinCast 10/50/90% price forecasts to return probabilities.

    The mapping is a piecewise-linear quantile CDF, not a hand-authored policy.
    SELL/HOLD/BUY correspond to short/flat/long outcomes after round-trip costs.
    """
    def cdf(x):
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        left = x <= low
        midlo = (x > low) & (x <= median)
        midhi = (x > median) & (x <= high)
        right = x > high
        # Beyond q10/q90 the model does not identify tail shape; use the
        # represented CDF mass at that boundary instead of inventing a tail.
        out[left] = .1
        # Between the represented quantiles, interpolate their CDF masses.
        out[midlo] = .1 + .4 * (x[midlo] - low[midlo]) / np.maximum(median[midlo] - low[midlo], 1e-8)
        out[midhi] = .5 + .4 * (x[midhi] - median[midhi]) / np.maximum(high[midhi] - median[midhi], 1e-8)
        out[right] = .9
        return np.clip(out, 0.0, 1.0)

    # Quantile tails outside q10/q90 are deliberately not extrapolated as a
    # confident forecast: place remaining mass on HOLD unless inside q range.
    p_sell = cdf(np.full_like(median, -cost))
    p_buy = 1.0 - cdf(np.full_like(median, cost))
    probs = np.stack((p_sell, np.maximum(0.0, 1.0 - p_sell - p_buy), p_buy), axis=-1)
    probs = np.maximum(probs, 1e-6)
    probs /= probs.sum(axis=-1, keepdims=True)
    value = p_sell * (-median - cost) + p_buy * (median - cost)
    return probs.astype(np.float32), value.astype(np.float32)


def create_teacher_cache(args, loader, manifest) -> dict:
    args.teacher_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = args.teacher_cache_dir / "teacher_outputs_train.npz"
    cache_meta_file = args.teacher_cache_dir / "teacher_outputs_train.json"
    # Reuse previously verified hashes for these immutable artifacts instead
    # of rereading 10+ GB on every retry. Fall back to a fresh digest when the
    # known metadata does not identify the exact files.
    teacher_hash = None
    checksum_file = args.teacher.parent / "checksum_f.txt"
    if checksum_file.exists() and args.teacher.stat().st_size == 3_966_703_063:
        teacher_hash = checksum_file.read_text(encoding="utf-8").split()[0]
    if teacher_hash is None:
        teacher_hash = sha256(args.teacher)
    source_hash = None
    old_run = ROOT / "runtime-global-market-training/one-epoch-training-final.json"
    if old_run.exists():
        old = json.loads(old_run.read_text(encoding="utf-8"))
        if (Path(old.get("source", "")).resolve() == args.source.resolve() and
                args.source.stat().st_size == int(manifest["sources"][0]["bytes"])):
            source_hash = old.get("source_sha256")
    if source_hash is None:
        source_hash = sha256(args.source)
    if cache_file.exists() and cache_meta_file.exists() and not args.rebuild_teacher_cache:
        meta = json.loads(cache_meta_file.read_text(encoding="utf-8"))
        if (meta.get("teacher_sha256") == teacher_hash and
                meta.get("source_sha256") == source_hash and
                meta.get("examples") == args.cache_examples and
                meta.get("horizon_steps") == args.horizon_steps):
            arrays = dict(np.load(cache_file, allow_pickle=False))
            return {**arrays, "meta": meta, "cache_path": str(cache_file)}
        raise RuntimeError("Existing teacher cache inputs differ; choose a new runtime or explicitly rebuild")
    if cache_file.exists() or cache_meta_file.exists():
        raise FileExistsError("Refusing to overwrite an existing teacher output cache")

    args.runtime.mkdir(parents=True, exist_ok=True)
    train_times = eligible_times(loader, "train", args.sequence_length, args.horizon_seconds)
    rng = np.random.default_rng(args.seed)
    ends = np.sort(rng.choice(train_times, size=min(args.cache_examples, len(train_times)), replace=False))
    n = len(ends); symbols = args.symbols_per_sample
    ids_all = np.zeros((n, symbols), np.int64)
    returns_all = np.zeros((n, symbols), np.float32)
    valid_all = np.zeros((n, symbols), np.bool_)
    probs_all = np.full((n, symbols, 3), 1 / 3, np.float32)
    values_all = np.zeros((n, symbols), np.float32)
    low_all = np.zeros((n, symbols), np.float32)
    median_all = np.zeros((n, symbols), np.float32)
    high_all = np.zeros((n, symbols), np.float32)

    original_exposure = loader.coverage.exposure.copy()
    exposure_rng = loader.coverage.rng
    coverage_file = args.loader_cache / "coverage_exposure.npz"
    previous_coverage_bytes = coverage_file.read_bytes() if coverage_file.exists() else None
    loader.coverage.rng = np.random.default_rng(args.seed + 1)
    examples = []
    for i, end in enumerate(ends):
        batch = loader.sample(1, symbols, args.sequence_length, "train",
                              horizon_seconds=args.horizon_seconds,
                              device="cpu", fixed_end_indices=[int(end)])
        ids = batch.sampled_ids[0]
        ids_all[i] = ids
        returns_all[i] = batch.target_return[0].numpy()
        valid = batch.target_valid[0].numpy()
        valid_all[i] = valid
        # FinCast is univariate: provide close-price context only for symbols
        # with a genuine, cost-evaluable target at this training timestamp.
        tix = np.arange(int(end) - args.sequence_length + 1, int(end) + 1, dtype=np.int64)
        for j in np.flatnonzero(valid):
            closes = np.asarray(loader.closes[tix, ids[j]], dtype=np.float32)
            finite = np.isfinite(closes) & (closes > 0)
            if int(finite.sum()) < 32:
                valid_all[i, j] = False
                continue
            first = int(np.flatnonzero(finite)[0])
            series = closes[first:]
            if len(series) > args.sequence_length:
                series = series[-args.sequence_length:]
            examples.append((i, int(j), series, float(closes[-1])))
    loader.coverage.exposure[:] = original_exposure
    loader.coverage.rng = exposure_rng
    if previous_coverage_bytes is not None:
        coverage_file.write_bytes(previous_coverage_bytes)

    if not examples:
        raise RuntimeError("No valid train-split price windows for FinCast")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested 3070 teacher run")
    teacher_start = time.perf_counter()
    print(f"fincast_cache_start examples={n} valid_series={len(examples)} ", flush=True)
    teacher = load_teacher(args)
    torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
    # API internally batches at per_core_batch_size; pass all valid series but
    # bounded by chunks so CPU/GPU buffers stay small and predictable.
    for offset in range(0, len(examples), args.teacher_batch_size):
        group = examples[offset:offset + args.teacher_batch_size]
        inputs = [x[2] for x in group]
        _, quantile_forecast = teacher.forecast(
            inputs, freq=[0] * len(inputs), forecast_context_len=args.sequence_length,
            normalize=False,
        )
        h = min(args.horizon_steps, quantile_forecast.shape[1]) - 1
        q = quantile_forecast[:, h, :]
        # API layout: point forecast followed by q=.1,.2,...,.9.
        q10 = q[:, 1].astype(np.float64)
        q50 = q[:, 5].astype(np.float64)
        q90 = q[:, 9].astype(np.float64)
        last = np.asarray([x[3] for x in group], np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            lo_ret, med_ret, hi_ret = q10 / last - 1.0, q50 / last - 1.0, q90 / last - 1.0
        good = np.isfinite(lo_ret) & np.isfinite(med_ret) & np.isfinite(hi_ret)
        # Clamp outlandish zero-shot extrapolations to a broad but finite daily
        # return range; predictions remain uncertainty-aware and cached raw in
        # return units used by the student.
        lo_ret = np.clip(lo_ret, -.25, .25)
        med_ret = np.clip(med_ret, -.25, .25)
        hi_ret = np.clip(hi_ret, -.25, .25)
        probabilities, values = quantile_action_targets(lo_ret, med_ret, hi_ret,
                                                         args.round_trip_cost)
        for k, (i, j, _, _) in enumerate(group):
            if not good[k]:
                valid_all[i, j] = False
                continue
            low_all[i, j], median_all[i, j], high_all[i, j] = lo_ret[k], med_ret[k], hi_ret[k]
            probs_all[i, j], values_all[i, j] = probabilities[k], values[k]
        if (offset // args.teacher_batch_size) % 25 == 0:
            torch.cuda.synchronize()
            print(f"fincast_forecasted={min(offset + len(group),len(examples))}/{len(examples)} "
                  f"elapsed={time.perf_counter()-teacher_start:.1f}s", flush=True)
    torch.cuda.synchronize()
    teacher_seconds = time.perf_counter() - teacher_start
    peak_teacher = int(torch.cuda.max_memory_allocated(0))
    teacher_params = sum(p.numel() for p in teacher._model.parameters())
    del teacher
    torch.cuda.empty_cache()
    gc = __import__("gc"); gc.collect()

    tmp = cache_file.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez(f, end_indices=ends, symbol_ids=ids_all, target_return=returns_all,
                 valid=valid_all, teacher_probs=probs_all, teacher_value=values_all,
                 teacher_q10=low_all, teacher_q50=median_all, teacher_q90=high_all)
    tmp.replace(cache_file)
    meta = {
        "status": "complete", "teacher": "Vincent05R/FinCast", "teacher_source":
        "https://huggingface.co/Vincent05R/FinCast", "teacher_sha256": teacher_hash,
        "source": str(args.source), "source_sha256": source_hash,
        "loader_manifest": manifest, "split": "train only", "test_opened": False,
        "examples": n, "valid_teacher_series": len(examples),
        "valid_distillation_symbols": int(valid_all.sum()),
        "symbols_per_sample": symbols, "sequence_length": args.sequence_length,
        "horizon_steps": args.horizon_steps, "frequency_id": 0,
        "teacher_architecture": {"layers": 50, "hidden_size": 1280, "heads": 16,
                                 "experts": 4, "top_k": 2, "input_patch": 32,
                                 "output_patch": 128, "quantiles": [i / 10 for i in range(1, 10)]},
        "mapping": "close history -> FinCast 10/50/90% future price -> 30-step returns -> piecewise quantile CDF for SELL/HOLD/BUY and expected net opportunity value",
        "round_trip_cost": args.round_trip_cost, "teacher_parameters": teacher_params,
        "teacher_seconds": teacher_seconds, "teacher_peak_vram_bytes": peak_teacher,
        "cache_file": str(cache_file), "cache_bytes": cache_file.stat().st_size,
        "test_accessed": False,
    }
    cache_meta_file.write_text(json.dumps(meta, indent=2, allow_nan=False, default=str), encoding="utf-8")
    report_json(args.runtime / "distillation.jsonl", {"event": "teacher_cache_complete", **meta}, append=True)
    return {"end_indices": ends, "symbol_ids": ids_all, "target_return": returns_all,
            "valid": valid_all, "teacher_probs": probs_all, "teacher_value": values_all,
            "teacher_q10": low_all, "teacher_q50": median_all, "teacher_q90": high_all,
            "meta": meta, "cache_path": str(cache_file)}


def make_batch(loader, cache: dict, idx: int, device: torch.device) -> MarketBatch:
    end = int(cache["end_indices"][idx])
    ids = cache["symbol_ids"][idx]
    tix = np.arange(end - 127, end + 1, dtype=np.int64)
    return MarketBatch(
        features=torch.as_tensor(np.asarray(loader.features[tix[:, None], ids[None, :]], np.float16)[None], device=device),
        symbol_ids=torch.as_tensor(ids[None], dtype=torch.long, device=device),
        market_ids=torch.as_tensor(np.asarray(loader.market_ids[ids], np.int64)[None], device=device),
        asset_ids=torch.as_tensor(np.asarray(loader.asset_ids[ids], np.int64)[None], device=device),
        valid_mask=torch.as_tensor(np.asarray(loader.observed[tix[:, None], ids[None, :]], bool)[None], device=device),
        market_context=torch.as_tensor(np.asarray(loader.context[tix], np.float32)[None], device=device),
        target_return=torch.as_tensor(cache["target_return"][idx:idx+1], device=device),
        target_valid=torch.as_tensor(cache["valid"][idx:idx+1], device=device),
        end_indices=np.asarray([end], np.int64), target_indices=np.zeros((1,), np.int64),
        sampled_ids=[ids],
    )


def evaluate(model, batches, device):
    model.eval()
    total, count = 0.0, 0
    actions = {"SELL": 0, "HOLD": 0, "BUY": 0}
    with torch.inference_mode():
        for batch in batches:
            x, sid, mid, aid, mask, context, ret, valid = batch_tensors(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits, values = model(x, sid, mid, aid, mask, context)
            valid = valid[0] & mask[0, -1]
            if not valid.any():
                continue
            _, pnl = utility(logits[0], ret[0], valid)
            chosen = logits[0].float().argmax(-1)
            total += float(pnl.gather(1, chosen[:, None]).squeeze(1)[valid].sum().cpu())
            count += int(valid.sum())
            for i, label in enumerate(("SELL", "HOLD", "BUY")):
                actions[label] += int(((chosen == i) & valid).sum())
    model.train()
    return {"net_return_sum": total, "valid_decisions": count,
            "mean_net_return": total / max(count, 1), "actions": actions}


def save_student(path: Path, model, payload: dict, optimizer, step: int) -> None:
    out = dict(payload)
    state = {}
    for key, value in model.state_dict().items():
        value = value.detach().cpu()
        # Keep the deployed checkpoint in the champion's FP16 backbone format;
        # updates are accumulated in FP32 master parameters during training.
        if key.startswith("backbone.") and value.is_floating_point():
            value = value.half()
        state[key] = value
    out.update({"state_dict": state,
                "optimizer_state": optimizer.state_dict(), "distillation_step": int(step)})
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(out, tmp)
    tmp.replace(path)


def student_phase(args, loader, manifest, cache):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for student distillation")
    device = torch.device("cuda:0")
    args.runtime.mkdir(parents=True, exist_ok=True)
    candidate_path = args.runtime / "candidate.pt"
    if not candidate_path.exists():
        # Preserve the source champion byte-for-byte and start a new isolated
        # candidate from it, without touching any prior candidate.
        shutil.copy2(args.champion, candidate_path)
    champion_sha = sha256(args.champion)
    training_sha = cache["meta"]["source_sha256"]
    teacher_sha = cache["meta"]["teacher_sha256"]
    payload_meta = {
        "config": None, "context_features": list(CONTEXT_FEATURES),
        "source_champion": str(args.champion),
        "source_champion_sha256": champion_sha, "teacher_source": "Vincent05R/FinCast",
        "teacher_checkpoint": str(args.teacher), "teacher_sha256": teacher_sha,
        "teacher_cache": cache["cache_path"], "teacher_cache_sha256": sha256(Path(cache["cache_path"])),
        "training_source": str(args.source), "training_source_sha256": training_sha,
        "training_split_only": True, "validation_split_only_for_gate": True,
        "test_split_accessed": False, "distillation_objective": "quantile-derived policy KL + forecast value + realized net-PnL policy/value rehearsal",
    }
    model, loaded = load_market_candidate_checkpoint(candidate_path, "cpu")
    cfg = TransformerConfig(**loaded["config"])
    # Preserve a larger cross-market symbol map from the student checkpoint.
    # The ITCH loader may cover only a subset (for example, the student may
    # also contain daily-market instruments). Never shrink or renumber it.
    candidate_symbol_map = loaded.get("symbol_map")
    if not isinstance(candidate_symbol_map, dict):
        raise RuntimeError("student candidate has no deterministic symbol_map")
    if len(candidate_symbol_map) != cfg.max_symbols:
        raise RuntimeError("student symbol_map size does not match its embedding capacity")
    mismatched = [name for name, idx in loader.symbol_map.items()
                 if candidate_symbol_map.get(name) != idx]
    if mismatched:
        raise RuntimeError(
            f"ITCH symbol IDs do not match the student checkpoint map ({len(mismatched)} mismatches); "
            "refusing to distill with shifted embeddings"
        )
    payload_meta["symbol_map"] = candidate_symbol_map
    payload_meta["config"] = asdict(cfg)
    payload_meta["student_parameters"] = parameter_count(model.backbone)
    payload_meta["training_master_dtype"] = "float32 with CUDA FP16 autocast"
    payload_meta["checkpoint_backbone_dtype"] = "float16"
    # Validation batches are fixed once and reused byte-for-byte before/after.
    val_times = eligible_times(loader, "validation", args.sequence_length, args.horizon_seconds)
    select = val_times[np.linspace(0, len(val_times)-1,
                                  min(args.validation_batches, len(val_times)), dtype=np.int64)]
    exp0 = loader.coverage.exposure.copy()
    rng0 = loader.coverage.rng
    loader.coverage.rng = np.random.default_rng(args.seed + 2)
    val_batches = [loader.sample(1, args.symbols_per_sample, args.sequence_length,
        "validation", horizon_seconds=args.horizon_seconds, device="cpu",
        fixed_end_indices=[int(t)]) for t in select]
    loader.coverage.exposure[:] = exp0
    loader.coverage.rng = rng0
    model.activation_checkpointing = True
    model.backbone.train()
    model.train()
    # Half-precision parameters cannot represent the very small policy-gradient
    # updates. Use FP32 master parameters and AMP for compute, then store the
    # deployable checkpoint back in the champion's FP16 format.
    model.backbone.float()
    model.to(device)
    before_current = evaluate(model, val_batches, device)
    before = loaded.get("validation_before", before_current)
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),
                                lr=args.learning_rate)
    start_step = 0
    if args.resume and candidate_path.exists():
        ck = torch.load(candidate_path, map_location="cpu", weights_only=False)
        if ck.get("source_champion_sha256") != champion_sha or ck.get("teacher_sha256") != teacher_sha:
            raise RuntimeError("resume candidate source hash mismatch")
        start_step = int(ck.get("distillation_step", 0))
        if ck.get("optimizer_state"):
            optimizer.load_state_dict(ck["optimizer_state"])
        model.load_state_dict(ck["state_dict"], strict=True)
        model.to(device)
    if start_step == 0:
        payload_meta["validation_before"] = before
        save_student(candidate_path, model, payload_meta, optimizer, 0)

    n_data = len(cache["end_indices"])
    if n_data < 1:
        raise ValueError("teacher cache has no training examples")
    cache_valid = cache["valid"].astype(bool)
    cache_probs = cache["teacher_probs"].astype(np.float32)
    cache_values = cache["teacher_value"].astype(np.float32)
    device_val = device
    torch.cuda.reset_peak_memory_stats(device)
    step_start = time.perf_counter()
    loss_sum = 0.0
    n_updates = 0
    rng = np.random.default_rng(args.seed + start_step)
    order = rng.integers(0, n_data, size=max(args.steps, 1), dtype=np.int64)
    for local_step, sample_ix in enumerate(order, start=1):
        global_step = start_step + local_step
        batch = make_batch(loader, cache, int(sample_ix), device_val)
        x, sid, mid, aid, mask, context, realized, valid = batch_tensors(batch, device)
        valid = valid[0] & mask[0, -1]
        teacher_valid = torch.as_tensor(cache_valid[sample_ix], device=device) & valid
        if not valid.any():
            continue
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits, values = model(x, sid, mid, aid, mask, context)
        probs = torch.softmax(logits[0].float(), dim=-1)
        expected, _ = utility(logits[0], realized[0], valid)
        policy_rl = -expected[valid].mean()
        value_rl = F.smooth_l1_loss(values[0, valid].float(), expected[valid].detach())
        loss = policy_rl + .5 * value_rl
        if teacher_valid.any():
            target_probs = torch.as_tensor(cache_probs[sample_ix], device=device)
            target_value = torch.as_tensor(cache_values[sample_ix], device=device)
            t = args.distill_temperature
            kl = F.kl_div(F.log_softmax(logits[0, teacher_valid].float() / t, dim=-1),
                          target_probs[teacher_valid], reduction="batchmean") * (t*t)
            teacher_value_loss = F.smooth_l1_loss(values[0, teacher_valid].float(),
                                                    target_value[teacher_valid])
            loss = loss + args.policy_distill_weight * kl + args.value_distill_weight * teacher_value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        optimizer.step()
        loss_sum += float(loss.detach().cpu())
        n_updates += 1
        if global_step % args.checkpoint_every == 0 or local_step == args.steps:
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - step_start
            log = {"event": "student_progress", "step": global_step,
                   "requested_steps": args.steps, "updates": n_updates,
                   "mean_loss": loss_sum/max(n_updates,1),
                   "steps_per_second": local_step/max(elapsed,1e-9),
                   "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                   "device": torch.cuda.get_device_name(device)}
            report_json(args.runtime / "distillation.jsonl", log, append=True)
            print(json.dumps(log), flush=True)
            payload_meta["validation_before"] = before
            save_student(candidate_path, model, payload_meta, optimizer, global_step)
    torch.cuda.synchronize(device)
    after = evaluate(model, val_batches, device)
    elapsed = time.perf_counter() - step_start
    result = {
        "event": "student_run_complete", "candidate": str(candidate_path),
        "source_champion": str(args.champion), "source_champion_sha256": champion_sha,
        "training_source_sha256": training_sha,
        "teacher_sha256": teacher_sha, "teacher_cache": cache["cache_path"],
        "device": torch.cuda.get_device_name(device), "student_parameters": parameter_count(model.backbone),
        "start_step": start_step, "end_step": start_step + args.steps,
        "optimizer_updates": n_updates, "mean_train_loss": loss_sum/max(n_updates,1),
        "elapsed_seconds": elapsed, "update_seconds_per_step": elapsed/max(n_updates,1),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "validation_before": before, "validation_after": after,
        "validation_net_return_improved": after["valid_decisions"] > 0 and
            before["valid_decisions"] > 0 and after["mean_net_return"] > before["mean_net_return"],
        "promotion": "not performed", "test_accessed": False,
    }
    payload_meta["validation_before"] = before
    payload_meta["validation_after_last_run"] = after
    save_student(candidate_path, model, payload_meta, optimizer, start_step + args.steps)
    report_json(args.runtime / "distillation.jsonl", result, append=True)
    (args.runtime / "latest-result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    del model
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION)
    p.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    p.add_argument("--loader-cache", type=Path, default=DEFAULT_LOADER_CACHE)
    p.add_argument("--teacher-cache-dir", type=Path, default=None)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--symbols-per-sample", type=int, default=128)
    p.add_argument("--horizon-steps", type=int, default=30)
    p.add_argument("--horizon-seconds", type=int, default=30)
    p.add_argument("--cache-examples", type=int, default=1000)
    p.add_argument("--teacher-batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--validation-batches", type=int, default=16)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-7)
    p.add_argument("--distill-temperature", type=float, default=2.0)
    p.add_argument("--policy-distill-weight", type=float, default=.15)
    p.add_argument("--value-distill-weight", type=float, default=.05)
    p.add_argument("--round-trip-cost", type=float, default=.0022)
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rebuild-teacher-cache", action="store_true")
    args = p.parse_args()
    if args.teacher_cache_dir is None:
        args.teacher_cache_dir = args.runtime
    if not args.teacher.exists() or not args.source.exists() or not args.champion.exists():
        raise FileNotFoundError("FinCast teacher, canonical source, or champion missing")
    if args.steps < 1 or args.cache_examples < 1:
        raise ValueError("steps and cache-examples must be positive")
    if args.sequence_length != 128 or args.symbols_per_sample != 128:
        raise ValueError("This FinCast transfer uses the current checkpoint's 128x128 input contract")
    args.runtime.mkdir(parents=True, exist_ok=True)
    loader, manifest = open_loader(args)
    try:
        cache = create_teacher_cache(args, loader, manifest)
        # An unloaded teacher is required before the 0.5B candidate reaches CUDA.
        result = student_phase(args, loader, manifest, cache)
        return result
    finally:
        loader.close()


if __name__ == "__main__":
    main()
