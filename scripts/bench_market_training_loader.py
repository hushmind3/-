"""Build the shared multi-market loader and run a bounded CUDA dry-run.

This command never edits or promotes the source champion. It writes all derived
panels and the one-step adapter candidate under runtime-global-market-training.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import psutil
import torch
from torch import nn

from stockrl.global_transformer import GlobalMarketTransformer, TransformerConfig, parameter_count, stable_id
from stockrl.market_training import (CONTEXT_FEATURES, CommonMarketTrainingLoader,
                                    ContextConditionedTransformer, ITCHSnapshotAdapter,
                                    load_market_candidate_checkpoint)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/external_sources/nasdaq/itch_snapshots_full_itch50_corrected.csv"
DEFAULT_CHAMPION = ROOT / "runtime-global-cuda-final/champion.pt"
DEFAULT_RUNTIME = ROOT / "runtime-global-market-training"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_rss_bytes() -> int:
    # psutil is the reliable Windows RSS source; ru_maxrss differs by platform.
    return int(psutil.Process().memory_info().rss)


class RSSHighWater:
    def __init__(self):
        self.peak = peak_rss_bytes()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._poll, name="rss-monitor", daemon=True)

    def _poll(self):
        process = psutil.Process()
        while not self.stop_event.wait(.05):
            self.peak = max(self.peak, int(process.memory_info().rss))

    def start(self): self.thread.start()
    def stop(self):
        self.stop_event.set(); self.thread.join()
        self.peak = max(self.peak, peak_rss_bytes())


def load_expanded_candidate(champion_path: Path, names: list[str], device: torch.device):
    payload = torch.load(champion_path, map_location="cpu", weights_only=False)
    old_cfg = TransformerConfig(**payload["config"])
    # Live champion checkpoints store the transformer beneath ``backbone.``
    # while older standalone checkpoints store its keys at the root. Normalize
    # both formats before expanding the symbol embedding.
    checkpoint_state = payload["state_dict"]
    if any(key.startswith("backbone.") for key in checkpoint_state):
        old_state = {key.removeprefix("backbone."): value
                     for key, value in checkpoint_state.items()
                     if key.startswith("backbone.")}
    else:
        old_state = checkpoint_state
    source_parameter_count = sum(t.numel() for t in old_state.values())
    cfg = replace(old_cfg, max_symbols=len(names))
    base = GlobalMarketTransformer(cfg)
    if "symbol_embedding.weight" not in old_state:
        raise KeyError("checkpoint transformer state has no symbol_embedding.weight")
    old_embedding = old_state["symbol_embedding.weight"]
    state = {k: v for k, v in old_state.items() if k != "symbol_embedding.weight"}
    incompatible = base.load_state_dict(state, strict=False)
    if incompatible.missing_keys != ["symbol_embedding.weight"] or incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint mismatch: {incompatible}")
    # The old panel hashed the bare ticker, so preserve that exact lookup when
    # widening the new deterministic market-qualified instrument map.
    legacy_ids = torch.tensor([stable_id(name.rsplit("|", 1)[-1], old_cfg.max_symbols)
                               for name in names], dtype=torch.long)
    with torch.no_grad():
        base.symbol_embedding.weight.copy_(old_embedding.index_select(0, legacy_ids))
    del payload, old_state, state, old_embedding
    base.half()
    initial_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    base.to(device)
    wrapped = ContextConditionedTransformer(base).to(device)
    wrapped.context_policy.float()
    wrapped.context_value.float()
    wrapped.freeze_backbone = False
    wrapped.activation_checkpointing = True
    for parameter in wrapped.backbone.parameters():
        parameter.requires_grad_(True)
    for parameter in wrapped.context_policy.parameters():
        parameter.requires_grad_(True)
    for parameter in wrapped.context_value.parameters():
        parameter.requires_grad_(True)
    return wrapped, cfg, initial_state, source_parameter_count


def batch_tensors(batch, device):
    return (batch.features.to(device=device, dtype=torch.float16),
            batch.symbol_ids.to(device), batch.market_ids.to(device),
            batch.asset_ids.to(device), batch.valid_mask.to(device),
            batch.market_context.to(device=device, dtype=torch.float32),
            batch.target_return.to(device=device, dtype=torch.float32),
            batch.target_valid.to(device))


def utility(logits, forward_return, valid, fee=0.001, slippage_bps=1.0):
    cost = 2.0 * (fee + slippage_bps / 10_000.0)
    actions = torch.stack((-forward_return - cost,
                           torch.zeros_like(forward_return),
                           forward_return - cost), dim=-1)
    probs = torch.softmax(logits.float(), dim=-1)
    return (probs * actions).sum(-1), actions


def estimate_compute(config, sequence_length: int, symbols: int) -> dict:
    d, heads, layers, mult = config.d_model, config.n_heads, config.n_layers, config.ff_mult
    temporal = (layers + 1) // 2
    cross = layers // 2
    tokens = sequence_length * symbols
    projection_ffn_macs = layers * tokens * (4 * d * d + 2 * d * d * mult)
    attention_macs = temporal * 2 * symbols * d * sequence_length**2
    attention_macs += cross * 2 * sequence_length * d * symbols**2
    return {"projection_and_ffn_macs": projection_ffn_macs,
            "attention_macs": attention_macs,
            "estimated_flops": 2 * (projection_ffn_macs + attention_macs)}


def checkpoint_delta_l1(model, initial_state):
    delta = 0.0
    changed = 0
    for name, parameter in model.backbone.named_parameters():
        current = parameter.detach().cpu()
        reference = initial_state[name]
        if not torch.equal(current, reference):
            changed += 1
        delta += float((current.float() - reference.float()).abs().sum())
    adapter_delta = float(model.context_policy.weight.detach().abs().sum() +
                          model.context_value.weight.detach().abs().sum())
    return {"backbone_weight_delta_l1": delta,
            "changed_backbone_parameter_tensors": changed,
            "context_adapter_weight_l1": adapter_delta}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--timeframe-seconds", type=int, default=1)
    parser.add_argument("--horizon-seconds", type=int, default=30)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if not args.source.exists():
        raise FileNotFoundError(f"corrected ITCH source not found: {args.source}")
    if not args.champion.exists():
        raise FileNotFoundError(f"source champion not found: {args.champion}")
    device = torch.device("cuda:0")
    args.runtime.mkdir(parents=True, exist_ok=True)
    cache = args.runtime / "shared-market-cache"
    rss_monitor = RSSHighWater(); rss_monitor.start()
    loader = CommonMarketTrainingLoader(
        cache, [args.source], [ITCHSnapshotAdapter(args.timeframe_seconds)],
        chunksize=args.chunksize, seed=2026, train_fraction=.70,
        validation_fraction=.15, context_stale_seconds=max(60, args.timeframe_seconds * 3))
    meta = loader.build(force=args.force_rebuild)
    if args.sequence_length != 128:
        print(f"benchmark_sequence_length={args.sequence_length} (requested); default supported=128", flush=True)
    print(json.dumps({"symbols": meta["symbols"], "time_rows": meta["times"],
                      "feature_dim": len(meta["feature_names"]),
                      "context_dim": len(meta["context_names"]),
                      "sequence_length": args.sequence_length,
                      "splits": loader.split_report()}, indent=2), flush=True)

    model, cfg, initial_state, source_parameter_count = load_expanded_candidate(
        args.champion, loader.symbol_names, device)
    expected = {str(n): {"features_shape": [1, args.sequence_length, n, len(meta["feature_names"])],
                         "features_bytes_fp16": args.sequence_length*n*len(meta["feature_names"])*2,
                         "estimated_compute": estimate_compute(cfg,args.sequence_length,n)}
                for n in (64,128,256)}
    print(json.dumps({"dryrun_plan_before_cuda_updates": expected,
                      "candidate_weight_bytes_fp16": parameter_count(model)*2,
                      "shared_panel_disk_bytes": meta["memory_mapped_bytes_estimate"],
                      "universe_symbols": meta["symbols"]}, indent=2), flush=True)
    process = psutil.Process()
    results = []
    # Sample each size at a train timestamp; all selected sets update exposure.
    for count in (64, 128, 256):
        model.backbone.load_state_dict(initial_state)
        nn.init.zeros_(model.context_policy.weight)
        nn.init.zeros_(model.context_value.weight)
        model.zero_grad(set_to_none=True)
        # SGD has no per-parameter moment buffers; the full 0.5B model can
        # update within an 8 GB card when block activations are checkpointed.
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        batch = loader.sample(1, count, args.sequence_length, "train",
                              horizon_seconds=args.horizon_seconds, device="cpu")
        x, sid, mid, aid, mask, context, returns, target_valid = batch_tensors(batch, device)
        validation_batch = None
        base_score = None
        if count == 128:
            validation_batch = loader.sample(1, 128, args.sequence_length, "validation",
                                              horizon_seconds=args.horizon_seconds, device="cpu")
            validation_tensors = batch_tensors(validation_batch, device)
            vx, vsid, vmid, vaid, vmask, vctx, vret, vvalid = validation_tensors
            model.eval()
            with torch.no_grad():
                base_logits, _ = model(vx, vsid, vmid, vaid, vmask, vctx)
                valid_val = vvalid[0] & vmask[0, -1]
                score, _ = utility(base_logits[0], vret[0], valid_val)
                base_score = float(score[valid_val].mean().cpu()) if valid_val.any() else None
            model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            model(x, sid, mid, aid, mask, context)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        logits, values = model(x, sid, mid, aid, mask, context)
        expected, action_returns = utility(logits[0], returns[0], target_valid[0])
        valid = target_valid[0] & mask[0, -1]
        if valid.any():
            loss = -expected[valid].mean() + 0.5 * nn.functional.smooth_l1_loss(
                values[0, valid].float(), expected[valid].detach())
        else:
            # Still execute real forward/backward when event data has no exact
            # start/end quote for this sparse sampled group.
            loss = (logits.float().square().mean() + values.float().square().mean())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        torch.cuda.synchronize(device)
        results.append({"symbols_per_sample": count,
                        "tensor_shape": list(x.shape),
                        "context_shape": list(context.shape),
                        "valid_reward_symbols": int(valid.sum().item()),
                        "step_seconds": elapsed,
                        "symbols_windows_per_second": count / elapsed,
                        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device),
                        "process_rss_bytes": peak_rss_bytes(),
                        "loss": float(loss.detach().cpu())})
        if count == 128:
            with torch.no_grad():
                candidate_logits, _ = model(vx, vsid, vmid, vaid, vmask, vctx)
                valid_val = vvalid[0] & vmask[0, -1]
                candidate_score, _ = utility(candidate_logits[0], vret[0], valid_val)
                candidate_score = float(candidate_score[valid_val].mean().cpu()) if valid_val.any() else None
            checkpoint = args.runtime / "candidate-dryrun.pt"
            delta_metrics = checkpoint_delta_l1(model, initial_state)
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({"state_dict": cpu_state, "config": asdict(cfg),
                        "context_features": list(CONTEXT_FEATURES),
                        "symbol_map": loader.symbol_map,
                        "source_champion": str(args.champion),
                        "source_champion_sha256": sha256(args.champion),
                        "dry_run_only": True, "promoted": False,
                        "optimizer_steps": 1}, checkpoint)
            del cpu_state
            val_metrics = {"champion_validation_expected_net_return": base_score,
                           "candidate_validation_expected_net_return": candidate_score,
                           "validation_observations": int(valid_val.sum().item()),
                           "validation_protocol": "single held-out minibatch dry-run; insufficient for promotion",
                           "validation_is_held_out": True,
                           "promotion_performed": False,
                           "weight_change": delta_metrics,
                           "candidate_checkpoint": str(checkpoint)}

        model.zero_grad(set_to_none=True)
        del optimizer
        torch.cuda.empty_cache()
    loader.save_coverage()
    coverage = loader.coverage_report()
    exposure_path = args.runtime / "symbol_exposure.csv"
    import pandas as pd
    pd.DataFrame({"instrument": loader.symbol_names,
                  "direct_symbol_id": np.arange(len(loader.symbol_names), dtype=np.int64),
                  "training_exposures": loader.coverage.exposure}).to_csv(exposure_path, index=False)
    symbols = len(loader.symbol_names)
    legacy_ids = [stable_id(name.rsplit("|", 1)[-1], 8192) for name in loader.symbol_names]
    collisions = len(legacy_ids) - len(set(legacy_ids))
    # Prove the isolated artifact can be restored and used for inference.
    del model, initial_state
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    reloaded, reload_payload = load_market_candidate_checkpoint(val_metrics["candidate_checkpoint"], device)
    torch.cuda.synchronize(device)
    reload_started = time.perf_counter()
    with torch.inference_mode():
        reload_logits, reload_values = reloaded(vx, vsid, vmid, vaid, vmask, vctx)
    torch.cuda.synchronize(device)
    reload_ms = (time.perf_counter() - reload_started) * 1000
    reload_result = {"loaded": True, "logits_shape": list(reload_logits.shape),
                     "values_shape": list(reload_values.shape),
                     "finite_outputs": bool(torch.isfinite(reload_logits).all() and torch.isfinite(reload_values).all()),
                     "inference_ms": reload_ms,
                     "parameter_count": parameter_count(reloaded.backbone),
                     "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    del reload_payload
    rss_monitor.stop()
    temporal_layers = (cfg.n_layers + 1) // 2
    cross_layers = cfg.n_layers // 2
    new_attention = {"sampled_symbols_per_sample": {
                         str(n): {"temporal_score_elements": temporal_layers * n * cfg.n_heads * args.sequence_length**2,
                                  "cross_asset_score_elements": cross_layers * args.sequence_length * cfg.n_heads * n * n}
                         for n in (64,128,256)}}
    result = {"source": str(args.source), "source_bytes": args.source.stat().st_size,
              "source_champion": str(args.champion), "source_champion_sha256": sha256(args.champion),
              "device": torch.cuda.get_device_name(device),
              "source_champion_parameter_count": source_parameter_count,
              "expanded_backbone_parameter_count": parameter_count(reloaded.backbone),
              "candidate_parameter_count": parameter_count(reloaded),
              "total_symbols": symbols, "symbols_used_by_loader": len(loader.symbol_map),
              "new_deterministic_symbol_ids_unique": len(set(loader.symbol_map.values())) == symbols,
              "legacy_8192_hash_collisions_within_this_universe": collisions,
              "time_rows": len(loader.time_keys), "feature_names": list(meta["feature_names"]),
              "sequence_length": args.sequence_length,
              "reward_horizon_seconds": args.horizon_seconds,
              "market_context_features": list(CONTEXT_FEATURES),
              "time_splits": loader.split_report(), "benchmark": results,
              "coverage": {k:v for k,v in coverage.items() if k != "exposure_by_symbol"},
              "coverage_exposure_file": str(exposure_path),
              "attention_score_elements_note": "full market is summarized once into shared context; symbols attend only within each sampled group",
              "sampled_attention_estimate": new_attention,
              "candidate_validation": val_metrics,
              "checkpoint_reload_inference": reload_result,
              "candidate_promoted": False,
              "peak_cuda_allocated_bytes": max([x["peak_vram_allocated_bytes"] for x in results] +
                                               [reload_result["peak_vram_allocated_bytes"]]),
              "peak_process_rss_bytes": rss_monitor.peak,
              "memory_mapped_bytes_estimate": meta["memory_mapped_bytes_estimate"]}
    report_path = args.runtime / "dryrun_metrics.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    loader.close()


if __name__ == "__main__":
    main()
