"""Train one unique-data epoch on the corrected ITCH snapshot candidate.

Only the canonical corrected snapshot CSV is read. Raw ITCH and older parsed
CSV versions stay on disk but are not additional training inputs. Champion is
read-only; the resumable candidate is written to the isolated runtime.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from bench_market_training_loader import (DEFAULT_CHAMPION, DEFAULT_RUNTIME, DEFAULT_SOURCE,
    batch_tensors, load_expanded_candidate, sha256, utility)
from stockrl.market_training import (CONTEXT_FEATURES, CommonMarketTrainingLoader,
    ContextConditionedTransformer, ITCHSnapshotAdapter)
from stockrl.global_transformer import parameter_count

ROOT = Path(__file__).resolve().parents[1]


def save_candidate(path: Path, model, cfg, loader, champion_path, champion_sha, source_path, source_sha, step, epoch_order,
                   next_offset, optimizer_steps, initial_val):
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": asdict(cfg), "context_features": list(CONTEXT_FEATURES),
        "symbol_map": loader.symbol_map, "source_champion": str(champion_path),
        "source_champion_sha256": champion_sha,
        "canonical_training_source": str(source_path),
        "canonical_training_source_sha256": source_sha,
        "deduplicated_inputs": [str(source_path)], "epochs_total": 1,
        "epoch_order": epoch_order, "next_offset": int(next_offset),
        "optimizer_steps": int(optimizer_steps), "training_split_only": True,
        "validation_metrics_before_training": initial_val,
        "test_split_opened": False,
        "coverage_exposure": loader.coverage.exposure.copy(),
        "coverage_rng_state": deepcopy(loader.coverage.rng.bit_generator.state),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def make_time_indices(loader, split, sequence_length, horizon_seconds):
    lo, hi = map(int, loader.splits[split])
    possible = np.arange(lo + sequence_length - 1, hi, dtype=np.int64)
    target = np.searchsorted(loader.time_keys,
                             np.asarray(loader.time_keys[possible], dtype=np.int64) + horizon_seconds,
                             side="left")
    return possible[target < hi]


def evaluate(model, batches, device):
    model.eval()
    total_pnl = 0.0
    valid_n = 0
    decisions = {"SELL": 0, "HOLD": 0, "BUY": 0}
    with torch.inference_mode():
        for batch in batches:
            x, sid, mid, aid, mask, context, returns, target_valid = batch_tensors(batch, device)
            logits, _ = model(x, sid, mid, aid, mask, context)
            valid = target_valid[0] & mask[0, -1]
            if not valid.any():
                continue
            _, action_pnl = utility(logits[0], returns[0], valid)
            action = logits[0].float().argmax(-1)
            selected = action_pnl.gather(1, action[:, None]).squeeze(1)
            total_pnl += float(selected[valid].sum().cpu())
            valid_n += int(valid.sum())
            for ix, name in enumerate(("SELL", "HOLD", "BUY")):
                decisions[name] += int(((action == ix) & valid).sum())
    model.train()
    return {"net_return_sum": total_pnl, "valid_decisions": valid_n,
            "mean_net_return": total_pnl / max(valid_n, 1), "actions": decisions}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION)
    p.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    p.add_argument("--sequence-length", type=int, default=128)
    p.add_argument("--symbols-per-sample", type=int, default=128)
    p.add_argument("--horizon-seconds", type=int, default=30)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--validation-every", type=int, default=1000)
    p.add_argument("--validation-batches", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this RTX 3070 training run")
    if not a.source.exists() or not a.champion.exists():
        raise FileNotFoundError("canonical ITCH snapshot or source champion is missing")
    if a.sequence_length != 128 or a.symbols_per_sample != 128:
        raise ValueError("the approved baseline run uses sequence_length=128 and symbols_per_sample=128")
    a.runtime.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    cache = a.runtime / "shared-market-cache"
    loader = CommonMarketTrainingLoader(cache, [a.source], [ITCHSnapshotAdapter(1)],
        chunksize=250_000, seed=2026, train_fraction=.70, validation_fraction=.15,
        context_stale_seconds=60)
    meta = loader.build(force=False)
    train_times = make_time_indices(loader, "train", a.sequence_length, a.horizon_seconds)
    validation_times = make_time_indices(loader, "validation", a.sequence_length, a.horizon_seconds)
    order = np.random.default_rng(20260927).permutation(train_times)

    candidate_path = a.runtime / "candidate-one-epoch-latest.pt"
    if not (a.resume and candidate_path.exists()):
        # Prior benchmark and diagnostic samples are not training exposures.
        loader.coverage.exposure[:] = 0
        loader.coverage.rng = np.random.default_rng(2026)

    # Fixed validation windows and fixed, reproducible symbol groups. Validation
    # sampling must not distort training coverage or sampler randomness.
    exp_before = loader.coverage.exposure.copy()
    rng_before = deepcopy(loader.coverage.rng.bit_generator.state)
    selected_val = validation_times[np.linspace(0, len(validation_times)-1,
                           min(a.validation_batches, len(validation_times)), dtype=np.int64)]
    val_batches = [loader.sample(1, a.symbols_per_sample, a.sequence_length, "validation",
        horizon_seconds=a.horizon_seconds, device="cpu", fixed_end_indices=[int(t)])
        for t in selected_val]
    loader.coverage.exposure[:] = exp_before
    loader.coverage.rng.bit_generator.state = rng_before

    model, cfg, initial_state, source_params = load_expanded_candidate(a.champion,
                                                        loader.symbol_names, device)
    source_sha = sha256(a.champion)
    training_sha = sha256(a.source)
    log_path = a.runtime / "one-epoch-training.jsonl"
    optimizer = torch.optim.SGD(model.parameters(), lr=a.learning_rate)
    offset = 0
    optimizer_steps = 0
    if a.resume and candidate_path.exists():
        saved = torch.load(candidate_path, map_location="cpu", weights_only=False)
        if saved.get("source_champion_sha256") != source_sha or saved.get("canonical_training_source_sha256") != training_sha:
            raise RuntimeError("resume checkpoint source hashes do not match current canonical inputs")
        model.load_state_dict(saved["state_dict"], strict=True)
        order = saved["epoch_order"]
        offset = int(saved["next_offset"])
        optimizer_steps = int(saved["optimizer_steps"])
        loader.coverage.exposure[:] = saved["coverage_exposure"]
        loader.coverage.rng.bit_generator.state = saved["coverage_rng_state"]

    baseline_val = (saved.get("validation_metrics_before_training") if a.resume and candidate_path.exists()
                    else evaluate(model, val_batches, device))
    if baseline_val is None:
        baseline_val = evaluate(model, val_batches, device)
    if offset == 0:
        with log_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"event":"start","device":torch.cuda.get_device_name(device),
                "source_sha256":source_sha,"canonical_files_once":[str(a.source)],
                "epochs":1,"optimizer_steps_target":len(order),"train_rows":len(train_times),
                "validation_rows":len(validation_times),"test_accessed":False,
                "sequence_length":a.sequence_length,"symbols_per_sample":a.symbols_per_sample,
                "baseline_validation":baseline_val},allow_nan=False)+"\n")
        save_candidate(candidate_path, model, cfg, loader, a.champion, source_sha, a.source, training_sha, 0, order, 0, 0, baseline_val)

    model.train()
    epoch_start = time.perf_counter()
    running_loss, loss_count, valid_count, skipped = 0.0, 0, 0, 0
    torch.cuda.reset_peak_memory_stats(device)
    for offset_i in range(offset, len(order)):
        end = int(order[offset_i])
        batch = loader.sample(1, a.symbols_per_sample, a.sequence_length, "train",
            horizon_seconds=a.horizon_seconds, device="cpu", fixed_end_indices=[end])
        x, sid, mid, aid, mask, context, returns, target_valid = batch_tensors(batch, device)
        valid = target_valid[0] & mask[0, -1]
        if valid.any():
            logits, values = model(x, sid, mid, aid, mask, context)
            expected, _ = utility(logits[0], returns[0], valid)
            loss = -expected[valid].mean() + 0.5 * nn.functional.smooth_l1_loss(
                values[0, valid].float(), expected[valid].detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
            optimizer.step()
            optimizer_steps += 1
            running_loss += float(loss.detach().cpu())
            loss_count += 1
            valid_count += int(valid.sum())
        else:
            skipped += 1
        step = offset_i + 1
        if step % 100 == 0 or step == 1 or step == len(order):
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - epoch_start
            coverage = loader.coverage_report()
            coverage = {k: coverage[k] for k in ("symbols", "sampled_symbols", "min_exposure",
                "mean_exposure", "median_exposure", "p90_exposure", "max_exposure", "zero_exposure")}
            row = {"event":"progress","epoch":1,"step":step,"steps_total":len(order),
                "optimizer_steps":optimizer_steps,"skipped_no_reward":skipped,
                "mean_train_loss":running_loss/max(loss_count,1),"valid_reward_symbols":valid_count,
                "steps_per_second":step/max(elapsed,1e-9),
                "peak_vram_bytes":torch.cuda.max_memory_allocated(device),
                "coverage":coverage}
            with log_path.open("a",encoding="utf-8") as f:
                f.write(json.dumps(row,allow_nan=False)+"\n")
            print(json.dumps({k:v for k,v in row.items() if k!="coverage"},allow_nan=False),flush=True)
        if step % a.validation_every == 0:
            metrics=evaluate(model,val_batches,device)
            with log_path.open("a",encoding="utf-8") as f:
                f.write(json.dumps({"event":"validation","step":step,"metrics":metrics},allow_nan=False)+"\n")
            print(json.dumps({"event":"validation","step":step,"metrics":metrics},allow_nan=False),flush=True)
        if step % a.checkpoint_every == 0 or step == len(order):
            loader.save_coverage()
            save_candidate(candidate_path,model,cfg,loader,a.champion,source_sha,a.source,training_sha,step,order,step,
                           optimizer_steps,baseline_val)
            print(json.dumps({"event":"checkpoint","step":step,"path":str(candidate_path)}),flush=True)

    final_val=evaluate(model,val_batches,device)
    candidate_beats = final_val["valid_decisions"] > 0 and baseline_val["valid_decisions"] > 0 and \
        final_val["mean_net_return"] > baseline_val["mean_net_return"]
    test_result = None
    if candidate_beats:
        # The test split is created and evaluated only after the final
        # validation gate has been completed.
        test_times = make_time_indices(loader, "test", a.sequence_length, a.horizon_seconds)
        n_test = min(a.validation_batches, len(test_times))
        exp_before = loader.coverage.exposure.copy()
        rng_before = deepcopy(loader.coverage.rng.bit_generator.state)
        selected_test = test_times[np.linspace(0, len(test_times)-1, n_test, dtype=np.int64)]
        test_batches = [loader.sample(1,a.symbols_per_sample,a.sequence_length,"test",
            horizon_seconds=a.horizon_seconds,device="cpu",fixed_end_indices=[int(t)])
            for t in selected_test]
        loader.coverage.exposure[:] = exp_before
        loader.coverage.rng.bit_generator.state = rng_before
        candidate_test = evaluate(model,test_batches,device)
        # Restore the pre-training backbone and zero context adapter for the
        # matching champion test score. The checkpoint already contains the
        # trained candidate and the original champion file remains untouched.
        model.backbone.load_state_dict(initial_state,strict=True)
        nn.init.zeros_(model.context_policy.weight)
        nn.init.zeros_(model.context_value.weight)
        champion_test = evaluate(model,test_batches,device)
        test_result={"candidate":candidate_test,"champion":champion_test,
                     "candidate_beats_champion":candidate_test["valid_decisions"]>0 and
                        champion_test["valid_decisions"]>0 and
                        candidate_test["mean_net_return"]>champion_test["mean_net_return"]}
    loader.save_coverage()
    exposure_path=a.runtime/"one-epoch-symbol-exposure.csv"
    import pandas as pd
    pd.DataFrame({"instrument":loader.symbol_names,
                  "direct_symbol_id":np.arange(len(loader.symbol_names)),
                  "training_exposures":loader.coverage.exposure}).to_csv(exposure_path,index=False)
    final={"status":"completed_one_epoch","source":str(a.source),
        "source_sha256":sha256(a.source),"champion":str(a.champion),"champion_sha256":source_sha,
        "candidate":str(candidate_path),"device":torch.cuda.get_device_name(device),
        "parameters":parameter_count(model.backbone),"train_time_rows":len(train_times),
        "validation_time_rows":len(validation_times),"sequence_length":a.sequence_length,
        "symbols_per_sample":a.symbols_per_sample,"horizon_seconds":a.horizon_seconds,
        "steps":len(order),"optimizer_steps":optimizer_steps,"skipped_no_reward":skipped,
        "mean_train_loss":running_loss/max(loss_count,1),"baseline_validation":baseline_val,
        "candidate_validation":final_val,"candidate_beats_champion_validation":candidate_beats,
        "test_result":test_result,"test_accessed":test_result is not None,
        "promotion":"not performed; report candidate eligibility only",
        "coverage":loader.coverage_report(),"coverage_csv":str(exposure_path),
        "elapsed_seconds":time.perf_counter()-epoch_start,
        "peak_vram_bytes":torch.cuda.max_memory_allocated(device),
        "log":str(log_path)}
    (a.runtime/"one-epoch-training-final.json").write_text(json.dumps(final,indent=2),encoding="utf-8")
    if test_result is not None:
        # The model was restored to champion for test comparison. Reload the
        # saved candidate state so the resumable/latest artifact reflects the
        # trained weights, not the temporary baseline evaluation weights.
        saved_candidate=torch.load(candidate_path,map_location=device,weights_only=False)
        model.load_state_dict(saved_candidate["state_dict"],strict=True)
    save_candidate(candidate_path,model,cfg,loader,a.champion,source_sha,a.source,training_sha,len(order),order,len(order),
                   optimizer_steps,baseline_val)
    print(json.dumps(final,indent=2,allow_nan=False),flush=True)
    loader.close()


if __name__ == "__main__":
    main()
