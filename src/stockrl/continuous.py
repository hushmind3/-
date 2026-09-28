"""Paper-trading observation, delayed outcome labeling, replay, and model promotion."""
from __future__ import annotations

import json
import shutil
import time
from copy import deepcopy
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .core import (Config, TemporalActorCritic, device_for, evaluate, infer, load_checkpoint,
                   load_market, observations, save_checkpoint, seed_all)


class ReplayBuffer:
    """Small persistent reservoir replay; each record is an already matured decision label."""
    def __init__(self, path: Path, capacity: int = 10000):
        self.path, self.capacity = path, capacity
        self.items: deque[dict[str, Any]] = deque(maxlen=capacity)
        if path.exists():
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                self.items.extend(payload.get("items", []))
            except Exception:
                # Keep a corrupt replay file recoverable rather than blocking observation.
                shutil.copy2(path, path.with_suffix(path.suffix + ".corrupt"))

    def add(self, item: dict[str, Any]) -> None:
        self.items.append(item)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save({"items": list(self.items)}, tmp)
        tmp.replace(self.path)


def _decision(model: TemporalActorCritic, df: pd.DataFrame, cfg: Config, device: torch.device) -> dict[str, Any]:
    result = infer(model, df, cfg.window, device)
    result["timestamp"] = str(df.date.iloc[-1])
    return result


def _paper_result(log: list[dict[str, Any]], current: pd.DataFrame, horizon: int, fee: float) -> None:
    """Mature every virtual decision after horizon rows and store reward/action labels."""
    if len(current) < 2:
        return
    latest_i = len(current) - 1
    close = current.close.to_numpy(float)
    for row in log:
        if row.get("matured") or row["index"] + horizon > latest_i:
            continue
        start = row["index"]
        end = start + horizon
        position = int(row["action_index"]) - 1
        reward = position * (close[end] / close[start] - 1.0) - fee * abs(position)
        row["matured"] = True
        row["exit_timestamp"] = str(current.date.iloc[end])
        row["reward"] = float(reward)
        row["correct"] = bool((reward > 0 and position != 0) or (position == 0 and abs(close[end]/close[start]-1) <= fee))


def _replay_supervised_update(model: TemporalActorCritic, replay: ReplayBuffer, cfg: Config,
                              device: torch.device, checkpoint_dir: Path, train_before_index: int,
                              batch_size: int = 64) -> bool:
    """Incremental actor/critic update from replayed realized rewards and action labels."""
    eligible = [r for r in replay.items if int(r.get("source_index", -1)) < train_before_index]
    if len(eligible) < 8:
        return False
    sample = eligible[-min(len(eligible), max(batch_size, 256)):]
    rng = np.random.default_rng(cfg.seed + len(replay.items))
    ix = rng.choice(len(sample), size=min(batch_size, len(sample)), replace=len(sample) < batch_size)
    batch = [sample[int(i)] for i in ix]
    xs = torch.as_tensor(np.asarray([r["obs"] for r in batch], np.float32), device=device)
    # Positive realized reward reinforces the chosen action; a losing decision
    # is corrected toward HOLD to avoid teaching the model to repeat the loss.
    actions = torch.as_tensor([r["action_index"] if r["reward"] > 0 else 1 for r in batch], dtype=torch.long, device=device)
    rewards = torch.as_tensor([r["reward"] for r in batch], dtype=torch.float32, device=device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 0.03)
    logits, values = model(xs)
    loss = torch.nn.functional.cross_entropy(logits, actions) + 0.2 * torch.nn.functional.smooth_l1_loss(values, rewards)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    save_checkpoint(checkpoint_dir / "candidate.pt", model, cfg)
    model.eval()
    return True


def _score(model: TemporalActorCritic, df: pd.DataFrame, cfg: Config, device: torch.device) -> dict[str, float]:
    x = observations(df, cfg.window, model.feature_names)
    # Every supplied row is the validation episode. Keep at least one reward transition.
    return evaluate(model, df, x, np.arange(len(df)), cfg, device)


def _promote_if_better(candidate: Path, champion: Path, validation: pd.DataFrame,
                       cfg: Config, device: torch.device, minimum_delta: float) -> dict[str, Any]:
    candidate_model, _ = load_checkpoint(candidate, device)
    new_score = _score(candidate_model, validation, cfg, device)
    old_score = None
    if champion.exists():
        old_model, _ = load_checkpoint(champion, device)
        old_score = _score(old_model, validation, cfg, device)
    promoted = old_score is None or new_score["return"] >= old_score["return"] + minimum_delta
    if promoted:
        shutil.copy2(candidate, champion)
    return {"promoted": promoted, "candidate": new_score, "champion_before": old_score}


def run_continuous(args: Any) -> None:
    cfg = Config(window=args.window, fee=args.fee, device=args.device, seed=args.seed)
    seed_all(cfg.seed); device = device_for(cfg.device)
    data_path = Path(args.data); state_dir = Path(args.state_dir); state_dir.mkdir(parents=True, exist_ok=True)
    champion = Path(args.champion); candidate = state_dir / "candidate.pt"
    replay = ReplayBuffer(state_dir / "replay.pt", args.replay_capacity)
    decisions_path = state_dir / "paper_decisions.jsonl"
    feedback_path = Path(args.feedback) if args.feedback else None
    seen: set[str] = set()
    if decisions_path.exists():
        for line in decisions_path.read_text(encoding="utf-8").splitlines():
            try: seen.add(json.loads(line)["timestamp"])
            except Exception: pass
    model = load_checkpoint(champion, device)[0] if champion.exists() else None
    if model is None:
        raise FileNotFoundError(f"Champion checkpoint not found: {champion}. Train an initial model first.")
    print(f"continuous_mode=paper device={device} poll_seconds={args.poll_seconds} horizon={args.horizon}")

    while True:
        try:
            full = load_market(data_path)
            if feedback_path and feedback_path.exists():
                feedback = pd.read_csv(feedback_path)
                if {"timestamp", "action", "reward"}.issubset(feedback.columns):
                    date_keys = full.date.astype(str).to_numpy()
                    key_to_index = {key: i for i, key in enumerate(date_keys)}
                    known = {str(r.get("feedback_id")) for r in replay.items if r.get("feedback_id") is not None}
                    for _, fb in feedback.iterrows():
                        key = str(pd.to_datetime(fb["timestamp"]))
                        if key not in key_to_index: continue
                        action_name = str(fb["action"]).upper()
                        action_index = {"SELL": 0, "HOLD": 1, "BUY": 2}.get(action_name)
                        feedback_id = f"{key}:{action_name}"
                        if action_index is None or feedback_id in known: continue
                        end_idx = key_to_index[key]
                        replay.add({"obs": observations(full.iloc[:end_idx+1], cfg.window, model.feature_names)[-1],
                                    "action_index": action_index, "reward": float(fb["reward"]),
                                    "source_index": end_idx, "feedback_id": feedback_id})
                    replay.save()
            # A matured decision needs a later timestamp; train/compare only on an old, fixed prefix.
            if len(full) >= max(cfg.window + args.horizon + 20, 80):
                now_key = str(full.date.iloc[-1])
                if now_key not in seen:
                    observation_df = full.iloc[:].copy()
                    result = _decision(model, observation_df, cfg, device)
                    action_index = {"SELL": 0, "HOLD": 1, "BUY": 2}[result["action"]]
                    # Persist a compact feature window so later updates need not reconstruct old raw feeds.
                    obs = observations(observation_df, cfg.window, model.feature_names)[-1]
                    row = {"index": len(full)-1, "timestamp": now_key, "action": result["action"],
                           "action_index": action_index, "probabilities": result["probabilities"],
                           "value": result["value"], "matured": False}
                    log = []
                    if decisions_path.exists():
                        for line in decisions_path.read_text(encoding="utf-8").splitlines():
                            try: log.append(json.loads(line))
                            except Exception: pass
                    log.append(row)
                    print("paper_decision=" + json.dumps(result, ensure_ascii=False))
                    # A local tool can poll this append-only signal file. It does not
                    # place orders; execution belongs to the external tool after user action.
                    signal_path = state_dir / "latest_signal.json"
                    tmp_signal = signal_path.with_suffix(".tmp")
                    tmp_signal.write_text(json.dumps({**result, "mode": "paper", "source": "stockrl",
                        "model": str(champion), "generated_at": time.time()}, ensure_ascii=False, indent=2), encoding="utf-8")
                    tmp_signal.replace(signal_path)
                    # Label decisions only once their future horizon has arrived.
                    _paper_result(log, full, args.horizon, cfg.fee)
                    matured = [r for r in log if r.get("matured") and not r.get("replayed")]
                    for r in matured:
                        start = int(r["index"])
                        if start >= cfg.window - 1:
                            replay.add({"obs": observations(full.iloc[:start+1], cfg.window, model.feature_names)[-1],
                                        "action_index": int(r["action_index"]), "reward": float(r["reward"]),
                                        "source_index": start})
                            r["replayed"] = True
                    decisions_path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in log), encoding="utf-8")
                    seen.add(now_key); replay.save()

                    # Online adaptation uses labeled replay only. Compare against the incumbent
                    # on a chronological slice ending before the latest observation.
                    if len(replay.items) >= args.min_replay and len(full) >= args.validation_rows + args.horizon + 2:
                        val = full.iloc[-(args.validation_rows + args.horizon):-args.horizon].reset_index(drop=True)
                        validation_start = len(full) - (args.validation_rows + args.horizon)
                        candidate_model = deepcopy(model)
                        if _replay_supervised_update(candidate_model, replay, cfg, device, state_dir, validation_start):
                            decision = _promote_if_better(candidate, champion, val, cfg, device, args.minimum_delta)
                            print("candidate_evaluation=" + json.dumps(decision))
                            if decision["promoted"]:
                                model, _ = load_checkpoint(champion, device)
                            else:
                                candidate.unlink(missing_ok=True)
            if args.once:
                break
            time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            print("continuous_loop_stopped_by_user"); break
        except (FileNotFoundError, pd.errors.EmptyDataError):
            if args.once: raise
            print("waiting_for_market_data_file"); time.sleep(args.poll_seconds)


def run_replay(args: Any) -> None:
    """Finite replay smoke/demo: make virtual decisions as if bars arrived in sequence."""
    cfg = Config(window=args.window, fee=args.fee, device=args.device, seed=args.seed)
    seed_all(cfg.seed); device=device_for(cfg.device); model, _ = load_checkpoint(args.checkpoint, device)
    df = load_market(args.data)
    state_dir = Path(args.state_dir); state_dir.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, Any]] = []
    replay = ReplayBuffer(state_dir / "replay.pt", args.replay_capacity)
    x = observations(df, cfg.window, model.feature_names)
    for i in range(max(cfg.window, 1), len(df)):
        if i % args.stride: continue
        partial = df.iloc[:i+1]
        pred = _decision(model, partial, cfg, device)
        action_index = {"SELL": 0, "HOLD": 1, "BUY": 2}[pred["action"]]
        log.append({"index": i, "timestamp": pred["timestamp"], "action": pred["action"],
                    "action_index": action_index, "matured": False})
        _paper_result(log, df, args.horizon, cfg.fee)
        for row in log:
            if row.get("matured") and not row.get("replayed"):
                replay.add({"obs": x[row["index"]], "action_index": row["action_index"],
                            "reward": row["reward"], "source_index": row["index"]})
                row["replayed"] = True
    replay.save()
    (state_dir / "replay_decisions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in log), encoding="utf-8")
    matured = [r for r in log if r.get("matured")]
    print(json.dumps({"decisions": len(log), "matured": len(matured), "replay_items": len(replay.items),
                      "mean_reward": float(np.mean([r["reward"] for r in matured])) if matured else 0.0}, indent=2))
