"""Asynchronous online actor-critic over global asset panels.

Market observation and candidate training use separate threads and separate
model instances. Only a held-out score improvement swaps the champion.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import json, os, random, shutil, threading, time
import re

import numpy as np
import psutil
import torch
from torch import nn
from torch.distributions import Categorical

from .global_transformer import (ACTION_NAMES, GlobalMarketPanel, GlobalMarketTransformer,
                                 TransformerConfig, parameter_count, load_compatible_state_dict)
from .paper_account import PaperAccount

REWARD_VERSION="net_trade_v2"
REWARD_DEFINITION=("net_trade_v2: one equal-notional isolated horizon trade; BUY/SELL realize signed price return, "
                   "minus entry+exit fees and slippage; HOLD remains cash")


@dataclass
class Experience:
    features: np.ndarray
    symbol_ids: np.ndarray
    market_ids: np.ndarray
    asset_ids: np.ndarray
    valid_mask: np.ndarray
    symbol_index: int
    action: int
    reward: float
    timestamp: str
    source: str = "paper"
    regime: float = 0.0
    reward_version: str = "net_trade_v2"
    market_context: np.ndarray | None = None
    portfolio_state: np.ndarray | None = None
    account_state: np.ndarray | None = None
    portfolio_reward: float | None = None
    portfolio_transition: bool = False
    forward_return: float | None = None


class GlobalReplayBuffer:
    """Bounded replay with explicit recent/old/extreme regime mixture."""
    def __init__(self, capacity: int = 100_000, seed: int = 7):
        self.capacity=capacity; self.items: deque[Experience]=deque(maxlen=capacity)
        self.rng=random.Random(seed); self.lock=threading.Lock()
    def add(self, exp: Experience) -> None:
        with self.lock: self.items.append(exp)
    def __len__(self):
        with self.lock: return len(self.items)
    def sample(self, batch_size: int) -> list[Experience]:
        with self.lock: items=list(self.items)
        if not items: return []
        teachers=[x for x in items if x.source.startswith("teacher")]
        paper=[x for x in items if not x.source.startswith("teacher")
               and getattr(x,"reward_version","legacy_transition_v1")==REWARD_VERSION]
        # Keep old samples on disk for auditability, but don't mix their former
        # position-transition rewards with the new isolated-horizon trade target.
        items=paper+teachers
        if not items: return []
        n=min(batch_size,len(items))
        teachers=[x for x in items if x.source.startswith("teacher")]
        paper=[x for x in items if not x.source.startswith("teacher")] or items
        valid_paper=[x for x in items if not x.source.startswith("teacher")
                     and getattr(x,"reward_version","legacy_transition_v1")==REWARD_VERSION]
        # Warm-start from demonstrations, then linearly remove their influence
        # as genuine net-PnL paper experiences accumulate. At 10k self outcomes
        # the replay becomes fully self-experience driven.
        teacher_fraction=(max(0.0,1.0-len(valid_paper)/10_000.0) if teachers else 0.0)
        recent=paper[-max(int(len(paper)*.2),1):]
        older=paper[:-len(recent)] or paper
        extreme=[x for x in paper if abs(x.regime)>.025 or abs(x.reward)>.02] or paper
        chosen=[]
        for _ in range(n):
            if teachers and (not valid_paper or self.rng.random()<teacher_fraction):
                chosen.append(self.rng.choice(teachers)); continue
            bucket=self.rng.choices((recent,older,extreme),weights=(.5,.3,.2),k=1)[0]
            chosen.append(self.rng.choice(bucket))
        return chosen
    def teacher_fraction(self)->float:
        with self.lock:
            teachers=any(x.source.startswith("teacher") for x in self.items)
            paper=sum(not x.source.startswith("teacher") for x in self.items)
        return max(0.0,1.0-paper/10_000.0) if teachers else 0.0
    def sample_source(self, source:str) -> Experience|None:
        with self.lock:
            rows=[x for x in self.items if x.source.startswith(source)]
            return self.rng.choice(rows) if rows else None
    def trainable_count(self)->int:
        with self.lock:
            return sum(x.source.startswith("teacher") or
                       getattr(x,"reward_version","legacy_transition_v1")==REWARD_VERSION for x in self.items)
    def save(self,path:Path):
        with self.lock: payload=list(self.items)
        _atomic_save(payload,path)
    def load(self,path:Path):
        if path.exists():
            data=torch.load(path,map_location="cpu",weights_only=False)
            with self.lock:
                for x in data[-self.capacity:]: self.items.append(x)


def _atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(obj,tmp); os.replace(tmp,path)


def _atomic_json(obj,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2),encoding="utf-8"); os.replace(tmp,path)


def should_promote(candidate_score:float, champion_score:float, minimum_delta:float=0.0)->bool:
    """Finite, strict promotion gate shared by online control and its checks."""
    return bool(np.isfinite(candidate_score) and np.isfinite(champion_score)
                and candidate_score>champion_score+minimum_delta)


def parse_horizon(value: int | str) -> tuple[str, int]:
    """Return (bars|seconds, amount); durations resolve to the first observed bar at/after target."""
    if isinstance(value, int):
        if value < 1: raise ValueError("horizon bars must be >= 1")
        return "bars", value
    match=re.fullmatch(r"\s*(\d+)\s*(bars?|s|sec|seconds?|m|min|minutes?|h|hours?)\s*",str(value).lower())
    if not match: raise ValueError("horizon must be e.g. 30s, 1m, 5m, 1bar, or an integer bar count")
    amount=int(match.group(1)); unit=match.group(2)
    if amount<1: raise ValueError("horizon duration must be >= 1")
    if unit.startswith("bar"): return "bars",amount
    scale=1 if unit in ("s","sec","second","seconds") else 60 if unit in ("m","min","minute","minutes") else 3600
    return "seconds",amount*scale


def net_action_reward(action:int, forward_return:float, previous_position:int,
                      fee:float, slippage:float) -> tuple[float,float,float]:
    """One equal-notional paper trade held to horizon; charge entry and exit costs."""
    position=int(action)-1
    # Each decision is evaluated as an isolated paper trade over its configured
    # horizon. BUY/SELL pay for opening and closing; HOLD stays in cash.
    turnover=2 if position else 0
    gross=float(position)*float(forward_return)
    fee_cost=float(fee)*turnover; slippage_cost=float(slippage)*turnover
    return gross-fee_cost-slippage_cost,fee_cost,slippage_cost


def save_model(path:Path, model:GlobalMarketTransformer, cfg:TransformerConfig, optimizer=None, step=0):
    payload={"state_dict":model.state_dict(),"config":asdict(cfg),"step":step,
             "optimizer":optimizer.state_dict() if optimizer is not None else None}
    if getattr(model,"_stockrl_uses_market_context",False):
        from .market_training import CONTEXT_FEATURES
        payload["context_features"]=list(CONTEXT_FEATURES)
        payload["symbol_map"]=model._stockrl_symbol_map
        payload["market_context_model"]=True
    _atomic_save(payload,path)


def load_model(path:Path,device:torch.device):
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    cfg=TransformerConfig(**ckpt["config"])
    state=ckpt["state_dict"]
    contextual=(bool(ckpt.get("market_context_model")) or
                ("context_policy.weight" in state and
                 any(k.startswith("backbone.") for k in state)))
    if contextual:
        from .market_training import CONTEXT_FEATURES, ContextConditionedTransformer
        if ckpt.get("context_features",list(CONTEXT_FEATURES))!=list(CONTEXT_FEATURES):
            raise ValueError("checkpoint market_context feature order does not match the inference schema")
        symbol_map=ckpt.get("symbol_map")
        if not isinstance(symbol_map,dict) or len(symbol_map)!=cfg.max_symbols:
            raise ValueError("context checkpoint must contain its complete deterministic symbol_map")
        backbone=GlobalMarketTransformer(cfg)
        if device.type=="cuda": backbone=backbone.half()
        model=ContextConditionedTransformer(backbone)
        model.context_policy.float(); model.context_value.float()
        load_compatible_state_dict(model,state,strict=True)
        model._stockrl_uses_market_context=True
        model._stockrl_symbol_map=symbol_map
    else:
        model=GlobalMarketTransformer(cfg)
        load_compatible_state_dict(model,state,strict=True)
        if device.type=="cuda": model=model.half()
    del ckpt
    model.to(device).eval()
    return model,cfg


class OnlineGlobalAgent:
    def __init__(self, state_dir: str|Path, device="auto", config:TransformerConfig|None=None,
                 capacity=100_000, window=128, horizon=1, fee=.001, slippage_bps=1.0,
                 min_replay=8, batch_size=2, updates_per_candidate=1, lr=2e-6, seed=7,
                 candidate_interval=256, initial_champion: str|Path|None=None,
                 teacher_replay_path: str|Path|None=None):
        from .core import device_for
        self.state_dir=Path(state_dir); self.state_dir.mkdir(parents=True,exist_ok=True)
        self.device=device_for(device); self.cfg=config or TransformerConfig(); self.window=window
        self.horizon=horizon; self.horizon_kind,self.horizon_amount=parse_horizon(horizon)
        self.fee=fee; self.slippage=slippage_bps/10_000
        self.min_replay=min_replay; self.batch_size=batch_size; self.updates_per_candidate=updates_per_candidate
        self.candidate_interval=max(1,candidate_interval)
        self.min_validation_dates=32
        self.lr=lr; self.seed=seed; self.lock=threading.RLock(); self.stop=threading.Event()
        self.replay=GlobalReplayBuffer(capacity,seed); self.replay_path=self.state_dir/"replay.pt"
        self.paper_replay_path=self.state_dir/"paper_replay.pt"
        # Load warm-start examples first, then append this runtime's persistent
        # self-experience so bounded replay evicts the oldest teacher rows first.
        if teacher_replay_path: self.replay.load(Path(teacher_replay_path))
        self.replay.load(self.replay_path)
        self.paper_experiences=[]
        if self.paper_replay_path.exists():
            try:
                self.paper_experiences=torch.load(self.paper_replay_path,map_location="cpu",weights_only=False)
                for exp in self.paper_experiences[-capacity:]: self.replay.add(exp)
            except (OSError,RuntimeError,EOFError): self.paper_experiences=[]
        self.champion_path=self.state_dir/"champion.pt"; self.backup_path=self.state_dir/"champion.previous.pt"
        if not self.champion_path.exists() and initial_champion and Path(initial_champion).is_file():
            # Seed this isolated runtime from a prior verified champion without
            # writing to or replacing the source checkpoint.
            shutil.copy2(initial_champion,self.champion_path)
        if self.champion_path.exists():
            try:
                self.champion,self.cfg=load_model(self.champion_path,self.device)
                seed_meta_path=(Path(initial_champion).with_name("champion.itch-promotion.json")
                                if initial_champion else None)
                seed_meta={}
                if seed_meta_path and seed_meta_path.exists():
                    try: seed_meta=json.loads(seed_meta_path.read_text(encoding="utf-8"))
                    except (OSError,json.JSONDecodeError): pass
                seed_is_context_candidate=(seed_meta.get("promotion_mode") ==
                                           "byte-for-byte candidate copy; no weight conversion")
                if (initial_champion and Path(initial_champion).is_file() and seed_is_context_candidate
                        and not getattr(self.champion,"_stockrl_uses_market_context",False)):
                    # Upgrade a previous web-profile's legacy checkpoint to the
                    # explicitly selected raw candidate, preserving the local
                    # checkpoint before the schema migration.
                    backup=self.champion_path.with_name("champion.pre-context-input.pt")
                    if backup.exists():
                        backup=self.champion_path.with_name(f"champion.pre-context-input-{int(time.time())}.pt")
                    shutil.copy2(self.champion_path,backup)
                    del self.champion
                    if self.device.type=="cuda": torch.cuda.empty_cache()
                    shutil.copy2(initial_champion,self.champion_path)
                    self.champion,self.cfg=load_model(self.champion_path,self.device)
            except Exception:
                if not self.backup_path.exists(): raise
                shutil.copy2(self.backup_path,self.champion_path)
                self.champion,self.cfg=load_model(self.champion_path,self.device)
        else:
            self.champion=GlobalMarketTransformer(self.cfg)
            if self.device.type=="cuda": self.champion=self.champion.half()
            self.champion=self.champion.to(self.device).eval()
            save_model(self.champion_path,self.champion,self.cfg)
        self.candidate=None; self.optimizer=None; self.steps=0; self.updates=0
        self.last_train_replay_size=0
        self.validation=[]
        self.portfolio_validation=[]
        self.positions={}
        self.paper_account=PaperAccount(self.state_dir/"paper_account.json",self.fee,self.slippage)
        if not self.paper_account.path.exists(): self.paper_account.save()
        positions_path=self.state_dir/"live_positions.json"
        if positions_path.exists():
            saved_positions=json.loads(positions_path.read_text(encoding="utf-8"))
            for key,value in saved_positions.items():
                try:
                    parsed_key=int(key)
                    parsed_key=parsed_key if str(parsed_key)==str(key) else str(key)
                except (TypeError,ValueError):
                    parsed_key=str(key)
                self.positions[parsed_key]=int(value)
        self.metrics={"observations":0,"decisions":0,"matured":0,"updates":0,"promotions":0,
                      "rejections":0,"nonfinite_updates":0,"teacher_examples_trained":0,"paper_examples_trained":0,
                      "paper_net_reward":0.0,"last_market_timestamp":None,"candidate_training":False,
                      "last_update_utc":None,"last_promotion_utc":None,"last_rejection_utc":None,
                      "inference_during_candidate":0,"reward_definition":REWARD_DEFINITION,
                      "update_losses":[],"inference_seconds":[],
                      "update_seconds":[],"weight_delta_l1":[]}
        metrics_path=self.state_dir/"metrics.json"
        if metrics_path.exists():
            try:
                self.metrics.update(json.loads(metrics_path.read_text(encoding="utf-8")))
            except (OSError,json.JSONDecodeError):
                pass
        self.metrics["legacy_reward_examples_ignored"]=sum(
            not x.source.startswith("teacher") and getattr(x,"reward_version","legacy_transition_v1")!=REWARD_VERSION
            for x in self.replay.items)
        self.metrics["portfolio_experiences"] = max(int(self.metrics.get("portfolio_experiences", 0)),
                                                     len(self.paper_experiences))
        self.updates=int(self.metrics.get("updates",0)); self.steps=self.updates
        # Candidate/optimizer objects are intentionally ephemeral. Restarting
        # resumes from the latest replay and incumbent model checkpoint.
        self.metrics["candidate_training"]=False
        self.last_train_replay_size=len(self.replay)
        self.validation_path=self.state_dir/"live_validation.pt"
        self.validation_meta_path=self.state_dir/"validation_config.json"
        validation_meta={}
        try: validation_meta=json.loads(self.validation_meta_path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError): pass
        if self.validation_path.exists() and validation_meta=={"reward_version":REWARD_VERSION,"horizon":str(self.horizon)}:
            try: self.validation=torch.load(self.validation_path,map_location="cpu",weights_only=False)
            except (OSError,RuntimeError,EOFError): self.validation=[]
        portfolio_validation_path=self.state_dir/"paper_validation.pt"
        if portfolio_validation_path.exists():
            try: self.portfolio_validation=torch.load(portfolio_validation_path,map_location="cpu",weights_only=False)
            except (OSError,RuntimeError,EOFError): self.portfolio_validation=[]
        elif self.validation_path.exists():
            self.metrics["legacy_validation_ignored"]=True
        _atomic_json({"reward_version":REWARD_VERSION,"horizon":str(self.horizon)},self.validation_meta_path)
        self.thread=threading.Thread(target=self._learner,name="global-learner",daemon=True)

    def _target_index(self,panel:GlobalMarketPanel,start:int,symbol:int)->int|None:
        future=np.flatnonzero(panel.observed[start+1:,symbol])+start+1
        if not len(future): return None
        if self.horizon_kind=="bars":
            return int(future[self.horizon_amount-1]) if len(future)>=self.horizon_amount else None
        target=panel.dates[start]+np.timedelta64(self.horizon_amount,"s")
        eligible=future[panel.dates[future]>=target]
        return int(eligible[0]) if len(eligible) else None

    def start(self): self.thread.start()
    def close(self):
        self.stop.set()
        if self.thread.is_alive(): self.thread.join(timeout=300)
        try:
            self.replay.save(self.replay_path)
        except (MemoryError, RuntimeError, OSError) as exc:
            # The previous replay snapshot is safer than taking down the live
            # observer while serializing a multi-GB in-memory deque.
            self.metrics["replay_save_error"] = f"{type(exc).__name__}: {exc}"
            self._write_metrics()

    def _infer(self,panel,index,portfolio_state=None,account_state=None):
        args=[x.to(self.device) for x in self._window(panel,index)]
        args[0]=args[0].to(dtype=next(self.champion.parameters()).dtype)
        with self.lock,torch.inference_mode():
            if self.device.type=="cuda": torch.cuda.synchronize(self.device)
            t=time.perf_counter()
            if getattr(self.champion,"_stockrl_uses_market_context",False) and portfolio_state is not None:
                pstate=torch.as_tensor(np.asarray(portfolio_state,dtype=np.float32)[None],device=self.device)
                astate=torch.as_tensor(np.asarray(account_state,dtype=np.float32)[None],device=self.device)
                logits,values,allocation=self.champion(*args,portfolio_state=pstate,
                                                        account_state=astate,return_allocation=True)
            else:
                logits,values=self.champion(*args); allocation=None
            elapsed=time.perf_counter()-t
            if self.device.type=="cuda":
                torch.cuda.synchronize(self.device); elapsed=time.perf_counter()-t
        self.metrics["inference_seconds"].append(elapsed)
        if self.metrics.get("candidate_training"):
            self.metrics["inference_during_candidate"]=int(self.metrics.get("inference_during_candidate",0))+1
        if len(self.metrics["inference_seconds"])>2000: self.metrics["inference_seconds"]=self.metrics["inference_seconds"][-2000:]
        return (logits[0].cpu().numpy(),values[0].cpu().numpy(),
                allocation[0].cpu().float().numpy() if allocation is not None else None)

    def _window(self,panel,index):
        contextual=getattr(self.champion,"_stockrl_uses_market_context",False)
        return panel.window(index,self.window,include_context=contextual)

    def _mature(self,pending,panel,end_index):
        keep=[]
        for dec in pending:
            # Exchanges do not share a common bar clock. Wait for the chosen
            # instrument's next N observed bars instead of rewarding a stock
            # on a timestamp where only FX/crypto happened to update.
            if dec.get("timestamp"):
                stamp=np.datetime64(dec["timestamp"])
                start_ix=int(np.searchsorted(panel.dates,stamp,side="left"))
                if start_ix>=len(panel.dates) or panel.dates[start_ix]!=stamp:
                    keep.append(dec); continue
                symbol_ix=panel.symbols.index(dec["symbol"]) if dec.get("symbol") in panel.symbols else dec["symbol_index"]
                dec["index"]=start_ix; dec["symbol_index"]=symbol_ix
            symbol_ix=dec["symbol_index"]
            target=self._target_index(panel,dec["index"],symbol_ix)
            if target is None:
                keep.append(dec); continue
            if end_index<target: keep.append(dec); continue
            i=symbol_ix; forward=panel.return_to(dec["index"],target,i)
            self.metrics["matured"]+=1
            if dec.get("is_validation",False):
                self.validation.append((dec["features"],dec["symbol_ids"],dec["market_ids"],dec["asset_ids"],
                    dec["valid_mask"],i,forward,dec["timestamp"],dec.get("previous_position",0),
                    dec.get("market_context")))
                continue
            action=dec["action"]
            reward,fee_cost,slippage_cost=net_action_reward(action,forward,dec.get("previous_position",0),
                                                            self.fee,self.slippage)
            self.metrics["paper_net_reward"]+=float(reward)
            action_name=ACTION_NAMES[action]
            rewards=self.metrics.setdefault("action_net_reward",{"SELL":0.0,"HOLD":0.0,"BUY":0.0})
            counts=self.metrics.setdefault("action_count",{"SELL":0,"HOLD":0,"BUY":0})
            rewards[action_name]=float(rewards.get(action_name,0.0))+float(reward)
            counts[action_name]=int(counts.get(action_name,0))+1
            self.metrics["fee_total"] = float(self.metrics.get("fee_total",0.0))+fee_cost
            self.metrics["slippage_total"] = float(self.metrics.get("slippage_total",0.0))+slippage_cost
            day=str(dec.get("timestamp","")[:10])
            daily=self.metrics.setdefault("daily_net_pnl",{})
            daily[day]=float(daily.get(day,0.0))+float(reward)
            regime=abs(float(panel.features[dec["index"],i,6]))
            self.replay.add(Experience(dec["features"],dec["symbol_ids"],dec["market_ids"],dec["asset_ids"],
                dec["valid_mask"],i,action,float(reward),dec["timestamp"],"paper",max(regime,abs(forward)),
                market_context=dec.get("market_context")))
        return keep

    def _mature_portfolio(self, pending, panel, equity_before: float):
        """Turn completed paper-account transitions into reward experiences.

        The reward is the account's net-of-cost normalized equity change.  It
        is attached to the exact decision window and portfolio state, so the
        learner and promotion gate optimize the same virtual account result.
        """
        if not pending:
            return pending
        now = float(self.paper_account.normalized_equity())
        keep=[]
        for dec in pending:
            reward = now - float(dec.get("equity_before", now))
            self.metrics["paper_account_reward"] = float(self.metrics.get("paper_account_reward", 0.0)) + reward
            forward_return=None
            if "index" in dec:
                target=self._target_index(panel,int(dec["index"]),int(dec["symbol_index"]))
                if target is not None and target <= len(panel.dates)-1:
                    forward_return=float(panel.return_to(int(dec["index"]),target,int(dec["symbol_index"])))
            exp=Experience(dec["features"],dec["symbol_ids"],dec["market_ids"],dec["asset_ids"],
                    dec["valid_mask"],int(dec["symbol_index"]),int(dec["action"]),float(reward),
                    dec["timestamp"],"paper_account",float(dec.get("regime",0.0)),
                    market_context=dec.get("market_context"), portfolio_state=dec.get("portfolio_state"),
                    account_state=dec.get("account_state"), portfolio_reward=float(reward),
                    portfolio_transition=True,forward_return=forward_return)
            if dec.get("is_validation",False):
                self.portfolio_validation.append(exp)
            else:
                self.replay.add(exp)
                self.paper_experiences.append(exp)
            self.metrics["portfolio_experiences"] = int(self.metrics.get("portfolio_experiences",0)) + 1
        return keep

    def follow_csv(self, data_path:str|Path, poll_seconds:float=5.0, initial_lookback_bars:int=0):
        """Observe an append-only UTC timestamp CSV until Ctrl+C.

        Feed columns follow the same global schema as the historical downloader.
        New bars can be minute/hourly; all instruments for a timestamp should
        be appended as a batch. Candidate training runs on the learner thread.
        """
        from .global_transformer import stable_id
        data_path=Path(data_path); cursor_path=self.state_dir/"live_cursor.json"
        pending_path=self.state_dir/"live_pending.pt"; portfolio_pending_path=self.state_dir/"paper_pending.pt"; decisions_path=self.state_dir/"decisions.csv"
        cursor=None
        last_file_signature=None
        if cursor_path.exists(): cursor=json.loads(cursor_path.read_text(encoding="utf-8")).get("last_timestamp")
        pending=torch.load(pending_path,map_location="cpu",weights_only=False) if pending_path.exists() else []
        portfolio_pending=torch.load(portfolio_pending_path,map_location="cpu",weights_only=False) if portfolio_pending_path.exists() else []
        last_persist=time.monotonic(); persist_interval=30.0
        self.start()
        try:
            while not self.stop.is_set():
                if (self.state_dir/"stop.request").exists():
                    break
                if not data_path.exists():
                    self.stop.wait(poll_seconds); continue
                try:
                    stat=data_path.stat()
                    file_signature=(stat.st_size,stat.st_mtime_ns)
                    if file_signature==last_file_signature:
                        self.stop.wait(poll_seconds); continue
                    panel=GlobalMarketPanel(data_path, max_symbols=self.cfg.max_symbols,
                        symbol_map=getattr(self.champion,"_stockrl_symbol_map",None),
                        # Keep closed-session indices, futures, yields, and other
                        # reference markets in the action set using their last
                        # known quote. The dashboard still marks their quote as
                        # stale; dropping them here hid their BUY/HOLD/SELL
                        # output entirely whenever their venue was closed.
                        recent_timestamps=512, active_stale_seconds=604800)
                except (OSError,ValueError):
                    # A producer may be in the middle of appending a CSV batch.
                    self.stop.wait(min(poll_seconds,1.0)); continue
                if not len(panel.dates):
                    self.stop.wait(poll_seconds); continue
                # The first feed batch establishes the ordered symbol universe;
                # migrate legacy integer-keyed paper positions to symbols.
                if any(isinstance(k,int) for k in self.positions):
                    self.positions={panel.symbols[k] if isinstance(k,int) and 0<=k<len(panel.symbols) else str(k):v
                                    for k,v in self.positions.items()}
                if cursor is None:
                    # Replay a small startup tail so delayed paper outcomes can
                    # mature immediately after the first launch. A persisted
                    # cursor always takes precedence after restart.
                    lookback=max(0,int(initial_lookback_bars))
                    start=max(0,len(panel.dates)-max(lookback,1))
                else:
                    cursor_dt=np.datetime64(cursor)
                    start=int(np.searchsorted(panel.dates,cursor_dt,side="right"))
                for ti in range(start,len(panel.dates)):
                    if self.stop.is_set(): break
                    # Autonomy is distinct from the model's HOLD action. When
                    # disabled, inference and counterfactual learning continue,
                    # but directional paper positions are left unchanged.
                    try:
                        mode_state=json.loads((self.state_dir/"autonomy.json").read_text(encoding="utf-8"))
                        paper_enabled=bool(mode_state.get("paper_enabled",mode_state.get("enabled",True)))
                        observe_enabled=bool(mode_state.get("observe_enabled",True))
                    except (OSError,json.JSONDecodeError,AttributeError):
                        paper_enabled=True; observe_enabled=True
                    self.metrics["autonomy_enabled"]=paper_enabled
                    self.metrics["paper_enabled"]=paper_enabled
                    self.metrics["observe_enabled"]=observe_enabled
                    self.paper_account.process_bar(panel,ti,paper_enabled)
                    portfolio_pending=self._mature_portfolio(portfolio_pending,panel,
                                                             self.paper_account.normalized_equity())
                    pending=self._mature(pending,panel,ti)
                    if not observe_enabled:
                        cursor=str(panel.dates[ti]); self.metrics["observations"]+=1
                        self.metrics["last_market_timestamp"]=cursor
                        self.paper_account.save()
                        _atomic_json({"last_timestamp":cursor},cursor_path)
                        continue
                    pstate,astate=self.paper_account.model_inputs(panel,ti)
                    logits,values,allocation=self._infer(panel,ti,pstate,astate)
                    probs=torch.softmax(torch.as_tensor(logits),-1).numpy()
                    window=self._window(panel,ti)
                    x,sid,mid,aid,mask=window[:5]
                    stamp=str(panel.dates[ti]); validation=stable_id(stamp,5)==0
                    rows=[]
                    for j,symbol in enumerate(panel.symbols):
                        # A closed venue may not print a bar at the current
                        # global timestamp. Use its forward-filled last quote
                        # for a visible BUY/HOLD/SELL decision, while the
                        # observed mask remains untouched for reward maturity.
                        if not panel.observed[:ti + 1, j].any(): continue
                        action=int(np.argmax(probs[j])); previous=self.positions.get(symbol,0)
                        pending.append({"index":ti,"symbol_index":j,"action":action,"previous_position":previous,
                          "timestamp":stamp,"symbol":symbol,"features":x[0].numpy().astype(np.float16),
                          "symbol_ids":sid[0].numpy(),"market_ids":mid[0].numpy(),"asset_ids":aid[0].numpy(),
                          "valid_mask":mask[0].numpy(),"is_validation":validation,"previous_position":previous,
                          "market_context":window[5][0].numpy().astype(np.float16) if len(window)>5 else None})
                        if paper_enabled:
                            portfolio_pending.append({"index":ti,"symbol_index":j,"action":action,"timestamp":stamp,
                              "is_validation":validation,
                              "equity_before":self.paper_account.normalized_equity(),
                              "features":x[0].numpy().astype(np.float16),"symbol_ids":sid[0].numpy(),
                              "market_ids":mid[0].numpy(),"asset_ids":aid[0].numpy(),"valid_mask":mask[0].numpy(),
                              "portfolio_state":np.asarray(pstate,dtype=np.float16),"account_state":np.asarray(astate,dtype=np.float32),
                              "market_context":window[5][0].numpy().astype(np.float16) if len(window)>5 else None})
                        rows.append({"date":stamp,"symbol":symbol,"action":ACTION_NAMES[action],"value":float(values[j]),
                          "p_sell":float(probs[j,0]),"p_hold":float(probs[j,1]),"p_buy":float(probs[j,2])})
                        self.metrics["decisions"]+=1
                        if paper_enabled:
                            self.positions[symbol]=action-1
                    self.paper_account.queue_decisions(panel,ti,probs,paper_enabled,allocation=allocation)
                    self.paper_account.save()
                    import pandas as pd
                    if rows:
                        pd.DataFrame(rows).to_csv(decisions_path,mode="a",header=not decisions_path.exists(),index=False)
                    self.metrics["observations"]+=1; cursor=stamp
                    self.metrics["last_market_timestamp"]=stamp
                    _atomic_json({"last_timestamp":cursor},cursor_path)
                    _atomic_json({str(k):v for k,v in self.positions.items()},self.state_dir/"live_positions.json")
                if pending_path.parent.exists() and (time.monotonic()-last_persist >= persist_interval):
                    _atomic_save(pending,pending_path)
                    _atomic_save(portfolio_pending,portfolio_pending_path)
                    _atomic_save(self.portfolio_validation,self.state_dir/"paper_validation.pt")
                    _atomic_save(self.paper_experiences[-10000:],self.paper_replay_path)
                    last_persist=time.monotonic()
                    _atomic_save(self.validation,self.validation_path)
                self._write_metrics()
                last_file_signature=file_signature
                self.stop.wait(poll_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop.set()
            if self.thread.is_alive(): self.thread.join(timeout=300)
            self.metrics["candidate_training"]=False
            _atomic_save(pending,pending_path)
            _atomic_save(portfolio_pending,portfolio_pending_path)
            _atomic_save(self.portfolio_validation,self.state_dir/"paper_validation.pt")
            _atomic_save(self.paper_experiences[-10000:],self.paper_replay_path)
            _atomic_save(self.validation,self.validation_path)
            self.checkpoint()
            (self.state_dir/"stop.request").unlink(missing_ok=True)

    def run_panel(self,panel:GlobalMarketPanel, max_observations:int|None=None, stride:int=1):
        """Synchronous market observer; learner thread runs concurrently."""
        self.start(); pending=[]; history=[]
        # Chronological split: first 70% can enter replay, next 15% selects a
        # champion, and the final 15% stays sealed for a backtest report.
        cutoff=max(2,int(len(panel.dates)*.70))
        test_start=max(cutoff+1,int(len(panel.dates)*.85))
        self.min_validation_dates=max(2,min(32,int((test_start-cutoff)*.5)))
        count=0
        for ti in range(max(self.window-1,0),len(panel.dates),stride):
            if max_observations is not None and count>=max_observations and ti<cutoff: continue
            pending=self._mature(pending,panel,ti)
            if cutoff<=ti<test_start:
                # A chronological tail is held out from replay/gradient updates.
                # Save point-in-time states and their later market returns for
                # comparing candidate and incumbent on identical unseen bars.
                window=self._window(panel,ti)
                x,sid,mid,aid,mask=window[:5]
                for j in range(len(panel.symbols)):
                    target=self._target_index(panel,ti,j)
                    if not panel.observed[ti,j] or target is None or target>=test_start: continue
                    gross=panel.return_to(ti,target,j)
                    self.validation.append((x[0].numpy().astype(np.float16),sid[0].numpy(),mid[0].numpy(),aid[0].numpy(),
                        mask[0].numpy(),j,gross,str(panel.dates[ti]),0,
                        window[5][0].numpy().astype(np.float16) if len(window)>5 else None))
                continue
            if ti>=test_start: continue
            logits,values,_allocation=self._infer(panel,ti); probs=torch.softmax(torch.as_tensor(logits),-1).numpy()
            window=self._window(panel,ti)
            x,sid,mid,aid,mask=window[:5]
            for j in range(len(panel.symbols)):
                target=self._target_index(panel,ti,j)
                if not panel.observed[ti,j] or target is None or target>=cutoff: continue
                action=int(np.argmax(probs[j])); previous=self.positions.get(j,0)
                pending.append({"index":ti,"symbol_index":j,"action":action,"previous_position":previous,
                  "timestamp":str(panel.dates[ti]),"features":x[0].numpy().astype(np.float16),
                  "symbol_ids":sid[0].numpy(),"market_ids":mid[0].numpy(),"asset_ids":aid[0].numpy(),
                  "valid_mask":mask[0].numpy(),
                  "market_context":window[5][0].numpy().astype(np.float16) if len(window)>5 else None})
                history.append({"date":str(panel.dates[ti]),"symbol":panel.symbols[j],"action":ACTION_NAMES[action],
                    "value":float(values[j]),"p_sell":float(probs[j,0]),"p_hold":float(probs[j,1]),"p_buy":float(probs[j,2])})
                self.metrics["decisions"]+=1
                self.positions[j]=action-1
            self.metrics["observations"]+=1; count+=1
            if count%10==0:
                self.replay.save(self.replay_path)
        # Complete all pending decisions with known later observations.
        if len(panel.dates): pending=self._mature(pending,panel,len(panel.dates)-1)
        self.replay.save(self.replay_path)
        pd_frame=None
        import pandas as pd
        pd_frame=pd.DataFrame(history)
        pd_frame.to_csv(self.state_dir/"decisions.csv",index=False)
        return history

    def evaluate_panel(self,panel:GlobalMarketPanel,model=None,start_index:int|None=None,
                       end_index:int|None=None,stride:int=1)->dict:
        """Evaluate one policy with sequential positions and net-of-cost PnL."""
        model=model or self.champion; start=max(self.window-1,start_index or 0)
        end=len(panel.dates) if end_index is None else min(len(panel.dates),end_index)
        step_returns=[]; rows=[]; action_counts={name:0 for name in ACTION_NAMES}
        gross_total=fee_total=slippage_total=net_total=0.0
        was_training=model.training; model.eval()
        try:
            with torch.inference_mode():
                for ti in range(start,end,stride):
                    contextual=getattr(model,"_stockrl_uses_market_context",False)
                    args=[x.to(self.device) for x in panel.window(ti,self.window,include_context=contextual)]
                    args[0]=args[0].to(dtype=next(model.parameters()).dtype)
                    logits,values=model(*args); actions=logits[0].argmax(-1).cpu().tolist()
                    values=values[0].float().cpu().tolist(); rewards=[]
                    for j,symbol in enumerate(panel.symbols):
                        if not panel.observed[ti,j]: continue
                        target=self._target_index(panel,ti,j)
                        if target is None or target>=end: continue
                        action=int(actions[j])
                        reward,fee_cost,slip_cost=net_action_reward(action,panel.return_to(ti,target,j),
                                                                     0,self.fee,self.slippage)
                        action_counts[ACTION_NAMES[action]]+=1
                        gross=float(action-1)*panel.return_to(ti,target,j)
                        gross_total+=gross; fee_total+=fee_cost; slippage_total+=slip_cost; net_total+=reward
                        rewards.append(reward)
                        rows.append({"date":str(panel.dates[ti]),"symbol":symbol,"action":ACTION_NAMES[action],
                            "gross_pnl_return":gross,"fee_return":fee_cost,"slippage_return":slip_cost,
                            "net_pnl_return":reward,"value":float(values[j])})
                    if rewards: step_returns.append((str(panel.dates[ti])[:10],float(np.mean(rewards))))
        finally:
            model.train(was_training)
        equity=1.0; peak=1.0; max_dd=0.0; daily={}
        for day,ret in step_returns:
            equity*=max(0.0,1+ret); peak=max(peak,equity); max_dd=max(max_dd,(peak-equity)/peak)
            daily[day]=daily.get(day,0.0)+ret
        return {"start_index":start,"end_index":end,"timestamps":len(step_returns),"decisions":len(rows),
            "net_return":net_total,"net_pnl_return_sum":net_total,"compounded_step_return":equity-1,
            "gross_pnl_return_sum":gross_total,
            "fee_return_sum":fee_total,"slippage_return_sum":slippage_total,"max_drawdown":max_dd,
            "action_counts":action_counts,"daily_net_return":daily,"fee_rate":self.fee,
            "slippage_bps":self.slippage*10000,"horizon":str(self.horizon)}

    def backtest_panel(self,panel:GlobalMarketPanel,stride:int=1)->dict:
        """Run the final 15% as a sealed, equal-weight paper backtest."""
        start=max(1,int(len(panel.dates)*.85))
        out=self.evaluate_panel(panel,self.champion,start_index=start,stride=stride)
        import pandas as pd
        # Recreate row-level scores only for the final report CSV.
        rows=[]; model=self.champion; was_training=model.training; model.eval()
        try:
            with torch.inference_mode():
                for ti in range(max(self.window-1,start),len(panel.dates),stride):
                    contextual=getattr(model,"_stockrl_uses_market_context",False)
                    args=[x.to(self.device) for x in panel.window(ti,self.window,include_context=contextual)]
                    args[0]=args[0].to(dtype=next(model.parameters()).dtype)
                    logits,values=model(*args); acts=logits[0].argmax(-1).cpu().tolist()
                    for j,symbol in enumerate(panel.symbols):
                        if not panel.observed[ti,j]: continue
                        target=self._target_index(panel,ti,j)
                        if target is None or target>=len(panel.dates): continue
                        action=int(acts[j])
                        reward,fee_cost,slip_cost=net_action_reward(action,panel.return_to(ti,target,j),0,self.fee,self.slippage)
                        rows.append({"date":str(panel.dates[ti]),"symbol":symbol,"action":ACTION_NAMES[action],
                            "gross_pnl_return":(action-1)*panel.return_to(ti,target,j),"fee_return":fee_cost,
                            "slippage_return":slip_cost,"net_pnl_return":reward,"value":float(values[0,j])})
        finally: model.train(was_training)
        pd.DataFrame(rows).to_csv(self.state_dir/"backtest.csv",index=False)
        self.metrics["backtest"]=out
        return out

    def import_teacher_csv(self, panel:GlobalMarketPanel, path:str|Path, symbol="MSFT", max_index:int|None=None,
                           source_name:str|None=None):
        """Import normalized SELL/HOLD/BUY labels as replayed imitation data."""
        import pandas as pd
        df=pd.read_csv(path)
        if not {"date","action"}.issubset(df.columns): raise ValueError("teacher CSV needs date,action columns")
        source_name=source_name or f"teacher:{Path(path).stem}"
        symbol_to_ix={s:i for i,s in enumerate(panel.symbols)}
        if symbol not in symbol_to_ix: raise ValueError(f"teacher symbol {symbol} is absent from panel")
        date_to_ix={pd.Timestamp(d):i for i,d in enumerate(panel.dates)}
        with self.replay.lock:
            existing={(e.source,e.timestamp,e.symbol_index,e.action) for e in self.replay.items}
        actions={"SELL":0,"HOLD":1,"WAIT":1,"BUY":2}
        added=0
        for row in df.itertuples(index=False):
            date=pd.Timestamp(row.date)
            if date.tzinfo is not None: date=date.tz_convert("UTC").tz_localize(None)
            ti=date_to_ix.get(date)
            if ti is None: ti=date_to_ix.get(date.normalize())
            row_symbol=str(getattr(row,"symbol",symbol)); j=symbol_to_ix.get(row_symbol)
            action=actions.get(str(row.action).upper())
            if ti is None or action is None or j is None or (max_index is not None and ti>=max_index): continue
            key=(source_name,str(date.normalize()),j,action)
            if key in existing: continue
            window=self._window(panel,ti); x,sid,mid,aid,mask=window[:5]
            self.replay.add(Experience(x[0].numpy().astype(np.float16),sid[0].numpy(),mid[0].numpy(),aid[0].numpy(),
                mask[0].numpy(),j,action,0.0,str(date),source_name,0.0,
                market_context=window[5][0].numpy().astype(np.float16) if len(window)>5 else None)); added+=1; existing.add(key)
        return added

    def _pack(self,batch):
        def tensor(key,dtype=None):
            a=np.stack([getattr(x,key) for x in batch]); t=torch.as_tensor(a,device=self.device)
            return t.to(dtype) if dtype else t
        feature_dtype=torch.float16 if self.device.type=="cuda" else torch.float32
        args=(tensor("features",feature_dtype),tensor("symbol_ids",torch.long),tensor("market_ids",torch.long),
              tensor("asset_ids",torch.long),tensor("valid_mask",torch.bool))
        if getattr(self.champion,"_stockrl_uses_market_context",False):
            from .market_training import CONTEXT_FEATURES
            rows=[]
            for experience in batch:
                context=getattr(experience,"market_context",None)
                if context is None:
                    context=np.zeros((self.window,len(CONTEXT_FEATURES)),dtype=np.float16)
                rows.append(context)
            args=args+(torch.as_tensor(np.stack(rows),device=self.device,dtype=torch.float32),)
        p_rows=[]; a_rows=[]
        for experience in batch:
            p_rows.append(experience.portfolio_state if experience.portfolio_state is not None
                          else np.zeros((experience.features.shape[1], 8), dtype=np.float16))
            a_rows.append(experience.account_state if experience.account_state is not None
                          else np.zeros(8, dtype=np.float32))
        return args, torch.as_tensor(np.stack(p_rows),device=self.device,dtype=torch.float32), torch.as_tensor(np.stack(a_rows),device=self.device,dtype=torch.float32)

    def _learner(self):
        while not self.stop.wait(.1):
            validation_dates=len({item[7] for item in self.validation})
            # Do not spend GPU time on legacy isolated-action samples. The
            # live candidate is promoted only from actual paper-account
            # transitions whose costs and allocation were executed in the
            # virtual ledger.
            if len(self.paper_experiences) < self.min_replay:
                continue
            if (self.replay.trainable_count()<self.min_replay or validation_dates<self.min_validation_dates or self.candidate is not None
                    or len(self.replay)-self.last_train_replay_size<max(self.min_replay,self.candidate_interval)): continue
            self.last_train_replay_size=len(self.replay)
            try:
                self._train_candidate()
            except Exception as exc:
                self.metrics["candidate_training"]=False; self.candidate=None
                self.metrics["candidate_errors"]=int(self.metrics.get("candidate_errors",0))+1
                self.metrics["last_candidate_error"]=f"{type(exc).__name__}: {exc}"
                try: self._write_metrics()
                except Exception: pass

    def _train_candidate(self):
        from datetime import datetime, timezone
        self.metrics["candidate_training"]=True
        with self.lock: source_champion=self.champion
        if getattr(source_champion,"_stockrl_uses_market_context",False):
            from .market_training import ContextConditionedTransformer
            backbone=GlobalMarketTransformer(self.cfg)
            if self.device.type=="cuda": backbone=backbone.half()
            self.candidate=ContextConditionedTransformer(backbone)
            self.candidate.context_policy.float(); self.candidate.context_value.float()
            self.candidate._stockrl_uses_market_context=True
            self.candidate._stockrl_symbol_map=source_champion._stockrl_symbol_map
        else:
            self.candidate=GlobalMarketTransformer(self.cfg)
            if self.device.type=="cuda": self.candidate=self.candidate.half()
        self.candidate=self.candidate.to(self.device)
        # Champion is read-only for the duration of this copy; never hold the
        # inference lock during a 0.5B state transfer.
        self.candidate.load_state_dict(source_champion.state_dict())
        candidate=self.candidate.train(); opt=torch.optim.AdamW(candidate.parameters(),lr=self.lr,weight_decay=.01,
            eps=1e-4 if self.device.type=="cuda" else 1e-8)
        original=self.champion
        elapsed=[]
        for update_ix in range(self.updates_per_candidate):
            batch=self.replay.sample(self.batch_size)
            if not batch: break
            self.metrics["teacher_examples_trained"]+=sum(e.source.startswith("teacher") for e in batch)
            self.metrics["paper_examples_trained"]+=sum(not e.source.startswith("teacher") for e in batch)
            args,pstate,astate=self._pack(batch); ti=time.perf_counter()
            use_portfolio=all(e.portfolio_state is not None and e.account_state is not None for e in batch) \
                          and getattr(candidate,"_stockrl_uses_market_context",False)
            if use_portfolio:
                logits,values,allocations=candidate(*args,portfolio_state=pstate,
                                                     account_state=astate,return_allocation=True)
            else:
                logits,values=candidate(*args); allocations=None
            if self.device.type=="cuda": torch.cuda.synchronize(self.device)
            rows=torch.arange(len(batch),device=self.device)
            act=torch.as_tensor([e.action for e in batch],device=self.device,dtype=torch.long)
            reward=torch.as_tensor([e.portfolio_reward if e.portfolio_reward is not None else e.reward
                                    for e in batch],device=self.device,dtype=torch.float32)
            chosen=logits[rows,torch.as_tensor([e.symbol_index for e in batch],device=self.device)].float()
            predicted=values[rows,torch.as_tensor([e.symbol_index for e in batch],device=self.device)].float()
            if not torch.isfinite(chosen).all() or not torch.isfinite(predicted).all():
                self.metrics["nonfinite_updates"]+=1
                continue
            dist=Categorical(logits=chosen); advantage=(reward-predicted.detach()).clamp(-1,1)
            teacher=torch.as_tensor([e.source.startswith("teacher") for e in batch],device=self.device,dtype=torch.bool)
            pg=-(dist.log_prob(act)*advantage).mean()+.5*nn.functional.smooth_l1_loss(predicted,reward)-.005*dist.entropy().mean()
            imitation=nn.functional.cross_entropy(chosen[teacher],act[teacher]) if teacher.any() else pg*0
            allocation_loss=pg*0
            if allocations is not None:
                alloc_selected=allocations[rows,torch.as_tensor([e.symbol_index for e in batch],device=self.device)]
                allocation_loss=-(reward.detach().clamp(-1,1)*torch.log(alloc_selected.clamp_min(1e-7))).mean()
            loss=pg+imitation+0.10*allocation_loss
            if not torch.isfinite(loss):
                self.metrics["nonfinite_updates"]+=1
                continue
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(candidate.parameters(),1.0); opt.step()
            self.metrics["update_losses"].append(float(loss.detach().cpu()))
            if self.device.type=="cuda": torch.cuda.synchronize(self.device)
            elapsed.append(time.perf_counter()-ti); self.steps+=1
        self.metrics["update_seconds"].extend(elapsed); self.metrics["updates"]+=len(elapsed)
        self.metrics["teacher_mix_probability"]=self.replay.teacher_fraction()
        if not all(torch.isfinite(p).all() for p in candidate.parameters()):
            self.metrics["nonfinite_updates"]+=1
            self.metrics["rejections"]+=1
            self.metrics["last_rejection_utc"]=datetime.now(timezone.utc).isoformat()
            history=self.metrics.setdefault("candidate_gate_history",[])
            history.append({"time_utc":self.metrics["last_rejection_utc"],"applied":False,
                            "reason":"가중치에 계산 불가능한 값이 발생해 적용하지 않음",
                            "candidate_score":None,"champion_score":None})
            self.metrics["candidate_gate_history"]=history[-20:]
            self.metrics["candidate_training"]=False
            self.candidate=None
            self._write_metrics()
            del candidate,opt
            return
        delta=sum((candidate.state_dict()[k].float()-v.detach().float()).abs().sum().item()
                  for k,v in original.state_dict().items())
        self.metrics["weight_delta_l1"].append(delta)
        self.metrics["last_update_utc"]=datetime.now(timezone.utc).isoformat()
        # Candidate promotion on a deterministic, held-out chronological tail.
        # It must beat champion in a net reward check before replacing it.
        score_new=self._score_candidate(candidate); score_old=self._score_candidate(self.champion)
        self.state_dir.mkdir(exist_ok=True,parents=True)
        # Candidate optimizer state is intentionally ephemeral across runs;
        # omit it from the artifact so evaluation/recovery only loads weights.
        save_model(self.state_dir/"candidate.pt",candidate,self.cfg,step=self.steps)
        self._commit_candidate(candidate,score_new,score_old)
        self._write_metrics()
        del candidate,opt; self.candidate=None; self.metrics["candidate_training"]=False

    def _commit_candidate(self,candidate,score_new:float,score_old:float)->bool:
        """Stage a valid champion checkpoint, then atomically swap the reader."""
        promote=should_promote(score_new,score_old,1e-9)
        self.metrics["last_candidate_validation_score"]=score_new
        self.metrics["last_champion_validation_score"]=score_old
        self.metrics["last_candidate_promoted"]=promote
        if promote:
            staged=self.state_dir/"champion.next.pt"
            save_model(staged,candidate,self.cfg,step=self.steps)
            if self.champion_path.exists(): shutil.copy2(self.champion_path,self.backup_path)
            with self.lock:
                os.replace(staged,self.champion_path)
                self.champion=candidate.eval(); self.metrics["promotions"]+=1
                from datetime import datetime, timezone
                self.metrics["last_promotion_utc"]=datetime.now(timezone.utc).isoformat()
        else:
            self.metrics["rejections"]+=1
            from datetime import datetime, timezone
            self.metrics["last_rejection_utc"]=datetime.now(timezone.utc).isoformat()
        history=self.metrics.setdefault("candidate_gate_history",[])
        history.append({"time_utc":self.metrics["last_promotion_utc" if promote else "last_rejection_utc"],
                        "applied":promote,"reason":("새 모델의 동일 구간 순보상이 더 높아 교체" if promote
                                                    else "새 모델의 동일 구간 순보상이 더 높지 않아 기존 모델 유지"),
                        "candidate_score":float(score_new),"champion_score":float(score_old)})
        self.metrics["candidate_gate_history"]=history[-20:]
        return promote

    def _score_candidate(self,model):
        """Net return over identical chronological holdout dates; held-out rows never train."""
        if len(self.portfolio_validation) < 8:
            # A candidate trained from paper-account rewards may not replace
            # the incumbent on the old isolated-action metric. Wait for a
            # genuinely unseen paper-account validation slice.
            return float("-inf")
        if getattr(model,"_stockrl_uses_market_context",False):
            # Portfolio gate: replay the same validation decisions using the
            # model's cash-inclusive allocation and the known future bar
            # return. This is the promotion metric, rather than weight delta.
            total=0.0; was_training=model.training; model.eval()
            try:
                groups={}
                for exp in self.portfolio_validation:
                    groups.setdefault(exp.timestamp,[]).append(exp)
                with torch.inference_mode():
                    for _, rows_exp in sorted(groups.items()):
                        exp0=rows_exp[0]
                        batch=[exp0]
                        args,pstate,astate=self._pack(batch)
                        logits,_,alloc=model(*args,portfolio_state=pstate,account_state=astate,return_allocation=True)
                        for exp in rows_exp:
                            j=int(exp.symbol_index)
                            if exp.forward_return is None: continue
                            action=int(logits[0,j].argmax())
                            weight=float(alloc[0,j].float())
                            turnover=2.0 if action != 1 else 0.0
                            total += weight * float(action-1) * float(exp.forward_return)
                            total -= turnover * (self.fee+self.slippage) * max(weight,0.0)
            finally:
                model.train(was_training)
            return float(total)
        if len(self.validation)<2: return float("-inf")
        groups={}
        for item in self.validation: groups.setdefault(item[7],[]).append(item)
        dates=sorted(groups); total_net=0.0; was_training=model.training; model.eval()
        try:
            with torch.inference_mode():
                for date in dates:
                    batch=groups[date]; feats,sids,mids,aids,masks=batch[0][:5]
                    x=torch.as_tensor(feats[None],device=self.device,dtype=next(model.parameters()).dtype)
                    sid=torch.as_tensor(sids[None],device=self.device,dtype=torch.long)
                    mid=torch.as_tensor(mids[None],device=self.device,dtype=torch.long)
                    aid=torch.as_tensor(aids[None],device=self.device,dtype=torch.long)
                    mask=torch.as_tensor(masks[None],device=self.device,dtype=torch.bool)
                    if getattr(model,"_stockrl_uses_market_context",False):
                        context=batch[0][9] if len(batch[0])>9 else None
                        if context is None: context=np.zeros((self.window,16),np.float16)
                        ctx=torch.as_tensor(context[None],device=self.device,dtype=torch.float32)
                        logits,_=model(x,sid,mid,aid,mask,ctx)
                    else:
                        logits,_=model(x,sid,mid,aid,mask)
                    rewards=[]
                    for item in batch:
                        j,gross=item[5],item[6]; action=int(logits[0,j].argmax())
                        reward,_,_=net_action_reward(action,gross,0,self.fee,self.slippage)
                        rewards.append(reward)
                    total_net+=float(sum(rewards))
        finally: model.train(was_training)
        return total_net

    def _write_metrics(self):
        proc=psutil.Process(); metrics=dict(self.metrics)
        metrics.update({"parameters":parameter_count(self.champion),"device":str(self.device),
          "horizon":str(self.horizon),"fee_rate":self.fee,"slippage_bps":self.slippage*10000,
          "rss_bytes":proc.memory_info().rss,"replay_count":len(self.replay),
          "trainable_replay_count":self.replay.trainable_count(),
          "replay_file_bytes":self.replay_path.stat().st_size if self.replay_path.exists() else 0,
          "champion_path":str(self.champion_path),"candidate_path":str(self.state_dir/"candidate.pt")})
        if self.device.type=="cuda":
            metrics.update({"cuda_device":torch.cuda.get_device_name(self.device),
                "cuda_memory_allocated_bytes":torch.cuda.memory_allocated(self.device),
                "cuda_memory_reserved_bytes":torch.cuda.memory_reserved(self.device),
                "cuda_peak_allocated_bytes":max(int(metrics.get("cuda_peak_allocated_bytes",0)),
                                                  int(torch.cuda.max_memory_allocated(self.device))),
                "cuda_total_memory_bytes":torch.cuda.get_device_properties(self.device).total_memory})
        for key in ("inference_seconds","update_seconds"):
            a=metrics[key]; metrics[key+"_p50"]=float(np.percentile(a,50)) if a else 0
            metrics[key+"_p95"]=float(np.percentile(a,95)) if a else 0
        _atomic_json(metrics,self.state_dir/"metrics.json")

    def checkpoint(self):
        with self.lock: save_model(self.champion_path,self.champion,self.cfg,step=self.steps)
        try:
            self.replay.save(self.replay_path)
        except (MemoryError, RuntimeError, OSError) as exc:
            self.metrics["replay_save_error"] = f"{type(exc).__name__}: {exc}"
        self._write_metrics()


def benchmark_model(model, panel, window=128, repeats=3, device=None):
    from .core import device_for
    dev=device or device_for(); model.to(dev)
    if dev.type=="cuda": model.half()
    model.eval(); contextual=getattr(model,"_stockrl_uses_market_context",False)
    args=[x.to(dev) for x in panel.window(min(len(panel.dates)-1,window),window,include_context=contextual)]
    args[0]=args[0].to(dtype=next(model.parameters()).dtype)
    samples=[]
    with torch.inference_mode():
        model(*args)
        if dev.type=="cuda": torch.cuda.synchronize(dev)
        for _ in range(repeats):
            if dev.type=="cuda": torch.cuda.synchronize(dev)
            t=time.perf_counter(); model(*args); samples.append(time.perf_counter()-t)
            if dev.type=="cuda":
                torch.cuda.synchronize(dev); samples[-1]=time.perf_counter()-t
    return {"inference_seconds_p50":float(np.percentile(samples,50)),"inference_seconds_p95":float(np.percentile(samples,95)),
            "inference_repeats":repeats}
