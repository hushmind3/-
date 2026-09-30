"""Asynchronous online actor-critic over global asset panels.

Market observation, candidate training, and frozen-snapshot validation use
separate model state. Only a future paper-account score improvement swaps the champion.
"""
from __future__ import annotations

from collections import deque, OrderedDict
from contextlib import closing, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import json, os, random, shutil, threading, time, pickle, sqlite3, zlib, queue, io, gc, math
import re
import hashlib

import numpy as np
import psutil
import torch
from torch import nn
from torch.distributions import Categorical

from .global_transformer import (ACTION_NAMES, GlobalMarketPanel, GlobalMarketTransformer,
                                 TransformerConfig, parameter_count, load_compatible_state_dict)
from .paper_account import PaperAccount, _currency
from .state_io import atomic_json
from .multiscale import (MULTISCALE_FEATURE_COUNT, TIMEFRAME_NAMES,
                         TIMEFRAME_FEATURE_NAMES)

REWARD_VERSION="symbol_and_portfolio_v5"
PAPER_EXPLORATION_EPSILON=0.05
ONLINE_TRAINABLE_BLOCKS=4
REWARD_DEFINITION=("symbol_and_portfolio_v5: online RL uses executed cash-only paper-account outcomes; "
                   "trade reward horizon starts after the linked next-bar fill; "
                   "per-symbol net PnL and normalized whole-account net equity change are kept separate")


class IncrementalMarketCSV:
    """Read an append-only feed once, then parse only complete appended rows."""
    def __init__(self, path: str | Path, retain_timestamps: int = 4096):
        self.path=Path(path); self.retain_timestamps=retain_timestamps
        self.frame=None; self.offset=0; self.partial=b""; self.header=b""; self.fingerprint=None
        self.processed_through=None

    def _fingerprint(self):
        stat=self.path.stat()
        with self.path.open("rb") as stream:
            prefix=stream.read(2048)
        return stat.st_size,stat.st_mtime_ns,getattr(stat,"st_ino",0),hashlib.sha256(prefix).digest()

    def _trim(self, frame):
        import pandas as pd
        if frame.empty or "date" not in frame or "symbol" not in frame:
            return frame
        stamps=pd.to_datetime(frame["date"],utc=True,errors="coerce")
        unique=stamps.dropna().drop_duplicates().sort_values()
        if len(unique)<=self.retain_timestamps:
            return frame
        cutoff=unique.iloc[-self.retain_timestamps]
        # Keep all unprocessed bars, plus the model's input history. A restart
        # or a slow observer must not silently discard the next rolling windows.
        if self.processed_through is None:
            return frame
        processed=pd.Timestamp(self.processed_through)
        if processed.tzinfo is None:
            processed=processed.tz_localize("UTC")
        else:
            processed=processed.tz_convert("UTC")
        prior=unique.loc[unique<=processed]
        if len(prior):
            cutoff=min(cutoff,prior.iloc[-min(128,len(prior))])
        recent=frame.loc[stamps>=cutoff]
        reference=~frame.get("asset_class",pd.Series("equity",index=frame.index)).astype(str).str.casefold().eq("equity")
        carry=frame.loc[(stamps<cutoff)&reference].groupby("symbol",sort=False).tail(20)
        return pd.concat((carry,recent),ignore_index=True).sort_values(
            ["date","symbol"],kind="stable").reset_index(drop=True)

    def refresh(self):
        import pandas as pd
        signature=self._fingerprint()
        reset=(self.frame is None or signature[0]<self.offset or
               (self.fingerprint is not None and signature[2:]!=self.fingerprint[2:]))
        if reset:
            data=self.path.read_bytes()
            end=data.rfind(b"\n")+1
            complete=data[:end]
            self.partial=data[end:]
            self.header=complete.splitlines(keepends=True)[0] if complete else b""
            self.frame=pd.read_csv(io.BytesIO(complete)) if complete else pd.DataFrame()
            self.offset=len(data)
            self.frame=self._trim(self.frame)
        elif signature[0]>self.offset:
            with self.path.open("rb") as stream:
                stream.seek(self.offset); appended=stream.read()
            self.offset+=len(appended)
            combined=self.partial+appended
            end=combined.rfind(b"\n")+1
            complete=combined[:end]; self.partial=combined[end:]
            if complete:
                fresh=pd.read_csv(io.BytesIO(self.header+complete))
                self.frame=pd.concat((self.frame,fresh),ignore_index=True)
                self.frame=self.frame.drop_duplicates(["date","symbol"],keep="last")
                self.frame=self._trim(self.frame)
        self.fingerprint=signature
        return self.frame,signature


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
    reward_version: str = "symbol_and_portfolio_v4"
    market_context: np.ndarray | None = None
    multiscale_state: np.ndarray | None = None
    portfolio_state: np.ndarray | None = None
    account_state: np.ndarray | None = None
    portfolio_reward: float | None = None
    portfolio_transition: bool = False
    portfolio_value_transition: bool = False
    forward_return: float | None = None
    behavior_log_prob: float | None = None
    trade_executed: bool = True


from .replay_store import GlobalReplayBuffer, ReplayStorageFull


def _atomic_save(obj, path: Path, temp_dir: Path | None = None):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp_dir=Path(temp_dir) if temp_dir is not None else path.parent
    temp_dir.mkdir(parents=True,exist_ok=True)
    tmp=temp_dir/(path.name+".tmp")
    try:
        torch.save(obj,tmp)
        # Replay may be acknowledged only after its checkpoint reaches disk.
        with tmp.open("rb+") as stream:
            stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp,path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_json(obj,path:Path):
    atomic_json(obj,path,default=lambda value: value.item() if isinstance(value,np.generic) else str(value))


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


def save_model(path:Path, model:GlobalMarketTransformer, cfg:TransformerConfig, optimizer=None, step=0,
               temp_dir:Path|None=None, replay_commit:dict|None=None):
    payload={"state_dict":model.state_dict(),"config":asdict(cfg),"step":step,
             "optimizer":optimizer.state_dict() if optimizer is not None else None}
    if replay_commit is not None:
        payload["replay_commit"]=replay_commit
    if getattr(model,"_stockrl_uses_market_context",False):
        from .market_training import CONTEXT_FEATURES
        payload["context_features"]=list(CONTEXT_FEATURES)
        payload["symbol_map"]=model._stockrl_symbol_map
        payload["market_context_model"]=True
        payload["multiscale_feature_order"]=[f"{scale}:{feature}"
            for scale in TIMEFRAME_NAMES for feature in TIMEFRAME_FEATURE_NAMES]
    _atomic_save(payload,path,temp_dir=temp_dir)


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
        multiscale_order=[f"{scale}:{feature}" for scale in TIMEFRAME_NAMES
                          for feature in TIMEFRAME_FEATURE_NAMES]
        if ckpt.get("multiscale_feature_order",multiscale_order)!=multiscale_order:
            raise ValueError("checkpoint multiscale feature order does not match the inference schema")
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
    model._stockrl_replay_commit=ckpt.get("replay_commit",{})
    del ckpt
    model.to(device).eval()
    return model,cfg


class MarketObservation:
    """Bounded immutable market input for asynchronous portfolio validation."""
    window=GlobalMarketPanel.window

    def __init__(self,panel,index,length):
        start=max(0,index-length+1)
        self.dates=panel.dates[start:index+1].copy()
        self.symbols=list(panel.symbols)
        self.groups=dict(panel.groups)
        self.features=panel.features[start:index+1].copy()
        self.observed=panel.observed[start:index+1].copy()
        self.ever_observed=panel.observed[:index+1].any(axis=0)
        self.closes=panel.closes[start:index+1].copy()
        self.symbol_ids=panel.symbol_ids.copy()
        self.market_ids=panel.market_ids.copy()
        self.asset_ids=panel.asset_ids.copy()
        self.market_context=(panel.market_context[start:index+1].copy()
                             if panel.market_context is not None else None)
        self.multiscale=panel.multiscale_at(index).copy()

    def multiscale_at(self,index):
        if index!=len(self.dates)-1:
            raise ValueError("observation has only its captured multiscale state")
        return self.multiscale


class OnlineGlobalAgent:
    def __init__(self, state_dir: str|Path, device="auto", config:TransformerConfig|None=None,
                 capacity=4_096, window=128, horizon=1, fee=.001, slippage_bps=1.0,
                 min_replay=8, batch_size=4, updates_per_candidate=8, lr=2e-6, seed=7,
                 candidate_interval=16, initial_champion: str|Path|None=None,
                 teacher_replay_path: str|Path|None=None,
                 model_dir: str|Path|None=None):
        from .core import device_for
        from .paths import ensure_project_path, validate_model_dir
        self.state_dir=ensure_project_path(state_dir, "runtime")
        self.model_dir=validate_model_dir(model_dir)
        if initial_champion is not None:
            source = Path(initial_champion).expanduser().resolve()
            if source.parent != self.model_dir or source.name not in {"champion.pt", "candidate.pt"}:
                raise ValueError(f"initial champion must be an existing checkpoint in {self.model_dir}")
            initial_champion = source
        self.state_dir.mkdir(parents=True,exist_ok=True)
        self.model_dir.mkdir(parents=True,exist_ok=True)
        self.device=device_for(device); self.cfg=config or TransformerConfig(); self.window=window
        self.horizon=horizon; self.horizon_kind,self.horizon_amount=parse_horizon(horizon)
        self.max_pending_age_seconds=30*86400
        self.fee=fee; self.slippage=slippage_bps/10_000
        self.min_replay=min_replay; self.batch_size=batch_size; self.updates_per_candidate=updates_per_candidate
        # Keep the live learner responsive while training from a real replay
        # minibatch; UI settings may request a shorter interval.
        self.candidate_interval=max(1,min(int(candidate_interval),512))
        self.min_validation_dates=0
        self.lr=lr; self.seed=seed; self.lock=threading.RLock(); self.stop=threading.Event()
        self.replay=GlobalReplayBuffer(capacity,seed,self.state_dir/"replay.sqlite3")
        # Replay is a durable FIFO SQLite journal; checkpoints
        # remain the only model files. The optional teacher source is read-only.
        if teacher_replay_path: self.replay.load(Path(teacher_replay_path))
        self.champion_path=self.model_dir/"champion.pt"
        if initial_champion is not None and not Path(initial_champion).is_file():
            raise FileNotFoundError(f"configured champion checkpoint does not exist: {initial_champion}")
        if not self.champion_path.exists() and initial_champion:
            # Seed this isolated runtime from a prior verified champion without
            # writing to or replacing the source checkpoint.
            shutil.copy2(initial_champion,self.champion_path)
        if self.champion_path.exists():
            self.champion,self.cfg=load_model(self.champion_path,self.device)
        else:
            if model_dir is not None or initial_champion is not None:
                raise FileNotFoundError(f"configured champion checkpoint does not exist: {self.champion_path}")
            self.champion=GlobalMarketTransformer(self.cfg)
            if self.device.type=="cuda": self.champion=self.champion.half()
            self.champion=self.champion.to(self.device).eval()
            save_model(self.champion_path,self.champion,self.cfg,temp_dir=self.state_dir)
        self.candidate=None; self.validation_candidate=None; self.optimizer=None; self.steps=0; self.updates=0
        self.candidate_model_lock=threading.RLock()
        self.candidate_live_inference_lock=threading.Lock()
        self.candidate_live_model_lock=threading.Lock()
        self.candidate_live_model=None
        self.candidate_live_model_version=None
        self.candidate_live_queue=queue.Queue(maxsize=2)
        self.candidate_version=0
        self.last_validated_candidate_version=-1
        self.validation_restart_needs_fresh_trial=False
        self.candidate_trained_replay_row_ids=set()
        self.candidate_replay_passes=2
        self.candidate_replay_uses={}
        self.candidate_lineage_path=self.state_dir/"candidate_lineage.json"
        self.candidate_retry_attempts=0; self.candidate_retry_after=0.0
        self.policy_rng=random.Random(seed+1)
        self.validation_policy_rng=random.Random(seed+2)
        self.validation=[]
        self.portfolio_validation=[]
        self.validation_lock=threading.Lock()
        self.validation_window_dates=64
        self.validation_dates=deque()
        self.portfolio_validation_dates=deque()
        self.positions={}
        self.paper_account=PaperAccount(self.state_dir/"paper_account.json",self.fee,self.slippage)
        if not self.paper_account.path.exists(): self.paper_account.save()
        self.candidate_live_account=PaperAccount(
            self.state_dir/"candidate_observer_account.json",self.fee,self.slippage)
        self.candidate_live_state_path=self.state_dir/"candidate_observer_state.json"
        self.candidate_live_thread=threading.Thread(
            target=self._candidate_live_worker,name="candidate-live-observer",daemon=True)
        self.validation_window_bars=128
        self.validation_queue=queue.Queue(maxsize=32)
        self.validation_last_enqueued_minute=None
        self.validation_generation=0
        self.validation_queue_invalid_reason=None
        self.validation_queue_delay_limit_seconds=300.0
        self.validation_queue_thread=threading.Thread(
            target=self._validation_worker,name="candidate-validation",daemon=True)
        self.validation_state_path=self.state_dir/"candidate_validation.json"
        self.validation_champion_account=PaperAccount(
            self.state_dir/"candidate_validation_champion.json",self.fee,self.slippage)
        self.validation_candidate_account=PaperAccount(
            self.state_dir/"candidate_validation_candidate.json",self.fee,self.slippage)
        self.validation_start_after=None
        self.validation_source_sha256=None
        self.validation_bars=0
        self.validation_active=False
        self.current_market_timestamp=None
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
                      "paper_experiences_seen":0,"paper_experiences_since_candidate":0,
                      "paper_net_reward":0.0,"last_market_timestamp":None,"candidate_training":False,
                      "agent_health":"starting","agent_input_errors":0,
                      "agent_last_input_error":None,"agent_last_input_error_utc":None,
                      "observation_caught_up":False,"unmatched_live_symbols":[],
                      "unmatched_live_symbol_count":0,
                      "last_update_utc":None,"last_promotion_utc":None,"last_rejection_utc":None,
                      "inference_during_candidate":0,"reward_definition":REWARD_DEFINITION,
                      "candidate_live_inference_count":0,
                      "candidate_live_inference_seconds_total":0.0,
                      "candidate_live_queue_drops":0,
                      "candidate_live_errors":0,
                      "candidate_live_status":"waiting_for_candidate",
                      "candidate_live_last_timestamp":self.candidate_live_account.state.get("last_timestamp"),
                      "champion_live_inference_count":0,
                      "champion_live_inference_seconds_total":0.0,
                      "champion_validation_inference_count":0,
                      "champion_validation_inference_seconds_total":0.0,
                      "candidate_validation_inference_count":0,
                      "candidate_validation_inference_seconds_total":0.0,
                      "candidate_replay_rows_held_for_validation":0,
                      "candidate_replay_rows_consumed":0,
                      "candidate_replay_rows_audit_status":"tracked",
                      "candidate_replay_cleanup_pending":False,
                      "update_losses":[],"inference_seconds":[],
                      "update_seconds":[],"weight_delta_l1":[]}
        metrics_path=self.state_dir/"metrics.json"
        if metrics_path.exists():
            try:
                self.metrics.update(json.loads(metrics_path.read_text(encoding="utf-8")))
            except (OSError,json.JSONDecodeError):
                pass
        # Preserve incompatible historical rows for inspection, never erase an
        # unlearned queue on restart. Current observations use the current schema.
        self.metrics["legacy_reward_examples_ignored"]=self.replay.stats()["unsupported"]
        prior_reward_definition=self.metrics.get("reward_definition")
        if prior_reward_definition!=REWARD_DEFINITION:
            # Counts from the previous account-wide reward are not eligible to
            # open a training cycle for the new symbol-level reward schema.
            self.metrics["paper_experiences_since_candidate"]=0
            self.metrics["reward_schema_reset_from"]=prior_reward_definition
        self.metrics["reward_definition"]=REWARD_DEFINITION
        self.metrics.pop("protected_champion_sha256",None)
        self.metrics["runtime_code_version"]="rolling-complete-learning-20261001"
        self.metrics["agent_session_started_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
        for key in ("update_losses","update_seconds","weight_delta_l1"):
            self.metrics[key]=list(self.metrics.get(key,[]))[-2000:]
        self.promotion_baseline_path=self.state_dir/"promotion_baseline.json"
        current_champion_sha=self._sha256_file(self.champion_path).upper()
        # SHA identifies the champion captured for a validation window; it is
        # not a fixed allowlist. Clear obsolete fixed-SHA hold messages.
        self.promotion_baseline_sha256=current_champion_sha
        if str(self.metrics.get("promotion_blocked_reason", "")).startswith(
                ("champion lineage unresolved", "promotion held: champion lineage")):
            self.metrics["promotion_blocked_reason"]=None
            self.metrics["candidate_validation_error"]=None
        self.metrics.setdefault("paper_experiences_seen",
                                int(self.metrics.get("paper_examples_trained",0)))
        self.replay.paper_outcomes_seen=int(self.metrics["paper_experiences_seen"])
        self.metrics["replay_persistence"]="durable_fifo_shared_frames_sqlite"
        self.metrics["portfolio_experiences"] = int(self.metrics.get("portfolio_experiences",0))
        self.updates=int(self.metrics.get("updates",0)); self.steps=self.updates
        # Unfinished replay is restored without pruning after restart.
        self.metrics["candidate_training"]=False
        self.metrics["candidate_optimizer_steps_current"]=0
        self.metrics["candidate_samples_current"]=0
        self.metrics["last_candidate_error"]=None
        validation_state={}
        try:
            validation_state=json.loads(self.validation_state_path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError):
            pass
        # Training consumption is independent of the validation result. The
        # candidate checkpoint itself records the committed replay use counts.
        self.metrics["candidate_replay_rows_held_for_validation"]=0
        self.validation_meta_path=self.state_dir/"validation_config.json"
        saved_validation_config={}
        try:
            saved_validation_config=json.loads(self.validation_meta_path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError):
            pass
        if saved_validation_config.get("reward_version")!=REWARD_VERSION:
            if validation_state.get("status")=="collecting":
                validation_state={"status":"discarded","reason":"paper reward schema changed"}
                _atomic_json(validation_state,self.validation_state_path)
            # A candidate trained with the old, currency-bugged reward cannot
            # be carried into the new policy cycle.
            shutil.copy2(self.champion_path,self.model_dir/"candidate.pt")
        try:
            lineage=json.loads(self.candidate_lineage_path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError):
            lineage={}
        self.candidate_version=max(0,int(lineage.get("candidate_version",self.metrics.get("updates",0))))
        self.candidate_trained_replay_row_ids=set(
            int(row_id) for row_id in lineage.get("trained_replay_row_ids",[]))
        self.candidate_replay_uses={int(key):int(value) for key,value in lineage.get("training_uses",{}).items()}
        for row_id in self.candidate_trained_replay_row_ids:
            self.candidate_replay_uses.setdefault(row_id,1)
        if validation_state.get("status")=="promoted":
            finalized=set(int(row_id) for row_id in validation_state.get("trained_replay_row_ids",[]))
            self.candidate_trained_replay_row_ids.difference_update(finalized)
        candidate_path=self.model_dir/"candidate.pt"
        if candidate_path.is_file():
            self.candidate,candidate_cfg=load_model(candidate_path,self.device)
            if asdict(candidate_cfg)!=asdict(self.cfg):
                raise ValueError("candidate checkpoint architecture does not match champion")
            if self.device.type=="cuda":
                self.candidate=self.candidate.to("cpu")
            replay_commit=getattr(self.candidate,"_stockrl_replay_commit",{})
            restored_uses=replay_commit.get("uses",self.candidate_replay_uses)
            self.replay.acknowledge_training(restored_uses,self.candidate_replay_passes)
            self.candidate_version=max(self.candidate_version,int(replay_commit.get("candidate_version",0)))
            self.candidate_replay_uses={}
            self.candidate_trained_replay_row_ids=set()
            self.metrics["candidate_checkpoint_loaded"]=True
            self.metrics["candidate_has_learning"]=(
                self._sha256_file(candidate_path)!=self._sha256_file(self.champion_path))
            observer=self._new_model_like(self.candidate,torch.device("cpu"))
            observer.load_state_dict(self.candidate.state_dict())
            observer.eval(); observer.requires_grad_(False)
            self.candidate_live_model=observer
            self.candidate_live_model_version=self.candidate_version
        if validation_state.get("status")=="collecting":
            self.validation_restart_needs_fresh_trial=True
        else:
            self.last_validated_candidate_version=int(
                validation_state.get("source_candidate_version",-1))
        if validation_state.get("status")=="collecting":
            # The trial model exists only in RAM. A restart invalidates that
            # frozen snapshot; keep candidate.pt and its replay untouched and
            # start a fresh trial from its latest checkpoint after loading feed.
            validation_state.update({"status":"discarded",
                "reason":"restart invalidated the in-memory validation snapshot",
                "discarded_bars":int(validation_state.get("bars",0))})
            _atomic_json(validation_state,self.validation_state_path)
            self.validation_champion_account.reset()
            self.validation_candidate_account.reset()
            self.validation_bars=0
            self.validation_active=False
            self.metrics["candidate_validation_bars"]=0
            self.metrics["candidate_validation_restart_discarded"]=True
        self._atomic_json({"version":1,"candidate_version":self.candidate_version,
            "trained_replay_row_ids":sorted(self.candidate_trained_replay_row_ids),
            "training_uses":self.candidate_replay_uses},
            self.candidate_lineage_path)
        self.metrics["candidate_skip_reason"]="새 paper 경험과 검증 시각을 기다리는 중"
        # Older runtime metrics may contain an obsolete held stage; resume the
        # normal candidate cycle without discarding replay or validation state.
        if self.metrics.get("candidate_stage") == "promotion_held":
            self.metrics["candidate_stage"] = "waiting"
        # New observations enter the same durable queue and trigger learning.
        self.last_train_replay_size=int(self.metrics.get("last_train_replay_size",0))
        _atomic_json({"reward_version":REWARD_VERSION,"horizon":str(self.horizon)},self.validation_meta_path)
        self.thread=threading.Thread(target=self._learner,name="global-learner",daemon=True)

    @staticmethod
    def _sha256_file(path:Path)->str:
        digest=hashlib.sha256()
        with Path(path).open("rb") as stream:
            while chunk:=stream.read(1024*1024): digest.update(chunk)
        return digest.hexdigest()

    def _append_validation(self, target:list, date_order:deque, timestamp:str, item) -> None:
        """Keep a fixed, recent validation window shared by both models."""
        with self.validation_lock:
            if not date_order or date_order[-1] != timestamp:
                date_order.append(timestamp)
                while len(date_order)>self.validation_window_dates:
                    expired=date_order.popleft()
                    if target is self.validation:
                        target[:]=[row for row in target if row[7]!=expired]
                    else:
                        target[:]=[row for row in target if row.timestamp!=expired]
            target.append(item)

    def _new_model_like(self, source, device):
        contextual=bool(getattr(source,"_stockrl_uses_market_context",False))
        source_dtype=next(source.parameters()).dtype
        if contextual:
            from .market_training import ContextConditionedTransformer
            backbone=GlobalMarketTransformer(self.cfg)
            if device.type=="cuda" or source_dtype==torch.float16: backbone=backbone.half()
            model=ContextConditionedTransformer(backbone)
            model.context_policy.float(); model.context_value.float()
            model._stockrl_uses_market_context=True
            model._stockrl_symbol_map=dict(source._stockrl_symbol_map)
        else:
            model=GlobalMarketTransformer(self.cfg)
            if device.type=="cuda" or source_dtype==torch.float16: model=model.half()
        return model.to(device)

    def _restore_candidate_checkpoint(self):
        path=self.model_dir/"candidate.pt"
        if not path.is_file():
            with self.lock:
                source=self.champion
                self.candidate=self._new_model_like(source,self.device)
                self.candidate.load_state_dict(source.state_dict())
        else:
            candidate,cfg=load_model(path,self.device)
            if asdict(cfg)!=asdict(self.cfg):
                raise ValueError("candidate checkpoint architecture does not match champion")
            self.candidate=candidate
            commit=getattr(candidate,"_stockrl_replay_commit",{})
            self.replay.acknowledge_training(commit.get("uses",{}),self.candidate_replay_passes)
            self.candidate_version=max(self.candidate_version,int(commit.get("candidate_version",0)))
        if self.device.type=="cuda": self.candidate.to("cpu")
        self.metrics["candidate_checkpoint_loaded"]=True

    def _target_index(self,panel:GlobalMarketPanel,start:int,symbol:int)->int|None:
        future=np.flatnonzero(panel.observed[start+1:,symbol])+start+1
        if not len(future): return None
        if self.horizon_kind=="bars":
            return int(future[self.horizon_amount-1]) if len(future)>=self.horizon_amount else None
        target=panel.dates[start]+np.timedelta64(self.horizon_amount,"s")
        eligible=future[panel.dates[future]>=target]
        return int(eligible[0]) if len(eligible) else None

    def start(self):
        if self.validation_queue_thread.ident is None: self.validation_queue_thread.start()
        if self.candidate_live_thread.ident is None: self.candidate_live_thread.start()
        if self.thread.ident is None: self.thread.start()
    def close(self):
        self.stop.set()
        if self.thread.is_alive(): self.thread.join(timeout=300)
        if self.validation_queue_thread.is_alive(): self.validation_queue_thread.join(timeout=300)
        if self.candidate_live_thread.is_alive(): self.candidate_live_thread.join(timeout=300)

    def _queue_candidate_live_observation(self,panel,index,paper_enabled,uniforms):
        item=(panel,int(index),bool(paper_enabled),tuple(float(x) for x in uniforms))
        try:
            self.candidate_live_queue.put_nowait(item)
        except queue.Full:
            try:
                self.candidate_live_queue.get_nowait()
                self.candidate_live_queue.task_done()
                self.metrics["candidate_live_queue_drops"]=(
                    int(self.metrics.get("candidate_live_queue_drops",0))+1)
            except queue.Empty:
                pass
            try: self.candidate_live_queue.put_nowait(item)
            except queue.Full: pass

    def _candidate_live_worker(self):
        """Run an independent observational paper account for current candidate weights."""
        while not self.stop.is_set():
            try:
                panel,index,paper_enabled,uniforms=self.candidate_live_queue.get(timeout=.25)
            except queue.Empty:
                continue
            try:
                stamp=str(panel.dates[index])
                latest=str(self.current_market_timestamp or stamp)
                # If a newer bar is already queued, skip stale decisions and
                # keep the observer account close to the live market.
                if stamp<latest and not self.candidate_live_queue.empty():
                    self.metrics["candidate_live_queue_drops"]=(
                        int(self.metrics.get("candidate_live_queue_drops",0))+1)
                    continue
                account=self.candidate_live_account
                account.process_bar(panel,index,paper_enabled)
                if self.candidate_live_model is None:
                    self.metrics["candidate_live_status"]="waiting_for_candidate_update"
                    account.save()
                    self._atomic_json({"status":self.metrics["candidate_live_status"],
                        "last_timestamp":stamp,"candidate_version":self.candidate_version,
                        "last_decisions":[]},self.candidate_live_state_path)
                    self.metrics["candidate_live_last_timestamp"]=stamp
                    continue
                pstate,astate=account.model_inputs(panel,index)
                window=self._window(panel,index)
                started=time.perf_counter()
                with self.candidate_live_model_lock:
                    model=self.candidate_live_model
                    if model is None: raise RuntimeError("candidate observer snapshot is not loaded")
                    originally_offloaded=(self.device.type=="cuda" and
                        next(model.parameters()).device.type=="cpu")
                    with self.candidate_live_inference_lock:
                        if originally_offloaded: model.to(self.device)
                        device=next(model.parameters()).device
                        was_training=model.training
                        model.eval()
                        try:
                            args=[value.to(device) for value in window]
                            args[0]=args[0].to(dtype=next(model.parameters()).dtype)
                            with torch.inference_mode():
                                if getattr(model,"_stockrl_uses_market_context",False):
                                    pt=torch.as_tensor(np.asarray(pstate,dtype=np.float32)[None],device=device)
                                    at=torch.as_tensor(np.asarray(astate,dtype=np.float32)[None],device=device)
                                    mt=torch.as_tensor(panel.multiscale_at(index)[None],device=device)
                                    logits,_,allocation=model(*args,portfolio_state=pt,
                                        account_state=at,return_allocation=True,
                                        multiscale_state=mt)
                                    allocation=allocation[0].float().cpu().numpy()
                                else:
                                    logits,_=model(*args); allocation=None
                            if device.type=="cuda": torch.cuda.synchronize(device)
                        finally:
                            model.train(was_training)
                            if originally_offloaded: model.to("cpu")
                probabilities,actions=self._paper_policy_actions(
                    logits[0].float().cpu().numpy(),pstate,uniforms)
                elapsed=time.perf_counter()-started
                account.queue_decisions(panel,index,probabilities,paper_enabled,
                    allocation=allocation,actions=actions)
                account.save()
                decisions=[]
                for symbol_index,(symbol,action) in enumerate(zip(panel.symbols,actions)):
                    if not panel.observed[index,symbol_index]: continue
                    market,asset=panel.groups[symbol]
                    if asset in ("equity","etf"):
                        decisions.append({"symbol":symbol,"action":ACTION_NAMES[action]})
                self.metrics["candidate_live_inference_count"]=(
                    int(self.metrics.get("candidate_live_inference_count",0))+1)
                self.metrics["candidate_live_inference_seconds_total"]=(
                    float(self.metrics.get("candidate_live_inference_seconds_total",0.0))+elapsed)
                self.metrics["candidate_live_status"]=("training_and_observing"
                    if self.metrics.get("candidate_training") else "observing")
                self.metrics["candidate_live_error"]=None
                self.metrics["candidate_live_last_timestamp"]=stamp
                self._atomic_json({"status":self.metrics["candidate_live_status"],
                    "last_timestamp":stamp,"candidate_version":self.candidate_live_model_version,
                    "policy_mode":"same_epsilon_sampling_and_random_draws_as_champion",
                    "candidate_training":bool(self.metrics.get("candidate_training")),
                    "last_inference_seconds":elapsed,"inference_count":self.metrics[
                        "candidate_live_inference_count"],"last_decisions":decisions},
                    self.candidate_live_state_path)
            except Exception as exc:
                self.metrics["candidate_live_errors"]=(
                    int(self.metrics.get("candidate_live_errors",0))+1)
                self.metrics["candidate_live_status"]="error"
                self.metrics["candidate_live_error"]=f"{type(exc).__name__}: {exc}"[:500]
                self._atomic_json({"status":"error","last_timestamp":stamp,
                    "candidate_version":self.candidate_live_model_version,
                    "error":self.metrics["candidate_live_error"],"last_decisions":[]},
                    self.candidate_live_state_path)
            finally:
                self.candidate_live_queue.task_done()

    def _infer(self,panel,index,portfolio_state=None,account_state=None):
        args=[x.to(self.device) for x in self._window(panel,index)]
        with self.lock:
            model=self.champion
        args[0]=args[0].to(dtype=next(model.parameters()).dtype)
        with torch.inference_mode():
            if self.device.type=="cuda": torch.cuda.synchronize(self.device)
            t=time.perf_counter()
            if getattr(model,"_stockrl_uses_market_context",False) and portfolio_state is not None:
                pstate=torch.as_tensor(np.asarray(portfolio_state,dtype=np.float32)[None],device=self.device)
                astate=torch.as_tensor(np.asarray(account_state,dtype=np.float32)[None],device=self.device)
                mstate=torch.as_tensor(panel.multiscale_at(index)[None],device=self.device)
                logits,values,allocation=model(*args,portfolio_state=pstate,
                                                account_state=astate,return_allocation=True,
                                                multiscale_state=mstate)
            else:
                logits,values=model(*args); allocation=None
            elapsed=time.perf_counter()-t
            if self.device.type=="cuda":
                torch.cuda.synchronize(self.device); elapsed=time.perf_counter()-t
        self.metrics["inference_seconds"].append(elapsed)
        self.metrics["champion_live_inference_count"]=(
            int(self.metrics.get("champion_live_inference_count",0))+1)
        self.metrics["champion_live_inference_seconds_total"]=(
            float(self.metrics.get("champion_live_inference_seconds_total",0.0))+elapsed)
        if self.metrics.get("candidate_training"):
            self.metrics["inference_during_candidate"]=int(self.metrics.get("inference_during_candidate",0))+1
        if len(self.metrics["inference_seconds"])>2000: self.metrics["inference_seconds"]=self.metrics["inference_seconds"][-2000:]
        return (logits[0].cpu().numpy(),values[0].cpu().numpy(),
                allocation[0].cpu().float().numpy() if allocation is not None else None)

    @staticmethod
    def _sample_actions(probabilities,uniforms):
        """Sample normalized SELL/HOLD/BUY rows with caller-supplied common noise."""
        probabilities=np.asarray(probabilities,dtype=np.float64)
        uniforms=np.asarray(uniforms,dtype=np.float64)
        if probabilities.ndim!=2 or probabilities.shape[1]!=3 or uniforms.shape!=(len(probabilities),):
            raise ValueError("action probabilities and uniforms have incompatible shapes")
        cumulative=np.cumsum(probabilities,axis=1)
        return np.minimum((uniforms[:,None]>=cumulative).sum(axis=1),2).astype(int).tolist()

    @staticmethod
    def _account_action_probabilities(logits, portfolio_state=None, explore=False):
        """Use the learned SELL/HOLD/BUY policy, with explicit training exploration."""
        scores=np.asarray(logits,dtype=np.float64).copy()
        if scores.ndim!=2 or scores.shape[1]!=3:
            raise ValueError("action logits must have shape [symbols,3]")
        held=(np.asarray(portfolio_state)[:,0]>0.5 if portfolio_state is not None
              else np.zeros(len(scores),dtype=bool))
        if held.shape!=(len(scores),):
            raise ValueError("portfolio_state does not match action logits")
        scores-=scores.max(axis=-1,keepdims=True)
        exp_scores=np.exp(scores)
        probabilities=exp_scores/exp_scores.sum(axis=-1,keepdims=True)
        if explore:
            epsilon=min(PAPER_EXPLORATION_EPSILON,1.0/max(1,len(scores)))
            probabilities=(1.0-epsilon)*probabilities+epsilon/3.0
        return probabilities

    @classmethod
    def _paper_policy_actions(cls, logits, portfolio_state, uniforms):
        """Apply the same stochastic policy rule and random draws to both live ledgers."""
        probabilities=cls._account_action_probabilities(
            logits,portfolio_state,explore=True)
        return probabilities,cls._sample_actions(probabilities,uniforms)

    @staticmethod
    def _deterministic_actions(probabilities):
        """Use the highest-probability action; exact ties resolve to no-op HOLD."""
        probabilities=np.asarray(probabilities,dtype=np.float64)
        actions=np.argmax(probabilities,axis=-1).astype(int)
        ties=np.isclose(probabilities,probabilities.max(axis=-1,keepdims=True),
                        rtol=0.0,atol=1e-8).sum(axis=-1)>1
        actions[ties]=1
        return actions.tolist()

    def _window(self,panel,index):
        contextual=getattr(self.champion,"_stockrl_uses_market_context",False)
        return panel.window(index,self.window,include_context=contextual)

    def _mature(self,pending,panel,end_index):
        keep=[]
        for dec in pending:
            if dec.get("blocked_reason"):
                keep.append(dec); continue
            if "entry_price" in dec:
                # Keep the entry price with the decision so rolling the panel
                # cannot make the original row disappear before its outcome.
                stamp=np.datetime64(dec["timestamp"])
                now=panel.dates[end_index]
                symbol=dec.get("symbol")
                if symbol not in panel.symbols:
                    keep.append(dec); continue
                symbol_ix=panel.symbols.index(symbol)
                if now<=stamp or not panel.observed[end_index,symbol_ix]:
                    keep.append(dec); continue
                if self.horizon_kind=="bars":
                    dec["bars_elapsed"]=int(dec.get("bars_elapsed",0))+1
                    matured=dec["bars_elapsed"]>=self.horizon_amount
                else:
                    target=stamp+np.timedelta64(self.horizon_amount,"s")
                    matured=now>=target
                if not matured:
                    keep.append(dec); continue
                entry=float(dec["entry_price"]); exit_price=float(panel.closes[end_index,symbol_ix])
                if not np.isfinite(entry*exit_price) or entry<=0:
                    self.metrics["pending_expired_after_window"] = int(
                        self.metrics.get("pending_expired_after_window",0))+1
                    continue
                forward=exit_price/entry-1.0
                dec["index"]=end_index; dec["symbol_index"]=symbol_ix
            else:
                # Finite historical runs still refer to their fixed panel rows.
                if dec.get("timestamp"):
                    stamp=np.datetime64(dec["timestamp"])
                    start_ix=int(np.searchsorted(panel.dates,stamp,side="left"))
                    if start_ix>=len(panel.dates) or panel.dates[start_ix]!=stamp:
                        keep.append(dec); continue
                    symbol_ix=(panel.symbols.index(dec["symbol"])
                               if dec.get("symbol") in panel.symbols else dec["symbol_index"])
                    dec["index"]=start_ix; dec["symbol_index"]=symbol_ix
                symbol_ix=dec["symbol_index"]
                target=self._target_index(panel,dec["index"],symbol_ix)
                if target is None or end_index<target:
                    keep.append(dec); continue
                forward=panel.return_to(dec["index"],target,symbol_ix)
            i=symbol_ix
            self.metrics["matured"]+=1
            if dec.get("promotion_holdout",False):
                self.metrics["promotion_validation_outcomes"] = int(
                    self.metrics.get("promotion_validation_outcomes",0))+1
                self.replay.acknowledge_pending("regular",dec)
                continue
            if dec.get("is_validation",False):
                self._append_validation(self.validation,self.validation_dates,dec["timestamp"],
                    (dec["features"],dec["symbol_ids"],dec["market_ids"],dec["asset_ids"],
                    dec["valid_mask"],i,forward,dec["timestamp"],dec.get("previous_position",0),
                    dec.get("market_context")))
                self.replay.acknowledge_pending("regular",dec)
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
            # This isolated BUY/SELL proxy assumes a trade at the decision
            # price. It is useful as a diagnostic, but is not an executed
            # cash-only account transition, so never place it in training replay.
            self.replay.acknowledge_pending(
                "regular",dec)
        return keep

    def _mature_portfolio(self, pending, panel, end_index: int, filled_orders=()):
        """Turn completed paper-account transitions into reward experiences.

        The reward uses the account's net-of-cost normalized equity change.
        Trade actions begin their horizon on the linked next-bar fill; HOLD
        observations use their decision time because they create no order.
        """
        if not pending:
            return pending
        now = float(self.paper_account.normalized_equity())
        fills_by_id={str(fill.get("decision_id")):fill for fill in filled_orders
                     if fill.get("decision_id")}
        fills_by_order={(str(fill.get("order_date")),str(fill.get("symbol"))):fill
                        for fill in filled_orders if fill.get("order_date")}
        keep=[]; account_transition_added=set()
        for dec in pending:
            if dec.get("blocked_reason"):
                keep.append(dec); continue
            if dec.get("reward_version")!=REWARD_VERSION:
                dec["blocked_reason"]="reward schema is incompatible"
                keep.append(dec)
                continue
            stamp=np.datetime64(dec["timestamp"]); current=panel.dates[end_index]
            symbol=dec.get("symbol")
            if symbol not in panel.symbols:
                keep.append(dec); continue
            symbol_ix=panel.symbols.index(symbol)
            if current<=stamp or not panel.observed[end_index,symbol_ix]:
                keep.append(dec); continue
            input_symbols=dec.get("input_symbols")
            input_symbol_index=(input_symbols.index(symbol) if input_symbols and symbol in input_symbols
                                else int(dec["symbol_index"]))
            if not 0<=input_symbol_index<dec["features"].shape[1]:
                dec["blocked_reason"]="decision symbol is missing from its saved input"
                keep.append(dec)
                continue
            if int(dec.get("action",1))!=1 and not dec.get("fill_expected"):
                # Still learn its zero executed outcome. Never credit unrelated
                # portfolio drift to an order that did not change this position.
                dec["trade_executed"]=False
            if dec.get("fill_expected") and not dec.get("fill_seen"):
                fill=(fills_by_id.get(str(dec.get("decision_id"))) or
                      fills_by_order.get((str(dec.get("timestamp")),str(symbol))))
                if fill is not None:
                    dec["fill_seen"]=True
                    dec["fill_timestamp"]=str(fill["date"])
                    dec["fill_price"]=float(fill["price"])
                    dec["symbol_pnl_before"]=float(
                        fill.get("symbol_pnl_before_fill",dec.get("symbol_pnl_before",0.0)))
                    dec["bars_elapsed"]=0
                    keep.append(dec)
                    continue
                queued=self.paper_account.state.get("pending",{}).get(symbol)
                still_queued=bool(queued and (
                    queued.get("decision_id")==dec.get("decision_id") or
                    (not queued.get("decision_id") and queued.get("date")==dec.get("timestamp"))))
                if still_queued:
                    keep.append(dec)
                else:
                    dec["fill_expected"]=False
                    dec["trade_executed"]=False
                    self.metrics["paper_unfilled_decisions"]=int(
                        self.metrics.get("paper_unfilled_decisions",0))+1
                if still_queued:
                    continue
            reward_start=np.datetime64(dec.get("fill_timestamp",dec["timestamp"]))
            if current<=reward_start:
                keep.append(dec); continue
            if self.horizon_kind=="bars":
                dec["bars_elapsed"]=int(dec.get("bars_elapsed",0))+1
                matured=dec["bars_elapsed"]>=self.horizon_amount
            else:
                matured=current>=reward_start+np.timedelta64(self.horizon_amount,"s")
            if not matured:
                keep.append(dec); continue
            account_reward = now - float(dec.get("equity_before", now))
            symbol_reward = (self.paper_account.symbol_net_pnl(symbol)
                             - float(dec.get("symbol_pnl_before", 0.0)))
            if dec.get("trade_executed") is False:
                symbol_reward=0.0
            if dec.get("promotion_holdout",False):
                self.metrics["promotion_validation_outcomes"] = int(
                    self.metrics.get("promotion_validation_outcomes",0))+1
                self.replay.acknowledge_pending("portfolio",dec)
                continue
            entry=float(dec.get("fill_price",dec.get("entry_price",0.0)))
            exit_price=float(panel.closes[end_index,symbol_ix])
            forward_return=(exit_price/entry-1.0 if entry>0 and np.isfinite(entry*exit_price) else None)
            exp=Experience(dec["features"],dec["symbol_ids"],dec["market_ids"],dec["asset_ids"],
                    dec["valid_mask"],input_symbol_index,int(dec["action"]),float(symbol_reward),
                    dec["timestamp"],"paper_account_symbol",float(dec.get("regime",0.0)),
                    reward_version=REWARD_VERSION,
                    market_context=dec.get("market_context"),
                    multiscale_state=dec.get("multiscale_state"),
                    portfolio_state=dec.get("portfolio_state"),
                    account_state=dec.get("account_state"),forward_return=forward_return,
                    behavior_log_prob=dec.get("behavior_log_prob"),
                    trade_executed=dec.get("trade_executed",True))
            if dec.get("is_validation",False):
                self._append_validation(self.portfolio_validation,self.portfolio_validation_dates,
                                        exp.timestamp,exp)
                self.replay.acknowledge_pending("portfolio",dec)
            else:
                timestamp=dec["timestamp"]
                # Keep one allocation-credit row per symbol. It carries that
                # symbol's realized contribution separately from the shared
                # whole-account result; never attach the whole account return
                # to whichever symbol happened to mature first.
                first_account_transition=(timestamp not in account_transition_added and
                    not self.replay.has_portfolio_value(timestamp))
                account_exp=Experience(dec["features"],dec["symbol_ids"],dec["market_ids"],dec["asset_ids"],
                    dec["valid_mask"],input_symbol_index,int(dec["action"]),float(symbol_reward),timestamp,
                    "paper_account_portfolio",float(dec.get("regime",0.0)),
                    reward_version=REWARD_VERSION,
                    market_context=dec.get("market_context"),
                    multiscale_state=dec.get("multiscale_state"),
                    portfolio_state=dec.get("portfolio_state"),
                    account_state=dec.get("account_state"),portfolio_reward=float(account_reward),
                    portfolio_transition=True,portfolio_value_transition=first_account_transition,
                    forward_return=forward_return,
                    behavior_log_prob=dec.get("behavior_log_prob"),
                    trade_executed=dec.get("trade_executed",True))
                self.replay.add_many([exp,account_exp],pending_ack=("portfolio",
                    f"{dec.get('timestamp','')}|{dec.get('symbol',dec.get('symbol_index',''))}"))
                if first_account_transition:
                    account_transition_added.add(timestamp)
                    self.metrics["paper_account_reward"] = float(
                        self.metrics.get("paper_account_reward",0.0))+account_reward
                    self.metrics["portfolio_experiences"] = int(
                        self.metrics.get("portfolio_experiences",0))+1
                self.metrics["paper_experiences_seen"]=int(
                    self.metrics.get("paper_experiences_seen",0))+1
                self.metrics["paper_experiences_since_candidate"]=int(
                    self.metrics.get("paper_experiences_since_candidate",0))+1
                self.replay.note_paper_outcome()
        return keep

    def follow_csv(self, data_path:str|Path, poll_seconds:float=5.0, initial_lookback_bars:int=0):
        """Observe an append-only UTC timestamp CSV until Ctrl+C.

        Feed columns follow the same global schema as the historical downloader.
        New bars can be minute/hourly; all instruments for a timestamp should
        be appended as a batch. Candidate training runs on the learner thread.
        """
        data_path=Path(data_path); cursor_path=self.state_dir/"live_cursor.json"
        decisions_path=self.state_dir/"decisions.csv"
        cursor=None
        last_file_signature=None
        last_error_signature=None
        if cursor_path.exists(): cursor=json.loads(cursor_path.read_text(encoding="utf-8")).get("last_timestamp")
        account_cursor=self.paper_account.state.get("last_timestamp")
        if account_cursor and (cursor is None or account_cursor>cursor): cursor=account_cursor
        pending=self.replay.load_pending("regular")
        portfolio_pending=self.replay.load_pending("portfolio")
        market_reader=IncrementalMarketCSV(data_path,retain_timestamps=4096)
        self.metrics["pending_experiences"] = len(pending) + len(portfolio_pending)
        self.start()
        try:
            while not self.stop.is_set():
                if (self.state_dir/"stop.request").exists():
                    break
                if not data_path.exists():
                    self.stop.wait(poll_seconds); continue
                try:
                    stat=data_path.stat()
                    file_signature=(stat.st_size,stat.st_mtime_ns,getattr(stat,"st_ino",0))
                    if file_signature==last_file_signature:
                        self.stop.wait(poll_seconds); continue
                    self.metrics["observation_caught_up"]=False
                    self.metrics["agent_health"]="catching_up"
                    market_reader.processed_through=cursor
                    raw_frame,_reader_signature=market_reader.refresh()
                    quote_rows=raw_frame.groupby("symbol",sort=False).tail(1)
                    if {"bid","ask"}.issubset(quote_rows.columns):
                        bids=np.asarray(quote_rows["bid"],dtype=float)
                        asks=np.asarray(quote_rows["ask"],dtype=float)
                        self.metrics["quoted_bid_ask_symbols"]=int(
                            (np.isfinite(bids)&np.isfinite(asks)&(bids>0)&(asks>=bids)).sum())
                    else:
                        self.metrics["quoted_bid_ask_symbols"]=0
                    panel=GlobalMarketPanel(data_path, max_symbols=self.cfg.max_symbols,
                        symbol_map=getattr(self.champion,"_stockrl_symbol_map",None),
                        # Keep closed-session indices, futures, yields, and other
                        # reference markets in the action set using their last
                        # known quote. The dashboard still marks their quote as
                        # stale; dropping them here hid their BUY/HOLD/SELL
                        # output entirely whenever their venue was closed.
                        recent_timestamps=None, active_stale_seconds=604800,
                        raw_frame=raw_frame)
                    if (self.candidate is not None and not self.validation_active
                            and not self.metrics.get("candidate_training")
                            and self.metrics.get("candidate_has_learning")
                            and (self.validation_restart_needs_fresh_trial or
                                 self.candidate_version>self.last_validated_candidate_version)):
                        self._begin_candidate_validation(
                            self.candidate,start_after=str(panel.dates[-1]))
                except (OSError,ValueError) as exc:
                    # A producer may be in the middle of appending a CSV batch.
                    # Expose a stable error per file version; otherwise an
                    # input failure looks like a healthy but idle agent.
                    if file_signature != last_error_signature:
                        from datetime import datetime, timezone
                        self.metrics["agent_health"]="input_error"
                        self.metrics["agent_input_errors"]=int(
                            self.metrics.get("agent_input_errors",0))+1
                        self.metrics["agent_last_input_error"]=f"{type(exc).__name__}: {exc}"
                        self.metrics["agent_last_input_error_utc"]=datetime.now(timezone.utc).isoformat()
                        self.metrics["observation_caught_up"]=False
                        error_record={
                            "time_utc":self.metrics["agent_last_input_error_utc"],
                            "file_bytes":stat.st_size,"file_mtime_ns":stat.st_mtime_ns,
                            "error":self.metrics["agent_last_input_error"],
                        }
                        error_log=self.state_dir/"agent_errors.jsonl"
                        if error_log.exists() and error_log.stat().st_size >= 2*1024*1024:
                            error_log.write_text("",encoding="utf-8")
                        with error_log.open("a",encoding="utf-8") as log:
                            log.write(json.dumps(error_record,ensure_ascii=False)+"\n")
                            log.flush(); os.fsync(log.fileno())
                        self._write_metrics()
                        last_error_signature=file_signature
                    self.stop.wait(min(poll_seconds,1.0)); continue
                if last_error_signature is not None:
                    self.metrics["agent_health"]="catching_up"
                    self.metrics["agent_last_input_error"]=None
                    self.metrics["agent_last_input_error_utc"]=None
                    last_error_signature=None
                self.metrics["unmatched_live_symbols"] = list(
                    getattr(panel,"unmatched_symbols",()))
                self.metrics["unmatched_live_symbol_count"] = len(
                    self.metrics["unmatched_live_symbols"])
                if not len(panel.dates):
                    self.stop.wait(poll_seconds); continue
                oldest_available=(panel.recent_cutoff if panel.recent_cutoff is not None
                                  else panel.dates[0])
                if cursor is not None and np.datetime64(cursor) < oldest_available:
                    self.metrics["agent_health"]="history_gap"
                    self.metrics["observation_caught_up"]=False
                    self.metrics["agent_history_gap"]={
                        "saved_cursor":str(cursor),"oldest_available":str(oldest_available),
                        "retained_timestamps":4096,
                    }
                    self.metrics["candidate_skip_reason"]=(
                        "저장 cursor가 보존된 시세 범위보다 오래되어 연속 처리를 대기 중")
                    self._write_metrics()
                    last_file_signature=file_signature
                    self.stop.wait(poll_seconds)
                    continue
                self.metrics["agent_history_gap"]=None
                restore_windows={}
                restore_multiscale={}
                expired_regular=expired_portfolio=0
                def restore_pending(items, is_portfolio):
                    nonlocal expired_regular, expired_portfolio
                    restored=[]
                    for dec in items:
                        if dec.get("blocked_reason"):
                            restored.append(dec); continue
                        if "features" in dec:
                            restored.append(dec)
                            continue
                        symbol=dec.get("symbol")
                        if symbol not in panel.symbols:
                            restored.append(dec)
                            continue
                        stamp=np.datetime64(dec["timestamp"])
                        index=int(np.searchsorted(panel.dates,stamp,side="left"))
                        if index>=len(panel.dates) or panel.dates[index]!=stamp:
                            dec["blocked_reason"]="legacy pending input is unavailable"
                            restored.append(dec)
                            if is_portfolio: expired_portfolio+=1
                            else: expired_regular+=1
                            continue
                        window=restore_windows.get(index)
                        if window is None:
                            window=self._window(panel,index)
                            restore_windows[index]=window
                            restore_multiscale[index]=panel.multiscale_at(index).astype(np.float16)
                        x0,sid0,mid0,aid0,mask0=window[:5]
                        dec.update({"index":index,"symbol_index":panel.symbols.index(symbol),
                            "features":x0[0].numpy().astype(np.float16),"symbol_ids":sid0[0].numpy(),
                            "market_ids":mid0[0].numpy(),"asset_ids":aid0[0].numpy(),
                            "valid_mask":mask0[0].numpy(),
                            "market_context":window[5][0].numpy().astype(np.float16) if len(window)>5 else None,
                            "multiscale_state":restore_multiscale[index]})
                        if is_portfolio and dec.get("portfolio_state") is not None:
                            old_state=np.asarray(dec["portfolio_state"])
                            old_symbols=dec.get("input_symbols")
                            if old_symbols and len(old_symbols)==len(old_state):
                                by_symbol=dict(zip(old_symbols,old_state))
                                dec["portfolio_state"]=np.asarray([
                                    by_symbol.get(name,np.zeros(8,dtype=np.float16))
                                    for name in panel.symbols],dtype=np.float16)
                            elif old_state.shape!=(len(panel.symbols),8):
                                dec["blocked_reason"]="legacy portfolio input shape is incompatible"
                                restored.append(dec)
                                expired_portfolio+=1
                                continue
                            dec["input_symbols"]=list(panel.symbols)
                        restored.append(dec)
                    return restored
                pending=restore_pending(pending,False)
                portfolio_pending=restore_pending(portfolio_pending,True)
                if expired_regular or expired_portfolio:
                    self.metrics["pending_expired_after_window"] = int(
                        self.metrics.get("pending_expired_after_window",0))+expired_regular
                    self.metrics["portfolio_pending_expired_after_window"] = int(
                        self.metrics.get("portfolio_pending_expired_after_window",0))+expired_portfolio
                    self.metrics["pending_experiences"] = len(pending)+len(portfolio_pending)
                    self.replay.save_pending(pending,portfolio_pending)
                # The first feed batch establishes the ordered symbol universe;
                # migrate legacy integer-keyed paper positions to symbols.
                if any(isinstance(k,int) for k in self.positions):
                    self.positions={panel.symbols[k] if isinstance(k,int) and 0<=k<len(panel.symbols) else str(k):v
                                    for k,v in self.positions.items()}
                if cursor is None:
                    # Every available timestamp enters the rolling observation
                    # stream. Context length does not cap daily throughput.
                    start=0
                else:
                    cursor_dt=np.datetime64(cursor)
                    start=int(np.searchsorted(panel.dates,cursor_dt,side="right"))
                for ti in range(start,len(panel.dates)):
                    if self.stop.is_set(): break
                    self.current_market_timestamp=str(panel.dates[ti])
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
                    filled_orders=self.paper_account.process_bar(panel,ti,paper_enabled)
                    portfolio_pending=self._mature_portfolio(
                        portfolio_pending,panel,ti,filled_orders)
                    pending=self._mature(pending,panel,ti)
                    if not observe_enabled:
                        cursor=str(panel.dates[ti]); self.metrics["observations"]+=1
                        self.metrics["last_market_timestamp"]=cursor
                        self.metrics["pending_experiences"] = len(pending)+len(portfolio_pending)
                        self.replay.save_pending(pending,portfolio_pending)
                        self.paper_account.save()
                        _atomic_json({"last_timestamp":cursor},cursor_path)
                        continue
                    pstate,astate=self.paper_account.model_inputs(panel,ti)
                    logits,values,allocation=self._infer(panel,ti,pstate,astate)
                    uniforms=[self.policy_rng.random() for _ in range(len(logits))]
                    probs,actions=self._paper_policy_actions(logits,pstate,uniforms)
                    window=self._window(panel,ti)
                    x,sid,mid,aid,mask=window[:5]
                    # The frozen trial cannot learn these future observations.
                    # The continuing candidate may learn them; its NEXT trial
                    # starts strictly after its own snapshot time.
                    stamp=str(panel.dates[ti]); validation=False
                    multiscale_state=panel.multiscale_at(ti).astype(np.float16)
                    actual_symbols=np.asarray([
                        not str(symbol).startswith("__PAD__") for symbol in panel.symbols],dtype=bool)
                    if actual_symbols.any():
                        width=len(TIMEFRAME_FEATURE_NAMES)
                        self.metrics["multiscale_coverage"]={
                            scale:float(multiscale_state[actual_symbols,k*width+width-1].mean())
                            for k,scale in enumerate(TIMEFRAME_NAMES)}
                    rows=[]
                    pending_inputs={"features":x[0].numpy().astype(np.float16),
                        "symbol_ids":sid[0].numpy(),"market_ids":mid[0].numpy(),
                        "asset_ids":aid[0].numpy(),"valid_mask":mask[0].numpy(),
                        "market_context":window[5][0].numpy().astype(np.float16) if len(window)>5 else None,
                        "multiscale_state":multiscale_state}
                    portfolio_inputs={**pending_inputs,
                        "input_symbols":list(panel.symbols),
                        "portfolio_state":np.asarray(pstate,dtype=np.float16),
                        "account_state":np.asarray(astate,dtype=np.float32)}
                    for j,symbol in enumerate(panel.symbols):
                        # A closed venue may not print a bar at the current
                        # global timestamp. Use its forward-filled last quote
                        # for a visible BUY/HOLD/SELL decision, while the
                        # observed mask remains untouched for reward maturity.
                        if not panel.observed[:ti + 1, j].any(): continue
                        action=int(actions[j])
                        behavior_log_prob=float(np.log(max(float(probs[j,action]),1e-12)))
                        previous=self.positions.get(symbol,0)
                        market,asset=panel.groups[symbol]
                        if (paper_enabled and panel.observed[ti,j] and
                                _currency(market,asset) is not None):
                            decision_id=f"{stamp}|{symbol}"
                            portfolio_pending.append({**portfolio_inputs,"index":ti,"symbol_index":j,"symbol":symbol,
                              "action":action,"timestamp":stamp,"decision_id":decision_id,
                              "reward_version":REWARD_VERSION,
                              "behavior_log_prob":behavior_log_prob,
                              "entry_price":float(panel.closes[ti,j]),"bars_elapsed":0,
                              "regime":abs(float(panel.features[ti,j,6])),
                               "is_validation":False,"promotion_holdout":validation,
                              "equity_before":self.paper_account.normalized_equity(),
                              "symbol_pnl_before":self.paper_account.symbol_net_pnl(symbol),
                              })
                        rows.append({"date":stamp,"symbol":symbol,"action":ACTION_NAMES[action],"value":float(values[j]),
                          "p_sell":float(probs[j,0]),"p_hold":float(probs[j,1]),"p_buy":float(probs[j,2])})
                        self.metrics["decisions"]+=1
                        if paper_enabled:
                            self.positions[symbol]=action-1
                    submitted_order_ids=self.paper_account.queue_decisions(
                        panel,ti,probs,paper_enabled,allocation=allocation,actions=actions)
                    submitted_order_ids=submitted_order_ids or set()
                    for dec in portfolio_pending:
                        if dec.get("timestamp")==stamp:
                            dec["fill_expected"]=(dec.get("decision_id") in submitted_order_ids)
                    self.metrics["pending_experiences"] = len(pending)+len(portfolio_pending)
                    self.replay.save_pending(pending,portfolio_pending)
                    self.paper_account.save()
                    self._queue_candidate_live_observation(panel,ti,paper_enabled,uniforms)
                    self._collect_candidate_validation(panel,ti)
                    import pandas as pd
                    if rows:
                        # Decision history is an observation aid, not replay.
                        # Clear the same file when full; never make dated copies.
                        if decisions_path.exists() and decisions_path.stat().st_size >= 16*1024*1024:
                            decisions_path.unlink()
                        has_header = decisions_path.exists() and decisions_path.stat().st_size > 0
                        pd.DataFrame(rows).to_csv(decisions_path,mode="a",header=not has_header,index=False)
                    self.metrics["observations"]+=1; cursor=stamp
                    self.metrics["last_market_timestamp"]=stamp
                    _atomic_json({"last_timestamp":cursor},cursor_path)
                    _atomic_json({str(k):v for k,v in self.positions.items()},self.state_dir/"live_positions.json")
                # Catch-up means the agent finished the complete timestamp
                # snapshot it loaded. The feed may append another bar while
                # inference is running; waiting for an unchanged file can
                # otherwise keep the learner disabled indefinitely.
                caught_up = cursor is not None and np.datetime64(cursor) >= panel.dates[-1]
                self.metrics["observation_caught_up"]=bool(caught_up)
                if caught_up:
                    self.metrics["agent_health"]=("observing_with_unmapped_symbols"
                        if self.metrics["unmatched_live_symbol_count"] else "observing")
                else:
                    self.metrics["agent_health"]="catching_up"
                self._write_metrics()
                last_file_signature=file_signature
                self.stop.wait(poll_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop.set()
            if self.thread.is_alive(): self.thread.join(timeout=300)
            if self.validation_queue_thread.is_alive():
                self.validation_queue_thread.join(timeout=300)
            self.metrics["candidate_training"]=False
            self.metrics["pending_experiences"] = len(pending)+len(portfolio_pending)
            self.replay.save_pending(pending,portfolio_pending)
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
                    self._append_validation(self.validation,self.validation_dates,str(panel.dates[ti]),
                        (x[0].numpy().astype(np.float16),sid[0].numpy(),mid[0].numpy(),aid[0].numpy(),
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
        # Complete all pending decisions with known later observations.
        if len(panel.dates): pending=self._mature(pending,panel,len(panel.dates)-1)
        pd_frame=None
        import pandas as pd
        pd_frame=pd.DataFrame(history)
        pd_frame.to_csv(self.state_dir/"decisions.csv",index=False)
        return history

    def evaluate_panel(self,panel:GlobalMarketPanel,model=None,start_index:int|None=None,
                       end_index:int|None=None,stride:int=1,report_rows=None)->dict:
        """Evaluate real cash-only portfolio actions using the live paper ledger."""
        model=model or self.champion
        start=max(self.window-1,0 if start_index is None else start_index)
        end=len(panel.dates) if end_index is None else min(len(panel.dates),end_index)
        if stride<1:
            raise ValueError("stride must be positive")
        account=PaperAccount.in_memory(self.fee,self.slippage)
        initial=account.normalized_equity()
        peak=initial; max_dd=0.0; daily={}; action_counts={name:0 for name in ACTION_NAMES}
        decisions=0; observed_bars=0; previous_equity=initial
        was_training=model.training; model.eval()
        model_device=next(model.parameters()).device
        try:
            with torch.inference_mode():
                for ti in range(start,end):
                    fills=account.process_bar(panel,ti,True)
                    if (ti-start)%stride==0:
                        contextual=getattr(model,"_stockrl_uses_market_context",False)
                        args=[x.to(model_device) for x in panel.window(ti,self.window,include_context=contextual)]
                        args[0]=args[0].to(dtype=next(model.parameters()).dtype)
                        pstate,astate=account.model_inputs(panel,ti)
                        if contextual:
                            logits,values,allocation=model(*args,
                                portfolio_state=torch.as_tensor(np.asarray(pstate)[None],device=model_device,dtype=torch.float32),
                                account_state=torch.as_tensor(np.asarray(astate)[None],device=model_device,dtype=torch.float32),
                                multiscale_state=torch.as_tensor(panel.multiscale_at(ti)[None],device=model_device),
                                return_allocation=True)
                            allocation=allocation[0].float().cpu().numpy()
                        else:
                            logits,values=model(*args); allocation=None
                        probabilities=self._account_action_probabilities(logits[0].float().cpu().numpy(),pstate)
                        actions=self._deterministic_actions(probabilities)
                        account.queue_decisions(panel,ti,probabilities,True,allocation=allocation,actions=actions)
                        for j,symbol in enumerate(panel.symbols):
                            if panel.observed[ti,j] and _currency(*panel.groups[symbol]) is not None:
                                action_counts[ACTION_NAMES[actions[j]]]+=1
                                decisions+=1
                    equity=account.normalized_equity()
                    peak=max(peak,equity)
                    max_dd=max(max_dd,(peak-equity)/max(peak,1e-9))
                    day=str(panel.dates[ti])[:10]
                    daily[day]=daily.get(day,0.0)+(equity-previous_equity)/initial
                    previous_equity=equity; observed_bars+=1
                    if report_rows is not None:
                        report_rows.append({"date":str(panel.dates[ti]),
                            "net_asset_return":equity/initial-1.0,"fills":len(fills),
                            "KRW_equity":account._equity("KRW"),"USD_equity":account._equity("USD")})
        finally:
            model.train(was_training)
        books=account.snapshot()["books"]
        costs=sum(sum(float(book[k]) for k in ("fees","sell_tax","spread","slippage"))
                  /float(book["initial_cash"]) for book in books.values())/initial
        net_return=account.normalized_equity()/initial-1.0
        return {"start_index":start,"end_index":end,"timestamps":observed_bars,"decisions":decisions,
            "evaluation_mode":"cash_only_sequential_paper_account",
            "score_definition":"mean of KRW and USD seed-normalized net asset returns; no raw currency sum",
            "net_return":net_return,"net_pnl_return_sum":net_return,"compounded_step_return":net_return,
            "gross_pnl_return_sum":net_return+costs,"cost_return_sum":costs,
            "max_drawdown":max_dd,"action_counts":action_counts,"daily_net_return":daily,
            "fee_rate":self.fee,"slippage_bps":self.slippage*10000,"books":books,
            "trade_count":sum(book["trade_count"] for book in books.values()),
            "unfilled_orders_at_end":len(account.state["pending"]),
            "fills_at":"next available observation for that symbol",
            "data_usage":"historical diagnostic; held-out status requires explicit training lineage"}

    def backtest_panel(self,panel:GlobalMarketPanel,stride:int=1)->dict:
        """Run the final 15% with the same ledger used in live and promotion trials."""
        import pandas as pd
        started=time.perf_counter()
        start=max(1,int(len(panel.dates)*.85))
        rows=[]
        out=self.evaluate_panel(panel,self.champion,start_index=start,stride=stride,report_rows=rows)
        pd.DataFrame(rows,columns=["date","net_asset_return","fills","KRW_equity","USD_equity"]).to_csv(
            self.state_dir/"backtest.csv",index=False)
        out["elapsed_seconds"]=float(time.perf_counter()-started)
        out["checkpoint_sha256"]=self._sha256_file(self.champion_path)
        out["completed_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
        self._atomic_json(out,self.state_dir/"backtest.json")
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
        m_rows=[]
        for experience in batch:
            p_rows.append(experience.portfolio_state if experience.portfolio_state is not None
                          else np.zeros((experience.features.shape[1], 8), dtype=np.float16))
            a_rows.append(experience.account_state if experience.account_state is not None
                          else np.zeros(8, dtype=np.float32))
            m_rows.append(experience.multiscale_state if experience.multiscale_state is not None
                          else np.zeros((experience.features.shape[1],MULTISCALE_FEATURE_COUNT),dtype=np.float16))
        return (args,
                torch.as_tensor(np.stack(p_rows),device=self.device,dtype=torch.float32),
                torch.as_tensor(np.stack(a_rows),device=self.device,dtype=torch.float32),
                torch.as_tensor(np.stack(m_rows),device=self.device,dtype=torch.float32))

    def _learner(self):
        while not self.stop.wait(.1):
            stats=self.replay.stats(self.candidate_replay_passes)
            self.metrics["candidate_untrained_replay_count"]=stats["untrained"]
            self.metrics["candidate_eligible_replay_count"]=stats["eligible"]
            self.metrics["candidate_replay_passes"]=self.candidate_replay_passes
            if not stats["eligible"]:
                self.metrics["candidate_skip_reason"]="남은 학습 가능한 경험을 모두 처리했습니다. 새 경험 대기"
                continue
            if self.metrics.get("candidate_training"):
                continue
            if time.monotonic()<self.candidate_retry_after:
                self.metrics["candidate_skip_reason"]="Candidate 오류 뒤 경험을 보존하고 재시도 대기 중"
                continue
            self.last_train_replay_size=len(self.replay)
            self.metrics["candidate_skip_reason"]=None
            try:
                self._train_candidate()
            except Exception as exc:
                self.metrics["candidate_training"]=False
                self.metrics["candidate_errors"]=int(self.metrics.get("candidate_errors",0))+1
                self.metrics["last_candidate_error"]=f"{type(exc).__name__}: {exc}"[:1200]
                exc.__traceback__=None
                del exc
                self._schedule_candidate_retry()
                self._release_cuda_cache()
                try:
                    self._restore_candidate_checkpoint()
                    self._write_metrics()
                except Exception: pass

    def _schedule_candidate_retry(self):
        self.candidate_retry_attempts+=1
        delay=min(300.0,15.0*(2**min(self.candidate_retry_attempts-1,5)))
        self.candidate_retry_after=time.monotonic()+delay
        self.metrics["candidate_retry_delay_seconds"]=delay

    def _release_cuda_cache(self):
        gc.collect()
        if self.device.type=="cuda":
            try: torch.cuda.empty_cache()
            except RuntimeError: pass

    def _configure_candidate_trainables(self,candidate):
        """Fine-tune the upper transformer blocks and small policy adapters only."""
        backbone=getattr(candidate,"backbone",candidate)
        for parameter in candidate.parameters():
            parameter.requires_grad_(False)
        blocks=list(getattr(backbone,"blocks",()))
        trainable_blocks=blocks[-ONLINE_TRAINABLE_BLOCKS:]
        modules=list(trainable_blocks)
        modules.extend(getattr(backbone,name,None) for name in
                       ("final_norm","policy_head","value_head"))
        modules.extend(getattr(candidate,name,None) for name in
                       ("context_policy","context_value","portfolio_action",
                        "portfolio_allocation","portfolio_cash",
                        "multiscale_policy","multiscale_value"))
        for module in modules:
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        trainable=[parameter for parameter in candidate.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("candidate has no trainable parameters")
        self.metrics["candidate_trainable_transformer_blocks"]=len(trainable_blocks)
        self.metrics["candidate_trainable_parameter_count"]=sum(
            parameter.numel() for parameter in trainable)
        self.metrics["candidate_total_parameter_count"]=sum(
            parameter.numel() for parameter in candidate.parameters())
        return trainable

    def _experience_loss(self,logits,values,allocations,experience):
        chosen=logits[0,int(experience.symbol_index)].float()
        predicted=values[0,int(experience.symbol_index)].float()
        if not torch.isfinite(chosen).all() or not torch.isfinite(predicted).all():
            self.metrics["nonfinite_updates"]+=1
            return None
        action=torch.as_tensor([experience.action],device=self.device,dtype=torch.long)
        epsilon=min(PAPER_EXPLORATION_EPSILON,1.0/max(1,int(experience.features.shape[1])))
        probabilities=(1.0-epsilon)*torch.softmax(chosen,dim=-1)+epsilon/3.0
        dist=Categorical(probs=probabilities[None])
        reward=torch.as_tensor(float(experience.reward)*100.0,device=self.device,dtype=torch.float32)
        loss=chosen.sum()*0
        if not experience.portfolio_transition:
            advantage=(reward-predicted.detach()).clamp(-1,1)
            if experience.behavior_log_prob is not None and experience.trade_executed:
                old=torch.as_tensor(float(experience.behavior_log_prob),device=self.device,dtype=torch.float32)
                ratio=torch.exp((dist.log_prob(action)[0]-old).clamp(-20,20))
                loss=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)
            loss=loss+.5*nn.functional.smooth_l1_loss(predicted[None],reward[None])
            loss=loss-.0005*dist.entropy().mean()
        else:
            if experience.portfolio_value_transition:
                account_reward=torch.as_tensor(100.0*float(experience.portfolio_reward or 0.0),
                                               device=self.device,dtype=torch.float32)
                loss=loss+.25*nn.functional.smooth_l1_loss(values[0].float().mean(),account_reward)
            if allocations is not None and experience.trade_executed:
                selected=allocations[0,int(experience.symbol_index)].clamp_min(1e-7)
                loss=loss-.10*reward.detach().clamp(-1,1)*torch.log(selected)
        if experience.source.startswith("teacher"):
            loss=loss+nn.functional.cross_entropy(chosen[None],action)
        if not torch.isfinite(loss):
            self.metrics["nonfinite_updates"]+=1
            return None
        return loss

    def _train_candidate(self):
        from datetime import datetime, timezone
        training_started=time.perf_counter()
        trigger_experience_count=int(self.metrics.get("paper_experiences_since_candidate",0))
        self.metrics["candidate_training"]=True
        self.metrics["candidate_skip_reason"] = None
        self.metrics["last_candidate_compute_seconds"]=0.0
        self.metrics["last_candidate_step_compute_seconds"]=0.0
        self.metrics["last_candidate_total_seconds"]=0.0
        self.metrics["last_candidate_peak_allocated_bytes"]=None
        self.metrics["last_candidate_peak_reserved_bytes"]=None
        with self.lock: source_champion=self.champion
        candidate_path=self.model_dir/"candidate.pt"
        if self.candidate is None:
            self._restore_candidate_checkpoint()
        with self.candidate_model_lock:
            self.candidate=self.candidate.to(self.device)
            candidate=self.candidate.train()
        original=source_champion
        trainable_parameters=self._configure_candidate_trainables(candidate)
        opt=torch.optim.AdamW(
            trainable_parameters,lr=self.lr,weight_decay=.01,
            eps=1e-4 if self.device.type=="cuda" else 1e-8,foreach=False)
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            training_baseline_allocated=int(torch.cuda.memory_allocated(self.device))
        else:
            training_baseline_allocated=0
        elapsed=[]; compute_elapsed=[]
        candidate_samples=0; candidate_sample_keys=set(); used_experiences=[]
        sampled_ids=set()
        # The SQLite queue is authoritative; the small RAM cache is not the
        # backlog. Finish the oldest rows, including their second pass.
        self.metrics["last_candidate_optimizer_steps"]=0
        self.metrics["last_candidate_samples_trained"]=0
        self.metrics["candidate_optimizer_steps_current"]=0
        self.metrics["candidate_samples_current"]=0
        self.metrics["candidate_window_forwards_current"]=0
        self.metrics["candidate_optimizer_steps_target"]=self.updates_per_candidate
        self.metrics["candidate_samples_target"]=self.updates_per_candidate*self.batch_size
        for update_ix in range(self.updates_per_candidate):
            if self.stop.is_set():
                break
            batch=self.replay.pending_batch(self.batch_size,self.candidate_replay_passes,
                exclude_row_ids=sampled_ids)
            if not batch: break
            sampled_ids.update(self.replay.row_ids_for(batch))
            malformed=[e for e in batch if (
                (e.portfolio_state is not None and np.shape(e.portfolio_state)!=(e.features.shape[1],8)) or
                (e.account_state is not None and np.shape(e.account_state)!=(8,)) or
                (e.multiscale_state is not None and np.shape(e.multiscale_state)!=(e.features.shape[1],MULTISCALE_FEATURE_COUNT)))]
            if malformed:
                self.replay.quarantine(self.replay.row_ids_for(malformed),"saved portfolio input shape is incompatible")
                rejected={id(e) for e in malformed}
                batch=[e for e in batch if id(e) not in rejected]
                if not batch: continue
            candidate_sample_keys.update((e.source,e.timestamp,e.symbol_index,e.action,e.portfolio_transition)
                                         for e in batch)
            ti=time.perf_counter(); opt.zero_grad(set_to_none=True); valid_samples=0; loss_values=[]
            with self.candidate_live_inference_lock:
                if self.device.type=="cuda":
                    compute_start=torch.cuda.Event(enable_timing=True)
                    compute_end=torch.cuda.Event(enable_timing=True)
                    compute_start.record()
                else:
                    compute_ti=time.perf_counter()
                use_portfolio=getattr(candidate,"_stockrl_uses_market_context",False)
                # One forward per shared market/account window, with losses for
                # every selected symbol. This retains the same minibatch gradient
                # without recomputing the 0.5B backbone for each symbol outcome.
                successful_batch=[]
                groups=OrderedDict()
                for experience in batch:
                    key=getattr(experience,"_replay_window_key",None) or self.replay._window_key(experience)
                    groups.setdefault(key,[]).append(experience)
                for group in groups.values():
                    args,pstate,astate,mstate=self._pack([group[0]])
                    if use_portfolio:
                        logits,values,allocations=candidate(*args,portfolio_state=pstate,
                            account_state=astate,return_allocation=True,multiscale_state=mstate)
                    else:
                        logits,values=candidate(*args); allocations=None
                    losses=[]
                    for experience in group:
                        loss=self._experience_loss(logits,values,allocations,experience)
                        if loss is None:
                            continue
                        losses.append(loss)
                        successful_batch.append(experience)
                        loss_values.append(float(loss.detach().cpu()))
                    if losses:
                        (torch.stack(losses).sum()/len(batch)).backward()
                        valid_samples+=len(losses)
                    self.metrics["candidate_window_forwards_current"]+=1
                    del args,pstate,astate,mstate,logits,values,allocations,losses
                if not valid_samples:
                    continue
                nn.utils.clip_grad_norm_(candidate.parameters(),1.0)
                with self.candidate_model_lock:
                    opt.step()
                    if self.device.type=="cuda": torch.cuda.synchronize(self.device)
                if self.device.type=="cuda":
                    compute_end.record()
                else:
                    compute_elapsed.append(time.perf_counter()-compute_ti)
            candidate_samples+=valid_samples
            self.metrics["teacher_examples_trained"]+=sum(e.source.startswith("teacher") for e in successful_batch)
            self.metrics["paper_examples_trained"]+=sum(not e.source.startswith("teacher") for e in successful_batch)
            used_experiences.extend(successful_batch)
            self.metrics["update_losses"].append(float(np.mean(loss_values)))
            self.metrics["update_losses"]=self.metrics["update_losses"][-2000:]
            if self.device.type=="cuda":
                torch.cuda.synchronize(self.device)
                compute_elapsed.append(compute_start.elapsed_time(compute_end)/1000.0)
                self.metrics["last_candidate_peak_allocated_bytes"]=int(
                    torch.cuda.max_memory_allocated(self.device))
                self.metrics["last_candidate_peak_reserved_bytes"]=int(
                    torch.cuda.max_memory_reserved(self.device))
            elapsed.append(time.perf_counter()-ti); self.steps+=1
            self.metrics["candidate_optimizer_steps_current"]=len(elapsed)
            self.metrics["candidate_samples_current"]=candidate_samples
            self.metrics["last_candidate_compute_seconds"]=float(sum(compute_elapsed))
            self.metrics["last_candidate_step_compute_seconds"]=(
                float(sum(compute_elapsed)/len(compute_elapsed)) if compute_elapsed else 0.0)
            self.metrics["last_candidate_total_seconds"]=float(time.perf_counter()-training_started)
            self._write_metrics()
        self.metrics["update_seconds"].extend(elapsed); self.metrics["updates"]+=len(elapsed)
        self.metrics["last_candidate_update_seconds"]=float(sum(elapsed))
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
            self.metrics["last_candidate_peak_allocated_bytes"]=int(
                torch.cuda.max_memory_allocated(self.device))
            self.metrics["last_candidate_baseline_allocated_bytes"]=training_baseline_allocated
            self.metrics["last_candidate_peak_reserved_bytes"]=int(
                torch.cuda.max_memory_reserved(self.device))
        self.metrics["last_candidate_optimizer_steps"]=len(elapsed)
        self.metrics["last_candidate_samples_trained"]=candidate_samples
        self.metrics["last_candidate_unique_samples_trained"]=len(candidate_sample_keys)
        self.metrics["candidate_optimizer_steps_target"]=self.updates_per_candidate
        self.metrics["candidate_samples_target"]=self.updates_per_candidate*self.batch_size
        self.metrics["update_seconds"]=self.metrics["update_seconds"][-2000:]
        self.metrics["teacher_mix_probability"]=self.replay.teacher_fraction()
        if not elapsed:
            self.metrics["candidate_training"]=False
            if not self.stop.is_set():
                self.metrics["candidate_errors"]=int(self.metrics.get("candidate_errors",0))+1
                self.metrics["last_candidate_error"]="no finite optimizer update; replay retained for retry"
                self._schedule_candidate_retry()
            self.candidate.eval()
            if self.device.type=="cuda": self.candidate.to("cpu")
            self._write_metrics()
            del candidate,opt
            self._release_cuda_cache()
            return
        if not all(torch.isfinite(p).all() for p in candidate.parameters()):
            self.metrics["nonfinite_updates"]+=1
            self.metrics["rejections"]+=1
            self._schedule_candidate_retry()
            self.metrics["last_rejection_utc"]=datetime.now(timezone.utc).isoformat()
            history=self.metrics.setdefault("candidate_gate_history",[])
            history.append({"time_utc":self.metrics["last_rejection_utc"],"applied":False,
                            "reason":"가중치에 계산 불가능한 값이 발생해 적용하지 않음",
                            "candidate_score":None,"champion_score":None})
            self.metrics["candidate_gate_history"]=history[-20:]
            self.metrics["candidate_training"]=False
            self._restore_candidate_checkpoint()
            self._write_metrics()
            del candidate,opt
            self._release_cuda_cache()
            return
        delta=sum((candidate.state_dict()[k].float()-v.detach().float()).abs().sum().item()
                  for k,v in original.state_dict().items())
        self.metrics["weight_delta_l1"].append(delta)
        self.metrics["weight_delta_l1"]=self.metrics["weight_delta_l1"][-2000:]
        self.metrics["last_update_utc"]=datetime.now(timezone.utc).isoformat()
        self.metrics["last_candidate_error"]=None
        self.state_dir.mkdir(exist_ok=True,parents=True)
        # Candidate optimizer state is intentionally ephemeral across runs;
        # omit it from the artifact so evaluation/recovery only loads weights.
        trained_replay_row_ids=self.replay.row_ids_for(used_experiences)
        uses={getattr(e,"_replay_row_id",self.replay.row_ids.get(id(e),id(e))):
              int(getattr(e,"_replay_training_uses",0))+1 for e in used_experiences}
        replay_commit={"uses":uses,"passes":self.candidate_replay_passes,
                       "candidate_version":self.candidate_version+len(elapsed)}
        save_model(self.model_dir/"candidate.pt",candidate,self.cfg,step=self.steps,
                   temp_dir=self.state_dir,replay_commit=replay_commit)
        candidate._stockrl_replay_commit=replay_commit
        consumed=self.replay.acknowledge_training(uses,self.candidate_replay_passes)
        with self.candidate_model_lock:
            self.candidate_version=replay_commit["candidate_version"]
            self.candidate_trained_replay_row_ids=set(trained_replay_row_ids)
            self.candidate_replay_uses={}
            self._atomic_json({"version":2,"candidate_version":self.candidate_version,
                "queue_source":"replay.sqlite3","last_checkpoint_rows":trained_replay_row_ids},
                self.candidate_lineage_path)
            observer=self._new_model_like(candidate,torch.device("cpu"))
            observer.load_state_dict(candidate.state_dict())
            observer.eval(); observer.requires_grad_(False)
            with self.candidate_live_model_lock:
                self.candidate_live_model=observer
                self.candidate_live_model_version=self.candidate_version
        self.metrics["candidate_live_status"]="candidate_updated"
        self.metrics["candidate_has_learning"]=True
        self.metrics["candidate_training"]=False
        if not self.validation_active and not self.stop.is_set():
            self._begin_candidate_validation(candidate)
        # Keep the most recently trained candidate on CPU between update
        # batches. The frozen trial copy remains independent in memory.
        if self.device.type=="cuda": self.candidate.to("cpu")
        self.metrics["candidate_replay_rows_held_for_validation"]=0
        self.metrics["candidate_replay_rows_consumed"]=consumed
        self.metrics["last_candidate_window_forwards"]=self.metrics["candidate_window_forwards_current"]
        self.metrics["candidate_replay_rows_audit_status"]="tracked"
        self.last_train_replay_size=len(self.replay)
        self.metrics["last_train_replay_size"]=self.last_train_replay_size
        self.metrics["paper_experiences_since_candidate"]=max(
            0,int(self.metrics.get("paper_experiences_since_candidate",0))-trigger_experience_count)
        self.metrics["candidate_replay_since_last_update"]=int(
            self.metrics["paper_experiences_since_candidate"])
        self.metrics["candidate_stage"]=("sequential_paper_validation" if self.validation_active else
                                          "waiting_for_replay")
        self.metrics["last_candidate_total_seconds"]=float(time.perf_counter()-training_started)
        self.candidate_retry_attempts=0; self.candidate_retry_after=0.0
        self._write_metrics()
        del candidate,opt

    def _begin_candidate_validation(self,candidate,trained_replay_row_ids=None,start_after=None):
        """Freeze the current candidate in RAM and start a future-only trial."""
        if self.validation_active:
            self.metrics["candidate_validation_deferred"]=True
            return False
        if self.metrics.get("candidate_training"):
            self.metrics["candidate_validation_deferred"]=True
            return False
        if candidate is None:
            return False
        with self.candidate_model_lock:
            if self.metrics.get("candidate_training"):
                self.metrics["candidate_validation_deferred"]=True
                return False
            frozen=self._new_model_like(candidate,torch.device("cpu"))
            frozen.load_state_dict(candidate.state_dict())
            frozen.eval()
            frozen.requires_grad_(False)
            self.validation_candidate=frozen
            if not self.metrics.get("candidate_training") and self.device.type=="cuda":
                candidate.to("cpu")
        self.validation_generation+=1
        generation=self.validation_generation
        self.validation_queue_invalid_reason=None
        self.validation_last_enqueued_minute=None
        self.metrics["promotion_blocked_reason"]=None
        while True:
            try:
                self.validation_queue.get_nowait()
                self.validation_queue.task_done()
            except queue.Empty:
                break
        self.validation_champion_account.reset()
        self.validation_candidate_account.reset()
        cursor_path=self.state_dir/"live_cursor.json"
        cursor_timestamp=None
        try:
            cursor_timestamp=json.loads(cursor_path.read_text(encoding="utf-8")).get("last_timestamp")
        except (OSError,json.JSONDecodeError,AttributeError):
            pass
        candidates=[str(x) for x in (self.current_market_timestamp,
            self.metrics.get("last_market_timestamp"),cursor_timestamp) if x]
        self.validation_start_after=(str(start_after) if start_after is not None
                                     else max(candidates,default=None))
        self.validation_bars=0
        self.validation_active=True
        self.validation_restart_needs_fresh_trial=False
        self.last_validated_candidate_version=self.candidate_version
        self.validation_trained_replay_row_ids=sorted(
            int(x) for x in self.candidate_trained_replay_row_ids)
        source_sha=self._sha256_file(self.champion_path)
        self.validation_source_sha256=source_sha
        self._atomic_json({"status":"collecting","start_after":self.validation_start_after,
                           "bars":0,"source_champion_sha256":source_sha,
                           "source_candidate_version":self.candidate_version,
                           "trained_replay_row_ids":self.validation_trained_replay_row_ids,
                           "same_market_timeline":False,"same_market_input":False,
                           "same_action_rule":False,
                           "comparison_valid":False,"last_decisions":{},
                           "replay_rows_finalized":False,
                           "generation":generation,
                           "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())},
                          self.validation_state_path)
        self.metrics["candidate_validation_active"]=True
        self.metrics["candidate_validation_snapshot_version"]=self.candidate_version
        self.metrics["candidate_validation_snapshot_created_utc"]=time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",time.gmtime())
        self.metrics["candidate_validation_start_after"]=self.validation_start_after
        self.metrics["candidate_validation_deferred"]=False
        # These counters describe the new in-memory trial. Do not surface
        # queue failures or inference totals from a discarded pre-restart run.
        self.metrics["candidate_validation_error"]=None
        self.metrics["candidate_validation_queue_overflows"]=0
        self.metrics["candidate_validation_queue_depth"]=0
        self.metrics["candidate_validation_queue_max_depth"]=0
        self.metrics["candidate_validation_queue_delay_seconds"]=0.0
        self.metrics["candidate_validation_inference_count"]=0
        self.metrics["candidate_validation_inference_seconds_total"]=0.0
        self.metrics["champion_validation_inference_count"]=0
        self.metrics["champion_validation_inference_seconds_total"]=0.0
        self.metrics["candidate_replay_rows_held_for_validation"]=0
        self.metrics["candidate_stage"]="sequential_paper_validation"
        self.metrics["candidate_validation_bars"]=0
        self.metrics["promotion_blocked_reason"]=None
        return True

    @staticmethod
    def _atomic_json(value,path):
        atomic_json(value,path)

    def _collect_candidate_validation(self,panel,index):
        if not self.validation_active or self.validation_candidate is None:
            return
        stamp=str(panel.dates[index])
        if self.validation_start_after and stamp<=str(self.validation_start_after):
            return
        minute=np.datetime64(panel.dates[index],"m")
        if self.validation_start_after and minute<=np.datetime64(self.validation_start_after,"m"):
            return
        if self.validation_last_enqueued_minute is not None and minute<=self.validation_last_enqueued_minute:
            return
        # One observation per market minute; a burst of individual tick stamps
        # must not fill or accelerate the 128-bar portfolio trial.
        snapshot=MarketObservation(panel,index,self.window)
        item=(self.validation_generation,snapshot,len(snapshot.dates)-1,stamp,time.monotonic())
        try:
            self.validation_queue.put_nowait(item)
        except queue.Full:
            self.validation_queue_invalid_reason="검증 대기열이 가득 차 연속 시세를 놓쳤습니다"
            self.metrics["candidate_validation_queue_overflows"]=(
                int(self.metrics.get("candidate_validation_queue_overflows",0))+1)
            return
        self.validation_last_enqueued_minute=minute
        depth=self.validation_queue.qsize()
        self.metrics["candidate_validation_queue_depth"]=depth
        self.metrics["candidate_validation_queue_max_depth"]=max(
            depth,int(self.metrics.get("candidate_validation_queue_max_depth",0)))

    def _validation_worker(self):
        while not self.stop.is_set():
            try:
                generation,panel,index,stamp,queued_at=self.validation_queue.get(timeout=.25)
            except queue.Empty:
                continue
            try:
                if generation!=self.validation_generation or not self.validation_active:
                    continue
                invalid=self.validation_queue_invalid_reason
                delay=time.monotonic()-queued_at
                self.metrics["candidate_validation_queue_delay_seconds"]=delay
                if invalid is None and delay>self.validation_queue_delay_limit_seconds:
                    invalid=(f"검증 처리 지연 {delay:.1f}초가 허용 한도 "
                             f"{self.validation_queue_delay_limit_seconds:.0f}초를 넘었습니다")
                if invalid is not None:
                    self.validation_queue_invalid_reason=None
                    self._finish_candidate_validation(None,None,invalid)
                    continue
                self._process_candidate_validation(panel,index,generation,stamp)
            except Exception as exc:
                if generation==self.validation_generation and self.validation_active:
                    self._finish_candidate_validation(None,None,
                        f"{type(exc).__name__}: {exc}")
            finally:
                self.validation_queue.task_done()
                self.metrics["candidate_validation_queue_depth"]=self.validation_queue.qsize()

    def _process_candidate_validation(self,panel,index,generation,stamp):
        if generation!=self.validation_generation or not self.validation_active or self.validation_candidate is None:
            return
        champ_account=self.validation_champion_account
        cand_account=self.validation_candidate_account
        if champ_account.state.get("last_timestamp")!=cand_account.state.get("last_timestamp"):
            self._finish_candidate_validation(None,None,"paper ledger timestamps diverged")
            return
        state=json.loads(self.validation_state_path.read_text(encoding="utf-8"))
        ledger_timestamp=champ_account.state.get("last_timestamp")
        if state.get("bars")!=self.validation_bars or state.get("last_timestamp")!=ledger_timestamp:
            self._finish_candidate_validation(None,None,"validation state and account timestamps diverged")
            return
        if ledger_timestamp==stamp:
            return
        if ledger_timestamp is not None and stamp<=ledger_timestamp:
            self._finish_candidate_validation(None,None,"validation market timestamps are not increasing")
            return
        completion_scores=None
        last_decisions={}
        try:
            # Both evaluators start from identical seed cash and see this same
            # chronological bar. Existing paper positions and live orders are
            # never used by this isolated comparison.
            champ_account.process_bar(panel,index,True)
            cand_account.process_bar(panel,index,True)
            window=self._window(panel,index)
            args=window
            with self.lock:
                champion=self.champion
            validation_candidate=self.validation_candidate
            accounts=(("champion",champion,champ_account),
                      ("candidate",validation_candidate,cand_account))
            for model_name,model,account in accounts:
                # Candidate observation and frozen candidate validation share
                # one GPU inference slot so they cannot duplicate the 0.5B
                # model allocation at the same time.
                inference_lock=(self.candidate_live_inference_lock
                                if model_name=="candidate" else nullcontext())
                with inference_lock:
                    snapshot_offloaded=(model_name=="candidate" and self.device.type=="cuda"
                                        and next(model.parameters()).device.type=="cpu")
                    if snapshot_offloaded:
                        model.to(self.device)
                    model_device=next(model.parameters()).device
                    pstate,astate=account.model_inputs(panel,index)
                    model_args=[x.to(model_device) for x in args]
                    model_args[0]=model_args[0].to(dtype=next(model.parameters()).dtype)
                    inference_started=time.perf_counter()
                    try:
                        if model_device.type=="cuda": torch.cuda.synchronize(model_device)
                        with torch.inference_mode():
                            if getattr(model,"_stockrl_uses_market_context",False):
                                pt=torch.as_tensor(np.asarray(pstate,dtype=np.float32)[None],device=model_device)
                                at=torch.as_tensor(np.asarray(astate,dtype=np.float32)[None],device=model_device)
                                mt=torch.as_tensor(panel.multiscale_at(index)[None],device=model_device)
                                logits,_,allocation=model(*model_args,portfolio_state=pt,
                                                          account_state=at,return_allocation=True,
                                                          multiscale_state=mt)
                                allocation=allocation[0].float().cpu().numpy()
                            else:
                                logits,_=model(*model_args); allocation=None
                            probabilities=self._account_action_probabilities(
                                logits[0].float().cpu().numpy(),pstate)
                        if model_device.type=="cuda": torch.cuda.synchronize(model_device)
                        elapsed=time.perf_counter()-inference_started
                    finally:
                        if snapshot_offloaded:
                            model.to("cpu")
                prefix=model_name+"_validation_inference_"
                self.metrics[prefix+"count"]=(int(self.metrics.get(prefix+"count",0))+1)
                self.metrics[prefix+"seconds_total"]=(
                    float(self.metrics.get(prefix+"seconds_total",0.0))+elapsed)
                # Promotion compares each frozen model's policy directly;
                # sampled actions and epsilon exploration add avoidable noise.
                actions=self._deterministic_actions(probabilities)
                account.queue_decisions(panel,index,probabilities,True,
                                        allocation=allocation,actions=actions)
                account.save()
                last_decisions[model_name]=[
                    {"symbol":symbol,"action":ACTION_NAMES[action]}
                    for symbol_index,(symbol,action) in enumerate(zip(panel.symbols,actions))
                    if panel.observed[index,symbol_index]
                    and _currency(*panel.groups[symbol]) is not None]
            self.validation_bars+=1
            self.metrics["candidate_validation_bars"]=self.validation_bars
            self.metrics["candidate_skip_reason"]=(
                f"순차 paper 검증 {self.validation_bars}/{self.validation_window_bars}개 bar 대기")
            state.update({"status":"collecting","start_after":self.validation_start_after,
                          "last_timestamp":stamp,"bars":self.validation_bars,
                          "same_market_timeline":True,"same_market_input":True,
                          "same_action_rule":True,
                          "comparison_valid":False,"last_decisions":last_decisions,
                          "source_champion_sha256":self.validation_source_sha256,
                          "generation":generation})
            self._atomic_json(state,self.validation_state_path)
            if self.validation_bars>=self.validation_window_bars:
                # KRW and USD returns are normalized by their matching seed
                # cash, so unrelated currencies are never added as raw money.
                candidate_score=cand_account.normalized_equity()-2.0
                champion_score=champ_account.normalized_equity()-2.0
                completion_scores=(candidate_score,champion_score)
        except Exception as exc:
            self._finish_candidate_validation(None,None,
                f"{type(exc).__name__}: {exc}")
            return
        if completion_scores is not None:
            self._finish_candidate_validation(*completion_scores,None)

    def _finish_candidate_validation(self,candidate_score,champion_score,error):
        candidate=self.validation_candidate
        source_sha=self._sha256_file(self.champion_path)
        state={}
        try: state=json.loads(self.validation_state_path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError): pass
        snapshot_version=int(state.get("source_candidate_version",
                                       self.metrics.get("candidate_validation_snapshot_version",0)))
        trained_replay_row_ids=state.get("trained_replay_row_ids",[])
        if error is None and self.validation_bars<self.validation_window_bars:
            error="sequential paper validation window is incomplete"
        if error is None and state.get("source_champion_sha256")!=source_sha:
            error="champion changed during candidate validation"
        promoted=False
        if error is None:
            try:
                promoted=self._commit_candidate(candidate,float(candidate_score),float(champion_score),
                    self.validation_bars,source_sha,trained_replay_row_ids,
                    candidate_version=snapshot_version,trial_state=state)
            except Exception as exc:
                error=f"{type(exc).__name__}: {exc}"
        if error is not None:
            self.metrics["promotion_blocked_reason"]=error
            self.metrics["candidate_validation_error"]=error
            self.metrics["rejections"]+=1
            from datetime import datetime,timezone
            self.metrics["last_rejection_utc"]=datetime.now(timezone.utc).isoformat()
            self.metrics["last_candidate_promoted"]=False
            self._atomic_json({**state,"status":"rejected","reason":error,
                               "bars":self.validation_bars,
                               "same_market_timeline":False,"same_market_input":False,
                               "same_action_rule":False,
                               "comparison_valid":False,
                               "source_champion_sha256":source_sha,
                               "replay_rows_consumed":0,
                               "replay_rows_finalized":True},self.validation_state_path)
        self.validation_active=False
        self.validation_candidate=None
        self.last_validated_candidate_version=snapshot_version
        self.metrics["candidate_validation_active"]=False
        self.metrics["candidate_validation_bars"]=self.validation_bars
        self.metrics["candidate_replay_rows_held_for_validation"]=0
        self.metrics["candidate_replay_rows_audit_status"]="tracked"
        self.metrics["candidate_stage"]=("training" if self.metrics.get("candidate_training") else "waiting")
        self.metrics["candidate_has_learning"]=(
            self.candidate_version>snapshot_version if promoted else
            bool(self.metrics.get("candidate_has_learning",False)))
        self.metrics["candidate_skip_reason"]=None
        self.validation_generation+=1
        self._write_metrics()
        del candidate
        self._release_cuda_cache()
        if self.candidate is not None and self.candidate_version>snapshot_version:
            self._begin_candidate_validation(self.candidate)

    def _commit_candidate(self,candidate,score_new:float,score_old:float,validation_bars:int,
                          source_champion_sha256:str,trained_replay_row_ids=None,
                          candidate_version:int|None=None,trial_state=None)->bool:
        """Stage a valid champion checkpoint, then atomically swap the reader."""
        promote=(validation_bars>=self.validation_window_bars
                 and self._sha256_file(self.champion_path)==source_champion_sha256
                 and should_promote(score_new,score_old,1e-9))
        self.metrics["last_candidate_validation_score"]=score_new
        self.metrics["last_champion_validation_score"]=score_old
        self.metrics["last_candidate_promoted"]=promote
        if promote:
            staged=self.state_dir/"champion.next"
            save_model(staged,candidate,self.cfg,step=self.steps,temp_dir=self.state_dir)
            promoted_model,_=load_model(staged,self.device)
            promoted_sha=self._sha256_file(staged).upper()
            baseline_next=self.state_dir/"promotion_baseline.next"
            self._atomic_json({"sha256":promoted_sha,"set_reason":"candidate passed 128-bar frozen-snapshot paper gate"},
                              baseline_next)
            with self.lock:
                if self._sha256_file(self.champion_path)!=source_champion_sha256:
                    raise RuntimeError("champion changed before atomic promotion")
                os.replace(staged,self.champion_path)
                os.replace(baseline_next,self.promotion_baseline_path)
                self.promotion_baseline_sha256=promoted_sha
                self.champion=promoted_model.eval(); self.metrics["promotions"]+=1
                from datetime import datetime, timezone
                self.metrics["last_promotion_utc"]=datetime.now(timezone.utc).isoformat()
        else:
            self.metrics["rejections"]+=1
            from datetime import datetime, timezone
            self.metrics["last_rejection_utc"]=datetime.now(timezone.utc).isoformat()
        self.metrics["promotion_blocked_reason"]=None
        history=self.metrics.setdefault("candidate_gate_history",[])
        reason=("sequential paper-account net return improved" if promote else
                "champion changed during comparison" if self._sha256_file(self.champion_path)!=source_champion_sha256 else
                "candidate paper-account net return did not beat champion")
        history.append({"time_utc":self.metrics["last_promotion_utc" if promote else "last_rejection_utc"],
                        "applied":promote,"reason":reason,
                        "candidate_score":float(score_new),"champion_score":float(score_old)})
        self.metrics["candidate_gate_history"]=history[-20:]
        self._atomic_json({"status":"promoted" if promote else "rejected",
                           "bars":validation_bars,"candidate_score":float(score_new),
                           "champion_score":float(score_old),"applied":promote,
                           "source_candidate_version":candidate_version,
                           "start_after":(trial_state or {}).get("start_after"),
                           "last_timestamp":(trial_state or {}).get("last_timestamp"),
                           "last_decisions":(trial_state or {}).get("last_decisions",{}),
                           "same_market_timeline":bool(validation_bars>=self.validation_window_bars),
                           "same_market_input":bool(validation_bars>=self.validation_window_bars),
                           "same_action_rule":bool(validation_bars>=self.validation_window_bars),
                           "comparison_valid":bool(validation_bars>=self.validation_window_bars),
                           "trained_replay_row_ids":sorted(set(trained_replay_row_ids or [])),
                           "replay_rows_consumed":0,
                           "replay_rows_finalized":True,
                           "source_champion_sha256":source_champion_sha256,
                           "result_champion_sha256":self._sha256_file(self.champion_path),
                           "reason":reason},self.validation_state_path)
        return promote

    def _write_metrics(self):
        proc=psutil.Process(); metrics=dict(self.metrics)
        replay_stats=self.replay.stats(self.candidate_replay_passes)
        validation_snapshot_bytes=(sum(p.numel()*p.element_size()
            for p in self.validation_candidate.parameters()) if self.validation_candidate is not None else 0)
        metrics.update({"parameters":parameter_count(self.champion),"device":str(self.device),
          "replay_persistence":"durable_fifo_shared_frames_sqlite",
          "learning_priority":"complete_daily_experience_coverage",
          "replay_untrained_count":replay_stats["untrained"],
          "candidate_eligible_replay_count":replay_stats["eligible"],
          "candidate_untrained_replay_count":replay_stats["untrained"],
          "replay_quarantined_count":replay_stats["quarantined"],
          "replay_unsupported_count":replay_stats["unsupported"],
          "replay_oldest_unfinished_timestamp":replay_stats.get("oldest"),
          "replay_storage_pressure":replay_stats.get("storage_pressure",False),
          "replay_max_bytes":None,
          "replay_storage_warning_bytes":self.replay.storage_warning_bytes,
          "replay_size_limit_enabled":False,
          "daily_learning":replay_stats["daily"],
          "daily_learning_timezone":"Asia/Seoul",
          "replay_eviction_enabled":False,
          "candidate_replay_passes":self.candidate_replay_passes,
          "promotion_score_mode":"sequential_paper_account_net_return",
          "promotion_uses_sequential_paper_account":True,
          "promotion_gate_ready":not bool(metrics.get("promotion_blocked_reason")),
          "promotion_blocked_reason":metrics.get("promotion_blocked_reason"),
          "promotion_baseline_sha256":self.promotion_baseline_sha256,
          "candidate_learning_enabled":not self.stop.is_set(),
          "candidate_start_ready":bool(replay_stats["eligible"]),
          "candidate_every":self.candidate_interval,
          "candidate_min_replay":1,
          "candidate_batch_size":self.batch_size,
          "candidate_optimizer_steps_target":self.updates_per_candidate,
          "candidate_samples_target":self.updates_per_candidate*self.batch_size,
          "candidate_min_validation_dates":self.validation_window_bars,
          "candidate_validation_bars":self.validation_bars,
          "candidate_validation_queue_depth":self.validation_queue.qsize(),
          "candidate_validation_queue_capacity":self.validation_queue.maxsize,
          "candidate_validation_queue_delay_limit_seconds":self.validation_queue_delay_limit_seconds,
          "candidate_replay_since_last_update":int(self.metrics.get("candidate_replay_since_last_update",0)),
          "last_train_replay_size":self.last_train_replay_size,
          "candidate_stage":("training_and_validation" if self.validation_active and
                             self.metrics.get("candidate_training") else
                             "sequential_paper_validation" if self.validation_active else
                             "training" if self.metrics.get("candidate_training") else "waiting"),
          "candidate_validation_active":bool(self.validation_active),
          "candidate_validation_snapshot_version":self.metrics.get("candidate_validation_snapshot_version"),
          "candidate_validation_snapshot_device":(
              str(next(self.validation_candidate.parameters()).device)
              if self.validation_candidate is not None else None),
          "candidate_validation_snapshot_parameter_bytes":validation_snapshot_bytes,
          "candidate_model_version":self.candidate_version,
          "candidate_has_learning":bool(self.metrics.get("candidate_has_learning",False)),
          "champion_model_parameter_bytes":sum(p.numel()*p.element_size() for p in self.champion.parameters()),
          "last_candidate_peak_extra_allocated_bytes":(
              max(0,int(self.metrics["last_candidate_peak_allocated_bytes"])-
                      int(self.metrics["last_candidate_baseline_allocated_bytes"]))
              if self.metrics.get("last_candidate_peak_allocated_bytes") is not None and
                 self.metrics.get("last_candidate_baseline_allocated_bytes") is not None else None),
          "validation_window_dates":len(self.validation_dates),
          "portfolio_validation_window_dates":len(self.portfolio_validation_dates),
          "horizon":str(self.horizon),"fee_rate":self.fee,"slippage_bps":self.slippage*10000,
          "rss_bytes":proc.memory_info().rss,"replay_count":len(self.replay),
          "trainable_replay_count":self.replay.trainable_count(),
          "replay_file_bytes":self.replay.disk_bytes(),
          "champion_path":str(self.champion_path),"candidate_path":str(self.model_dir/"candidate.pt")})
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
        # Do not rewrite champion on shutdown. Only the promotion gate may replace it.
        self._write_metrics()


def benchmark_model(model, panel, window=128, repeats=3, device=None):
    from .core import device_for
    dev=device or device_for(); model.to(dev)
    contextual=getattr(model,"_stockrl_uses_market_context",False)
    if dev.type=="cuda":
        # The large backbone uses FP16 while context adapters consume FP32.
        (model.backbone if contextual else model).half()
    model.eval()
    end_index=min(len(panel.dates)-1,window)
    args=[x.to(dev) for x in panel.window(end_index,window,include_context=contextual)]
    args[0]=args[0].to(dtype=next(model.parameters()).dtype)
    extra=({"multiscale_state":torch.as_tensor(panel.multiscale_at(end_index)[None],device=dev)}
           if contextual else {})
    samples=[]
    with torch.inference_mode():
        model(*args,**extra)
        if dev.type=="cuda": torch.cuda.synchronize(dev)
        for _ in range(repeats):
            if dev.type=="cuda": torch.cuda.synchronize(dev)
            t=time.perf_counter(); model(*args,**extra); samples.append(time.perf_counter()-t)
            if dev.type=="cuda":
                torch.cuda.synchronize(dev); samples[-1]=time.perf_counter()-t
    return {"inference_seconds_p50":float(np.percentile(samples,50)),"inference_seconds_p95":float(np.percentile(samples,95)),
            "inference_repeats":repeats}
