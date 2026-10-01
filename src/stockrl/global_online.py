"""Asynchronous online actor-critic over global asset panels.

Market observation, both learners, and frozen-snapshot validation use separate
model state. Each learner publishes saved updates; candidate promotion requires
a future paper-account score improvement against the frozen champion baseline.
"""
from __future__ import annotations

from collections import deque, OrderedDict
from contextlib import closing, nullcontext, contextmanager
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
from .account_diagnostics import summarize_policy
from .operating_rules import operating_rules
from .state_io import atomic_json
from .multiscale import (MULTISCALE_FEATURE_COUNT, TIMEFRAME_NAMES,
                         TIMEFRAME_FEATURE_NAMES, MULTISCALE_FEATURE_ORDER,
                         BASE_MULTISCALE_FEATURE_COUNT, LONG_CONTEXT_NAMES)

REWARD_VERSION="symbol_and_portfolio_v5"
PAPER_EXPLORATION_EPSILON=0.05
ONLINE_TRAINABLE_BLOCKS=4


class TrainingMetrics:
    """Keep each learner's measurements in the same runtime status object."""
    def __init__(self,store,learner):
        self.store=store; self.learner=learner
    def key(self,key):
        if self.learner=="candidate": return key
        if key.startswith("candidate_"): return "champion_"+key[len("candidate_"):]
        if key.startswith("last_candidate_"): return "last_champion_"+key[len("last_candidate_"):]
        if key in ("weight_delta_l1","paper_examples_trained","teacher_examples_trained"):
            return "champion_"+key
        return key
    def __getitem__(self,key): return self.store[self.key(key)]
    def __setitem__(self,key,value): self.store[self.key(key)]=value
    def get(self,key,default=None): return self.store.get(self.key(key),default)
    def setdefault(self,key,default=None): return self.store.setdefault(self.key(key),default)
REWARD_DEFINITION=("symbol_and_portfolio_v5: net-return percentage points; new paper decisions "
                   "use cumulative executed outcomes across the configured credit observations plus "
                   "a detached successor value, ending at account reset; legacy short outcomes remain "
                   "learnable; per-symbol contribution and whole-account result stay separate")


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
    origin_model: str = "champion"
    daily_history: np.ndarray | None = None
    credit_observations: int = 0
    bootstrap_window_key: str | None = None
    bootstrap_symbol_index: int | None = None
    bootstrap_discount: float = 0.0
    goal_state: np.ndarray | None = None
    goal_reward_points: float = 0.0
    portfolio_goal_reward_points: float = 0.0
    goal_terminal: bool = False
    goal_episode_id: str | None = None


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
                and candidate_score>0 and candidate_score>champion_score+minimum_delta)


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
        payload["multiscale_feature_order"]=list(MULTISCALE_FEATURE_ORDER)
    _atomic_save(payload,path,temp_dir=temp_dir)


def load_model(path:Path,device:torch.device,instrument_config:Path|None=None):
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
        multiscale_order=list(MULTISCALE_FEATURE_ORDER)
        saved_order=ckpt.get("multiscale_feature_order",multiscale_order)
        if saved_order!=multiscale_order[:len(saved_order)]:
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
    if instrument_config is not None:
        from .instrument_ids import extend_live_symbols
        instruments=json.loads(Path(instrument_config).read_text(encoding="utf-8"))["instruments"]
        cfg=extend_live_symbols(model,cfg,instruments)
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
        history=(panel.daily_history_at(index) if hasattr(panel,"daily_history_at") else None)
        self.daily_history=history.copy() if history is not None else None

    def multiscale_at(self,index):
        if index!=len(self.dates)-1:
            raise ValueError("observation has only its captured multiscale state")
        return self.multiscale

    def daily_history_at(self,index):
        return self.daily_history


class OnlineGlobalAgent:
    @staticmethod
    def _goal_kwargs(account,device):
        goal=account.goal_inputs()
        return {"goal_state":torch.as_tensor(np.asarray(goal)[None],device=device,dtype=torch.float32)} if goal is not None else {}

    @staticmethod
    def _saved_goal_kwargs(experience,device):
        goal=getattr(experience,"goal_state",None)
        return {"goal_state":torch.as_tensor(goal[None],device=device,dtype=torch.float32)} if goal is not None else {}

    @staticmethod
    def _daily_history_kwargs(panel,index,device):
        history=panel.daily_history_at(index) if hasattr(panel,"daily_history_at") else None
        return {"daily_history":torch.as_tensor(history[None],device=device,dtype=torch.float32)} if history is not None else {}

    def __init__(self, state_dir: str|Path, device="auto", config:TransformerConfig|None=None,
                 capacity=4_096, window=128, horizon=1, fee=.001, slippage_bps=1.0,
                 min_replay=8, batch_size=4, updates_per_candidate=8, lr=2e-6, seed=7,
                 candidate_interval=16, initial_champion: str|Path|None=None,
                 teacher_replay_path: str|Path|None=None,
                 model_dir: str|Path|None=None,instrument_config:str|Path|None=None):
        from .core import device_for
        from .paths import ensure_project_path, validate_model_dir
        self.state_dir=ensure_project_path(state_dir, "runtime")
        self.instrument_config=ensure_project_path(instrument_config,"instrument config") if instrument_config else None
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
        self.dual_learning_enabled=True
        self.champion_training_version=0
        self.validation_champion=None
        self.validation_promotion_epoch=0
        self.next_learning_role="champion"
        self.replay=GlobalReplayBuffer(capacity,seed,self.state_dir/"replay.sqlite3",dual_learning=True)
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
            self.champion,self.cfg=load_model(self.champion_path,self.device,self.instrument_config)
        else:
            if model_dir is not None or initial_champion is not None:
                raise FileNotFoundError(f"configured champion checkpoint does not exist: {self.champion_path}")
            self.champion=GlobalMarketTransformer(self.cfg)
            if self.device.type=="cuda": self.champion=self.champion.half()
            self.champion=self.champion.to(self.device).eval()
            save_model(self.champion_path,self.champion,self.cfg,temp_dir=self.state_dir)
        self.candidate=None; self.validation_candidate=None; self.optimizer=None; self.steps=0; self.updates=0
        self.candidate_model_lock=threading.RLock()
        from .gpu_scheduler import FairGpuScheduler
        self.candidate_live_inference_lock=FairGpuScheduler(
            preopen_learning=operating_rules().get("preopen_learning_priority",False))
        self.candidate_live_model_lock=threading.Lock()
        self.candidate_live_model=None
        self.candidate_live_model_version=None
        self.candidate_live_queue=queue.Queue(maxsize=2)
        self.candidate_version=0
        self.last_validated_candidate_version=-1
        self.validation_restart_needs_fresh_trial=False
        self.candidate_trained_replay_row_ids=set()
        self.candidate_replay_passes=1
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
        recovered=self.replay.observer_account()
        if recovered:
            self.candidate_live_account.state=recovered
        self.candidate_portfolio_pending=self.replay.load_pending("candidate_portfolio")
        self.candidate_live_state_path=self.state_dir/"candidate_observer_state.json"
        self.candidate_live_thread=threading.Thread(
            target=self._candidate_live_worker,name="candidate-live-observer",daemon=True)
        self.operating_rules=operating_rules()
        for account in (self.paper_account,self.candidate_live_account):
            account.configure_goal(self.operating_rules["goal_target_multiple"],self.operating_rules["goal_win_bonus_points"])
        self.reward_credit_observations=int(self.operating_rules.get("reward_credit_observations",60))
        self.reward_credit_seconds=int(self.operating_rules.get("reward_credit_seconds",3600))
        self.validation_window_bars=int(self.operating_rules["validation_min_market_minutes"])
        self.daily_promotion=self.operating_rules["promotion_schedule"]=="daily"
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
        for account in (self.validation_champion_account,self.validation_candidate_account):
            account.configure_goal(self.operating_rules["goal_target_multiple"],self.operating_rules["goal_win_bonus_points"])
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
        self.metrics["runtime_code_version"]="dual-model-single-pass-learning-20261001"
        self.metrics["champion_training"]=False
        for key in ("champion_paper_examples_trained","champion_teacher_examples_trained"):
            self.metrics.setdefault(key,0)
        self.metrics.setdefault("champion_weight_delta_l1",[])
        champion_commit=getattr(self.champion,"_stockrl_replay_commit",{})
        if champion_commit.get("learner")=="champion":
            self.replay.acknowledge_training(champion_commit.get("uses",{}),self.candidate_replay_passes,learner="champion")
        self.champion_training_version=max(int(self.metrics.get("champion_training_version",0)),
            int(champion_commit.get("model_version",0)) if champion_commit.get("learner")=="champion" else 0)
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
            self.candidate,candidate_cfg=load_model(candidate_path,self.device,self.instrument_config)
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
            self._publish_candidate_observer(self.candidate,self.candidate_version)
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
        self.metrics["replay_completed_on_policy_change"]=self.replay.finalize_completed(self.candidate_replay_passes)
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
        # Every caller immediately loads an existing state_dict. Allocate its
        # final storage directly instead of initializing 0.5B random FP32
        # parameters and then converting/discarding them before that copy.
        with torch.device("meta"):
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
        return model.to_empty(device=device)

    def _publish_candidate_observer(self, source, version):
        """Atomically publish completed weights into reusable observer storage."""
        started=time.perf_counter()
        with self.candidate_live_model_lock:
            observer=self.candidate_live_model
            reused=observer is not None
            if observer is None:
                observer=self._new_model_like(source,torch.device("cpu"))
            with self._gpu_work("candidate_publish"):
                observer.load_state_dict(source.state_dict())
            observer.eval(); observer.requires_grad_(False)
            encoder=getattr(observer,"daily_history_encoder",None)
            if encoder is not None: encoder.cache=None
            self.candidate_live_model=observer
            self.candidate_live_model_version=version
        self.metrics["candidate_observer_publish_seconds"]=time.perf_counter()-started
        self.metrics["candidate_observer_storage_reused"]=reused

    def _restore_candidate_checkpoint(self):
        path=self.model_dir/"candidate.pt"
        if not path.is_file():
            with self.lock:
                source=self.champion
                self.candidate=self._new_model_like(source,self.device)
                self.candidate.load_state_dict(source.state_dict())
        else:
            candidate,cfg=load_model(path,self.device,self.instrument_config)
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
        snapshot=MarketObservation(panel,index,self.window)
        self.replay.enqueue_market_observation(snapshot,bool(paper_enabled),tuple(float(x) for x in uniforms))
        return snapshot

    def _gpu_work(self,role):
        from contextlib import nullcontext
        scheduler=getattr(self,"candidate_live_inference_lock",None)
        if scheduler is None:
            return nullcontext()
        return scheduler.work(role) if hasattr(scheduler,"work") else scheduler

    def _candidate_live_worker(self):
        """Run an independent observational paper account for current candidate weights."""
        while not self.stop.is_set() or self.replay.market_observation_stats()["pending"]:
            observation_started=time.perf_counter()
            item=self.replay.next_market_observation()
            observation_profile={"read_seconds":time.perf_counter()-observation_started}
            if item is None:
                self.stop.wait(.25)
                continue
            stamp,data=item
            panel=MarketObservation.__new__(MarketObservation)
            panel.__dict__.update(data)
            index=len(panel.dates)-1
            paper_enabled=bool(data["paper_enabled"]);uniforms=data["uniforms"]
            if self.candidate_live_model is None:
                if self.stop.is_set():
                    break  # Required observations remain in SQLite for restart.
                self.stop.wait(.25)
                continue
            saved_account=pickle.loads(pickle.dumps(self.candidate_live_account.state,protocol=5))
            committed=False
            try:
                phase_started=time.perf_counter()
                account=self.candidate_live_account
                filled_orders=account.process_bar(panel,index,paper_enabled)
                self.candidate_portfolio_pending=self._mature_portfolio(
                    self.candidate_portfolio_pending,panel,index,filled_orders,account=account,origin_model="candidate")
                observation_profile["fills_and_rewards_seconds"]=time.perf_counter()-phase_started
                if self.candidate_live_model is None:
                    self.metrics["candidate_live_status"]="waiting_for_candidate_update"
                    account.save()
                    self._atomic_json({"status":self.metrics["candidate_live_status"],
                        "last_timestamp":stamp,"candidate_version":self.candidate_version,
                        "last_decisions":[]},self.candidate_live_state_path)
                    self.metrics["candidate_live_last_timestamp"]=stamp
                    continue
                phase_started=time.perf_counter()
                pstate,astate=account.model_inputs(panel,index)
                window=self._window(panel,index)
                observation_profile["input_seconds"]=time.perf_counter()-phase_started
                started=time.perf_counter()
                profile={}
                with self.candidate_live_model_lock:
                    model=self.candidate_live_model
                    if model is None: raise RuntimeError("candidate observer snapshot is not loaded")
                    observer_version=self.candidate_live_model_version
                    originally_offloaded=(self.device.type=="cuda" and
                        next(model.parameters()).device.type=="cpu")
                    with self._gpu_work("candidate_live"):
                        profile["wait_seconds"]=time.perf_counter()-started
                        phase_started=time.perf_counter()
                        if originally_offloaded: model.to(self.device)
                        profile["upload_seconds"]=time.perf_counter()-phase_started
                        device=next(model.parameters()).device
                        was_training=model.training
                        model.eval()
                        try:
                            phase_started=time.perf_counter()
                            args=[value.to(device) for value in window]
                            args[0]=args[0].to(dtype=next(model.parameters()).dtype)
                            with torch.inference_mode():
                                if getattr(model,"_stockrl_uses_market_context",False):
                                    pt=torch.as_tensor(np.asarray(pstate,dtype=np.float32)[None],device=device)
                                    at=torch.as_tensor(np.asarray(astate,dtype=np.float32)[None],device=device)
                                    mt=torch.as_tensor(panel.multiscale_at(index)[None],device=device)
                                    logits,_,allocation=model(*args,portfolio_state=pt,
                                        account_state=at,return_allocation=True,
                                        multiscale_state=mt,**self._daily_history_kwargs(panel,index,device),
                                        **self._goal_kwargs(account,device))
                                    allocation=allocation[0].float().cpu().numpy()
                                else:
                                    logits,_=model(*args); allocation=None
                            if device.type=="cuda": torch.cuda.synchronize(device)
                            # Wall time includes input preparation and waiting
                            # for shared CUDA work; this is not exclusive GPU time.
                            profile["forward_seconds"]=time.perf_counter()-phase_started
                        finally:
                            model.train(was_training)
                            phase_started=time.perf_counter()
                            if originally_offloaded: model.to("cpu")
                            profile["download_seconds"]=time.perf_counter()-phase_started
                probabilities,actions=self._paper_policy_actions(
                    logits[0].float().cpu().numpy(),pstate,uniforms)
                elapsed=time.perf_counter()-started
                profile["total_seconds"]=elapsed
                self.metrics["candidate_live_inference_profile"]=profile
                observation_profile["inference_seconds"]=elapsed
                phase_started=time.perf_counter()
                submitted=account.queue_decisions(panel,index,probabilities,paper_enabled,
                    allocation=allocation,actions=actions)
                x,sid,mid,aid,mask=window[:5]
                inputs={"features":x[0].numpy().astype(np.float16),"symbol_ids":sid[0].numpy(),
                    "goal_state":(np.asarray(account.goal_inputs(),dtype=np.float32) if account.goal_inputs() is not None else None),
                    "market_ids":mid[0].numpy(),"asset_ids":aid[0].numpy(),"valid_mask":mask[0].numpy(),
                    "market_context":window[5][0].numpy().astype(np.float16) if len(window)>5 else None,
                    "multiscale_state":panel.multiscale_at(index).astype(np.float16),
                    "daily_history":getattr(panel,"daily_history",None),
                    "input_symbols":list(panel.symbols),"portfolio_state":np.asarray(pstate,dtype=np.float16),
                    "account_state":np.asarray(astate,dtype=np.float32)}
                goal_before=account.goal_points();goal_complete=account.goal_summary().get("status")=="WIN"
                for j,symbol in enumerate(panel.symbols):
                    # Padding has a model ID but no traded instrument metadata.
                    # Check the observation mask before looking up its group.
                    if not paper_enabled or not panel.observed[index,j]:
                        continue
                    market,asset=panel.groups[symbol]
                    if _currency(market,asset) is not None:
                        action=int(actions[j]);decision_id=f"{stamp}|{symbol}"
                        self.candidate_portfolio_pending.append({**inputs,"index":index,"symbol_index":j,
                            "symbol":symbol,"action":action,"timestamp":stamp,"decision_id":decision_id,
                            "reward_version":REWARD_VERSION,"origin_model":"candidate",
                            "credit_observations_target":getattr(self,"reward_credit_observations",60),
                            "credit_seconds":getattr(self,"reward_credit_seconds",3600),
                            "behavior_log_prob":float(np.log(max(float(probabilities[j,action]),1e-12))),
                            "entry_price":float(panel.closes[index,j]),"bars_elapsed":0,
                            "regime":abs(float(panel.features[index,j,6])),"is_validation":False,
                            "equity_before":account.normalized_equity(),
                            "goal_points_before":goal_before,"goal_episode_id":account.state["episode_id"],
                            "goal_complete_before":goal_complete,
                            "goal_weight_before":max(float(pstate[j][1]),float(allocation[j]) if action==2 and allocation is not None else 0.0),
                            "symbol_pnl_before":account.symbol_net_pnl(symbol),
                            "fill_expected":decision_id in (submitted or set())})
                observation_profile["orders_and_pending_seconds"]=time.perf_counter()-phase_started
                phase_started=time.perf_counter()
                self.replay.commit_observer(stamp,account.state,self.candidate_portfolio_pending)
                observation_profile["database_commit_seconds"]=time.perf_counter()-phase_started
                committed=True
                policy_diagnostics=summarize_policy(panel,index,probabilities,actions,
                    allocation,account,submitted or set(),stamp)
                if policy_diagnostics["observed_tradable_symbols"]:
                    self.metrics["candidate_last_tradable_policy_diagnostics"]=policy_diagnostics
                phase_started=time.perf_counter()
                account.save()
                observation_profile["account_save_seconds"]=time.perf_counter()-phase_started
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
                observation_profile["total_seconds"]=time.perf_counter()-observation_started
                self.metrics["candidate_live_observation_profile"]=observation_profile
                self._atomic_json({"status":self.metrics["candidate_live_status"],
                    "last_timestamp":stamp,"candidate_version":observer_version,
                    "policy_mode":"same_epsilon_sampling_and_random_draws_as_champion",
                    "candidate_training":bool(self.metrics.get("candidate_training")),
                    "champion_training":bool(self.metrics.get("champion_training")),
                    "learning_wait_reason":self.metrics.get("learning_wait_reason"),
                    "learning_live_priority_enabled":bool(getattr(self,"live_priority_enabled",False)),
                    "gpu_scheduler":(self.candidate_live_inference_lock.snapshot()
                        if hasattr(self.candidate_live_inference_lock,"snapshot") else {}),
                    "last_inference_seconds":elapsed,"inference_profile":profile,
                    "observation_profile":observation_profile,"inference_count":self.metrics[
                        "candidate_live_inference_count"],"last_decisions":decisions,
                    "policy_diagnostics":policy_diagnostics,
                    "last_tradable_policy_diagnostics":self.metrics.get(
                        "candidate_last_tradable_policy_diagnostics",{})},
                    self.candidate_live_state_path)
            except Exception as exc:
                self.candidate_live_account.state=self.replay.observer_account() if committed else saved_account
                self.candidate_portfolio_pending=self.replay.load_pending("candidate_portfolio")
                self.metrics["candidate_live_errors"]=(
                    int(self.metrics.get("candidate_live_errors",0))+1)
                self.metrics["candidate_live_status"]="error"
                self.metrics["candidate_live_error"]=f"{type(exc).__name__}: {exc}"[:500]
                self._atomic_json({"status":"error","last_timestamp":stamp,
                    "candidate_version":self.candidate_live_model_version,
                    "error":self.metrics["candidate_live_error"],"last_decisions":[]},
                    self.candidate_live_state_path)
                if self.stop.is_set():
                    break  # Preserve the failed observation instead of spinning during shutdown.
                self.stop.wait(1.0)

    def _infer(self,panel,index,portfolio_state=None,account_state=None):
        requested=time.perf_counter()
        window=self._window(panel,index)
        with self.lock:
            model=self.champion
        with self._gpu_work("champion_live"):
            acquired=time.perf_counter()
            args=[x.to(self.device) for x in window]
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
                                                     multiscale_state=mstate,**self._daily_history_kwargs(panel,index,self.device),
                                                     **self._goal_kwargs(self.paper_account,self.device))
                else:
                    logits,values=model(*args); allocation=None
                elapsed=time.perf_counter()-t
                if self.device.type=="cuda":
                    torch.cuda.synchronize(self.device); elapsed=time.perf_counter()-t
            result=(logits[0].cpu().numpy(),values[0].cpu().numpy(),
                    allocation[0].cpu().float().numpy() if allocation is not None else None)
        self.metrics["inference_seconds"].append(elapsed)
        self.metrics["champion_live_inference_count"]=(
            int(self.metrics.get("champion_live_inference_count",0))+1)
        self.metrics["champion_live_inference_seconds_total"]=(
            float(self.metrics.get("champion_live_inference_seconds_total",0.0))+elapsed)
        self.metrics["champion_live_last_inference_seconds"]=elapsed
        if self.metrics.get("candidate_training"):
            self.metrics["inference_during_candidate"]=int(self.metrics.get("inference_during_candidate",0))+1
        if len(self.metrics["inference_seconds"])>2000: self.metrics["inference_seconds"]=self.metrics["inference_seconds"][-2000:]
        self.metrics["champion_live_inference_profile"]={"wait_seconds":acquired-requested,
            "forward_seconds":elapsed,"total_seconds":time.perf_counter()-requested}
        return result

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

    def _mature_portfolio(self, pending, panel, end_index: int, filled_orders=(),account=None,origin_model="champion"):
        """Turn completed paper-account transitions into reward experiences.

        The reward uses the account's net-of-cost normalized equity change.
        Trade actions begin their horizon on the linked next-bar fill; HOLD
        observations use their decision time because they create no order.
        """
        account=account or self.paper_account
        account.observe_goal(str(panel.dates[end_index]))
        self.metrics[origin_model+"_goal"]=account.goal_summary()
        pending_kind="portfolio" if origin_model=="champion" else "candidate_portfolio"
        score=self.replay.record_account_score(origin_model,str(panel.dates[end_index]),
            account.state.get("episode_id","legacy"),account.reward_points())
        self.metrics[origin_model+"_reward_score"]=score
        if not pending:
            return pending
        now = float(account.normalized_equity())
        fills_by_id={str(fill.get("decision_id")):fill for fill in filled_orders
                     if fill.get("decision_id")}
        fills_by_order={(str(fill.get("order_date")),str(fill.get("symbol"))):fill
                        for fill in filled_orders if fill.get("order_date")}
        keep=[]; account_transition_added=set(); credit_successor=None
        matured_experiences=[]; matured_acks=[]
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
            reset_terminal=bool(dec.get("reset_terminal"))
            goal_points=dec.get("reset_goal_points",account.goal_points()) if reset_terminal else account.goal_points()
            goal_before=dec.get("goal_points_before",{})
            same_goal_episode=reset_terminal or dec.get("goal_episode_id")==account.state.get("episode_id")
            goal_terminal=bool(goal_before and same_goal_episode and not dec.get("goal_complete_before")
                and all(goal_points.get(c,0)>0 for c in goal_points))
            terminal=reset_terminal or goal_terminal
            if not terminal and (current<=stamp or not panel.observed[end_index,symbol_ix]):
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
                    if not dec.get("credit_observations_target"):
                        dec["symbol_pnl_before"]=float(
                            fill.get("symbol_pnl_before_fill",dec.get("symbol_pnl_before",0.0)))
                    dec["bars_elapsed"]=0
                    keep.append(dec)
                    continue
                queued=account.state.get("pending",{}).get(symbol)
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
            if current<=reward_start and not terminal:
                keep.append(dec); continue
            if dec.get("credit_observations_target"):
                if str(current)<=dec.get("credit_last_timestamp",str(stamp)) and not terminal:
                    keep.append(dec); continue
                dec["credit_last_timestamp"]=str(current)
                dec["credit_observations_elapsed"]=int(dec.get("credit_observations_elapsed",0))+1
                matured=(current>=reward_start+np.timedelta64(int(dec["credit_seconds"]),"s")
                         if dec.get("credit_seconds") else
                         dec["credit_observations_elapsed"]>=int(dec["credit_observations_target"]))
            elif self.horizon_kind=="bars":
                dec["bars_elapsed"]=int(dec.get("bars_elapsed",0))+1
                matured=dec["bars_elapsed"]>=self.horizon_amount
            else:
                matured=current>=reward_start+np.timedelta64(self.horizon_amount,"s")
            if not matured and not terminal:
                keep.append(dec); continue
            final_equity=float(dec.get("reset_equity",now)) if terminal else now
            account_reward = final_equity - float(dec.get("equity_before", final_equity))
            symbol_reward = ((float(dec["reset_symbol_net_pnl"]) if reset_terminal else account.symbol_net_pnl(symbol))
                             - float(dec.get("symbol_pnl_before", 0.0)))
            if dec.get("trade_executed") is False:
                symbol_reward=0.0
            if dec.get("promotion_holdout",False):
                self.metrics["promotion_validation_outcomes"] = int(
                    self.metrics.get("promotion_validation_outcomes",0))+1
                self.replay.acknowledge_pending(pending_kind,dec)
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
                    daily_history=dec.get("daily_history"),
                    portfolio_state=dec.get("portfolio_state"),
                    account_state=dec.get("account_state"),forward_return=forward_return,
                    behavior_log_prob=dec.get("behavior_log_prob"),
                    trade_executed=dec.get("trade_executed",True))
            if dec.get("is_validation",False):
                self._append_validation(self.portfolio_validation,self.portfolio_validation_dates,
                                        exp.timestamp,exp)
                self.replay.acknowledge_pending(pending_kind,dec)
            else:
                timestamp=dec["timestamp"]
                # Keep one allocation-credit row per symbol. It carries that
                # symbol's realized contribution separately from the shared
                # whole-account result; never attach the whole account return
                # to whichever symbol happened to mature first.
                first_account_transition=(timestamp not in account_transition_added and
                    not self.replay.has_portfolio_value(timestamp,origin_model))
                account_exp=Experience(dec["features"],dec["symbol_ids"],dec["market_ids"],dec["asset_ids"],
                    dec["valid_mask"],input_symbol_index,int(dec["action"]),float(symbol_reward),timestamp,
                    "paper_account_portfolio",float(dec.get("regime",0.0)),
                    reward_version=REWARD_VERSION,
                    market_context=dec.get("market_context"),
                    multiscale_state=dec.get("multiscale_state"),
                    daily_history=dec.get("daily_history"),
                    portfolio_state=dec.get("portfolio_state"),
                    account_state=dec.get("account_state"),portfolio_reward=float(account_reward),
                    portfolio_transition=True,portfolio_value_transition=first_account_transition,
                    forward_return=forward_return,
                    behavior_log_prob=dec.get("behavior_log_prob"),
                    trade_executed=dec.get("trade_executed",True))
                exp.origin_model=origin_model;account_exp.origin_model=origin_model
                if goal_before and same_goal_episode:
                    currency=_currency(*panel.groups[symbol])
                    deltas={c:max(0.0,float(goal_points.get(c,0))-float(goal_before.get(c,0))) for c in goal_points}
                    contribution=deltas.get(currency,0.0)*min(1.0,max(0.0,float(dec.get("goal_weight_before",0.0))))
                    for experience in (exp,account_exp):
                        experience.goal_state=dec.get("goal_state")
                        experience.goal_reward_points=contribution if experience.trade_executed else 0.0
                        experience.goal_episode_id=dec.get("goal_episode_id")
                        experience.goal_terminal=goal_terminal
                    account_exp.portfolio_goal_reward_points=sum(deltas.values())
                if dec.get("credit_observations_target"):
                    for experience in (exp,account_exp):
                        experience.credit_observations=int(dec.get("credit_observations_elapsed",0))
                        if not terminal:
                            experience.bootstrap_discount=1.0
                            experience.bootstrap_symbol_index=symbol_ix
                    if not terminal:
                        # One shared successor input for all matured decisions at
                        # this market/account state. It is persisted in the same DB.
                        if credit_successor is None:
                            args=self._window(panel,end_index)
                            pstate,astate=account.model_inputs(panel,end_index)
                            credit_successor=Experience(args[0][0].numpy(),args[1][0].numpy(),
                                args[2][0].numpy(),args[3][0].numpy(),args[4][0].numpy(),0,1,0.0,
                                str(current),market_context=(args[5][0].numpy() if len(args)>5 else None),
                                portfolio_state=np.asarray(pstate,dtype=np.float16),
                                 account_state=np.asarray(astate,dtype=np.float16),
                                 goal_state=(np.asarray(account.goal_inputs(),dtype=np.float32) if account.goal_inputs() is not None else None),
                                multiscale_state=panel.multiscale_at(end_index).astype(np.float16),
                                daily_history=(panel.daily_history_at(end_index) if hasattr(panel,"daily_history_at") else None))
                        exp._bootstrap_experience=credit_successor
                        account_exp._bootstrap_experience=credit_successor
                matured_experiences.extend((exp,account_exp))
                matured_acks.append((pending_kind,
                    f"{dec.get('timestamp','')}|{dec.get('symbol',dec.get('symbol_index',''))}"))
                if first_account_transition:
                    account_transition_added.add(timestamp)
                    if not dec.get("credit_observations_target"):
                        self.metrics["paper_account_reward"] = float(
                            self.metrics.get("paper_account_reward",0.0))+account_reward
                    self.metrics["portfolio_experiences"] = int(
                        self.metrics.get("portfolio_experiences",0))+1
                self.metrics["paper_experiences_seen"]=int(
                    self.metrics.get("paper_experiences_seen",0))+1
                self.metrics["paper_experiences_since_candidate"]=int(
                    self.metrics.get("paper_experiences_since_candidate",0))+1
                self.replay.note_paper_outcome()
        if matured_experiences:
            # One atomic durable commit per observation. Neither an experience
            # nor its pending acknowledgement can be lost independently.
            self.replay.add_many(matured_experiences,pending_acks=matured_acks)
        return keep

    def follow_csv(self, data_path:str|Path, poll_seconds:float=5.0, initial_lookback_bars:int=0):
        """Observe an append-only UTC timestamp CSV until Ctrl+C.

        Feed columns follow the same global schema as the historical downloader.
        New bars can be minute/hourly; all instruments for a timestamp should
        be appended as a batch. Candidate training runs on the learner thread.
        """
        self.live_priority_enabled=True
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
                self.metrics["model_input_symbol_count"]=sum(not str(symbol).startswith("__PAD__") for symbol in panel.symbols)
                self.metrics["model_padding_symbol_count"]=sum(str(symbol).startswith("__PAD__") for symbol in panel.symbols)
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
                    if hasattr(panel,"daily_history_status_at"):
                        self.metrics["daily_history_input_status"]=panel.daily_history_status_at(ti)
                    actual_symbols=np.asarray([
                        not str(symbol).startswith("__PAD__") for symbol in panel.symbols],dtype=bool)
                    if actual_symbols.any():
                        width=len(TIMEFRAME_FEATURE_NAMES)
                        self.metrics["multiscale_coverage"]={
                            scale:float(multiscale_state[actual_symbols,k*width+width-1].mean())
                            for k,scale in enumerate(TIMEFRAME_NAMES)}
                        self.metrics["multiscale_input_status"]={
                            scale:{"observed_symbols":int(actual_symbols.sum()),
                                "available_symbols":int((multiscale_state[actual_symbols,k*width+4]>0).sum()),
                                "complete_history_symbols":int((multiscale_state[actual_symbols,k*width+5]>=.999).sum()),
                                "mean_history_coverage":float(multiscale_state[actual_symbols,k*width+5].mean())}
                            for k,scale in enumerate(TIMEFRAME_NAMES)}
                        self.metrics["long_context_input_status"]={
                            name:{"available_symbols":int((multiscale_state[actual_symbols,
                                BASE_MULTISCALE_FEATURE_COUNT+k*width+4]>0).sum()),
                                "observed_symbols":int(actual_symbols.sum()),
                                "mean_history_coverage":float(multiscale_state[actual_symbols,
                                    BASE_MULTISCALE_FEATURE_COUNT+k*width+5].mean())}
                            for k,name in enumerate(LONG_CONTEXT_NAMES)}
                    rows=[]
                    pending_inputs={"features":x[0].numpy().astype(np.float16),
                        "daily_history":panel.daily_history_at(ti) if hasattr(panel,"daily_history_at") else None,
                        "symbol_ids":sid[0].numpy(),"market_ids":mid[0].numpy(),
                        "asset_ids":aid[0].numpy(),"valid_mask":mask[0].numpy(),
                        "market_context":window[5][0].numpy().astype(np.float16) if len(window)>5 else None,
                        "multiscale_state":multiscale_state}
                    portfolio_inputs={**pending_inputs,
                        "goal_state":np.asarray(self.paper_account.goal_inputs(),dtype=np.float32),
                        "input_symbols":list(panel.symbols),
                        "portfolio_state":np.asarray(pstate,dtype=np.float16),
                        "account_state":np.asarray(astate,dtype=np.float32)}
                    goal_before=self.paper_account.goal_points();goal_complete=self.paper_account.goal_summary().get("status")=="WIN"
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
                               "credit_observations_target":getattr(self,"reward_credit_observations",60),
                               "credit_seconds":getattr(self,"reward_credit_seconds",3600),
                              "behavior_log_prob":behavior_log_prob,
                              "entry_price":float(panel.closes[ti,j]),"bars_elapsed":0,
                              "regime":abs(float(panel.features[ti,j,6])),
                               "is_validation":False,"promotion_holdout":validation,
                               "equity_before":self.paper_account.normalized_equity(),
                               "goal_points_before":goal_before,"goal_episode_id":self.paper_account.state["episode_id"],
                               "goal_complete_before":goal_complete,
                               "goal_weight_before":max(float(pstate[j][1]),float(allocation[j]) if action==2 and allocation is not None else 0.0),
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
                    self.metrics["champion_policy_diagnostics"]=summarize_policy(
                        panel,ti,probs,actions,allocation,self.paper_account,submitted_order_ids,stamp)
                    if self.metrics["champion_policy_diagnostics"]["observed_tradable_symbols"]:
                        self.metrics["champion_last_tradable_policy_diagnostics"]=self.metrics[
                            "champion_policy_diagnostics"]
                    for dec in portfolio_pending:
                        if dec.get("timestamp")==stamp:
                            dec["fill_expected"]=(dec.get("decision_id") in submitted_order_ids)
                    self.metrics["pending_experiences"] = len(pending)+len(portfolio_pending)
                    self.replay.save_pending(pending,portfolio_pending)
                    self.paper_account.save()
                    common_snapshot=self._queue_candidate_live_observation(panel,ti,paper_enabled,uniforms)
                    self._collect_candidate_validation(panel,ti,snapshot=common_snapshot)
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
            if self.candidate_live_thread.is_alive():
                self.candidate_live_thread.join(timeout=300)
            daily_request=self.state_dir/"daily_cycle.request"
            if daily_request.exists():
                request=json.loads(daily_request.read_text(encoding="utf-8"))
                if self.validation_active:
                    self._finish_candidate_validation(
                        self.validation_candidate_account.normalized_equity()-2.0,
                        self.validation_champion_account.normalized_equity()-2.0,None)
                _atomic_json({"finished_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                    "session_key":request.get("session_key"),
                    "validation_active":False,"bars":self.validation_bars},
                    self.state_dir/"daily_cycle.completed.json")
                daily_request.unlink(missing_ok=True)
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
        rules=getattr(self,"operating_rules",{})
        account.configure_goal(rules.get("goal_target_multiple",10.0),rules.get("goal_win_bonus_points",100.0))
        initial=account.normalized_equity()
        peak=initial; max_dd=0.0; daily={}; action_counts={name:0 for name in ACTION_NAMES}
        decisions=0; observed_bars=0; previous_equity=initial
        was_training=model.training; model.eval()
        model_device=next(model.parameters()).device
        try:
            with torch.inference_mode():
                for ti in range(start,end):
                    fills=account.process_bar(panel,ti,True)
                    account.observe_goal(str(panel.dates[ti]))
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
                                **self._daily_history_kwargs(panel,ti,model_device),
                                **self._goal_kwargs(account,model_device),
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
            multi=(experience.multiscale_state if experience.multiscale_state is not None
                   else np.zeros((experience.features.shape[1],MULTISCALE_FEATURE_COUNT),dtype=np.float16))
            if BASE_MULTISCALE_FEATURE_COUNT<=multi.shape[-1]<MULTISCALE_FEATURE_COUNT:
                multi=np.pad(multi,((0,0),(0,MULTISCALE_FEATURE_COUNT-multi.shape[-1])))
            m_rows.append(multi)
        return (args,
                torch.as_tensor(np.stack(p_rows),device=self.device,dtype=torch.float32),
                torch.as_tensor(np.stack(a_rows),device=self.device,dtype=torch.float32),
                torch.as_tensor(np.stack(m_rows),device=self.device,dtype=torch.float32))

    def _live_learning_wait_reason(self):
        if not getattr(self,"live_priority_enabled",False):
            return None
        scheduler=getattr(self,"candidate_live_inference_lock",None)
        if scheduler is None or not hasattr(scheduler,"snapshot"):
            return None
        state=scheduler.snapshot()
        if state.get("policy")=="preopen_replay_learning_first":
            return None
        roles=[state.get("active")]+[item["role"] for item in state.get("waiting",[])]
        live=[role for role in roles if role in ("champion_live","candidate_live")]
        if live:
            return "실시간 GPU 추론 요청에 양보: "+", ".join(live)
        return None

    @contextmanager
    def _learning_gpu_segment(self,role,durations):
        """Release CUDA after one shared-window backward or optimizer operation."""
        self.metrics["learning_wait_reason"]=self._live_learning_wait_reason()
        with self._gpu_work(role):
            self.metrics["learning_wait_reason"]=None
            started=time.perf_counter()
            if self.device.type=="cuda":
                start_event=torch.cuda.Event(enable_timing=True)
                end_event=torch.cuda.Event(enable_timing=True)
                start_event.record()
            try:
                yield
            finally:
                if self.device.type=="cuda":
                    end_event.record()
                    # Kernels must finish before another model uses the GPU.
                    end_event.synchronize()
                    durations.append(start_event.elapsed_time(end_event)/1000.0)
                else:
                    durations.append(time.perf_counter()-started)
                learner=role.split("_",1)[0]
                self.metrics[learner+"_learning_gpu_segments"]=int(
                    self.metrics.get(learner+"_learning_gpu_segments",0))+1
                self.metrics[learner+"_learning_gpu_segment_seconds_last"]=durations[-1]
                self.metrics[learner+"_learning_gpu_segment_seconds_max"]=max(
                    float(self.metrics.get(learner+"_learning_gpu_segment_seconds_max",0)),durations[-1])

    def _wait_for_live_inference(self):
        last_report=0.0
        while not self.stop.is_set():
            reason=self._live_learning_wait_reason()
            self.metrics["learning_wait_reason"]=reason
            if not reason:
                return True
            if time.monotonic()-last_report>=5:
                self._write_metrics();last_report=time.monotonic()
            self.stop.wait(.25)
        return False

    def _learner(self):
        while not self.stop.wait(.1):
            self.metrics["learner_last_heartbeat_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
            try:
                stats=self.replay.stats(self.candidate_replay_passes)
            except Exception as exc:
                self.metrics["learner_statistics_errors"]=int(self.metrics.get("learner_statistics_errors",0))+1
                self.metrics["learner_statistics_error"]=f"{type(exc).__name__}: {exc}"[:1200]
                self.stop.wait(1.0)
                continue
            self.metrics["learner_statistics_error"]=None
            previous_reason=self.metrics.get("learning_wait_reason")
            reason=self._live_learning_wait_reason()
            self.metrics["learning_wait_reason"]=reason
            if reason:
                self.metrics["candidate_skip_reason"]=reason
                if reason!=previous_reason:
                    self._write_metrics()
                self.stop.wait(.25)
                continue
            dual=getattr(self,"dual_learning_enabled",False)
            remaining=stats.get("model_remaining",{"candidate":stats["eligible"],"champion":0})
            self.metrics["candidate_untrained_replay_count"]=stats.get("model_untrained",{}).get("candidate",stats["untrained"])
            self.metrics["candidate_eligible_replay_count"]=remaining["candidate"]
            self.metrics["champion_eligible_replay_count"]=remaining.get("champion",0) if dual else 0
            self.metrics["candidate_replay_passes"]=self.candidate_replay_passes
            if not remaining["candidate"] and not (dual and remaining.get("champion",0)):
                self.metrics["candidate_skip_reason"]="남은 학습 가능한 경험을 모두 처리했습니다. 새 경험 대기"
                continue
            if self.metrics.get("candidate_training") or self.metrics.get("champion_training"):
                continue
            if time.monotonic()<self.candidate_retry_after:
                self.metrics["candidate_skip_reason"]="Candidate 오류 뒤 경험을 보존하고 재시도 대기 중"
                continue
            self.last_train_replay_size=len(self.replay)
            self.metrics["candidate_skip_reason"]=None
            preferred=getattr(self,"next_learning_role","candidate") if dual else "candidate"
            learner=preferred if remaining.get(preferred,0) else ("candidate" if preferred=="champion" else "champion")
            self.next_learning_role="candidate" if learner=="champion" else "champion"
            self.metrics["learning_active_role"]=learner
            try:
                if learner=="champion": self._train_champion()
                else: self._train_candidate()
            except Exception as exc:
                self.metrics[learner+"_training"]=False
                self.metrics[learner+"_errors"]=int(self.metrics.get(learner+"_errors",0))+1
                self.metrics["last_"+learner+"_error"]=f"{type(exc).__name__}: {exc}"[:1200]
                exc.__traceback__=None
                del exc
                self._schedule_candidate_retry()
                self._release_cuda_cache()
                try:
                    if learner=="candidate": self._restore_candidate_checkpoint()
                    else:
                        restored,_=load_model(self.champion_path,self.device,self.instrument_config)
                        with self.lock: self.champion=restored.eval()
                        commit=getattr(restored,"_stockrl_replay_commit",{})
                        if commit.get("learner")=="champion":
                            self.replay.acknowledge_training(commit.get("uses",{}),self.candidate_replay_passes,learner="champion")
                            self.champion_training_version=max(self.champion_training_version,int(commit.get("model_version",0)))
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

    def _configure_candidate_trainables(self,candidate,learner="candidate"):
        metrics=TrainingMetrics(self.metrics,learner)
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
                        "multiscale_policy","multiscale_value","daily_history_encoder",
                        "daily_history_policy","daily_history_value","goal_policy","goal_value","goal_cash"))
        for module in modules:
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        trainable=[parameter for parameter in candidate.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("candidate has no trainable parameters")
        metrics["candidate_trainable_transformer_blocks"]=len(trainable_blocks)
        metrics["candidate_trainable_parameter_count"]=sum(
            parameter.numel() for parameter in trainable)
        metrics["candidate_total_parameter_count"]=sum(
            parameter.numel() for parameter in candidate.parameters())
        return trainable

    def _experience_loss(self,logits,values,allocations,experience,successor_values=None):
        chosen=logits[0,int(experience.symbol_index)].float()
        predicted=values[0,int(experience.symbol_index)].float()
        if not torch.isfinite(chosen).all() or not torch.isfinite(predicted).all():
            self.metrics["nonfinite_updates"]+=1
            return None
        action=torch.as_tensor([experience.action],device=self.device,dtype=torch.long)
        epsilon=min(PAPER_EXPLORATION_EPSILON,1.0/max(1,int(experience.features.shape[1])))
        probabilities=(1.0-epsilon)*torch.softmax(chosen,dim=-1)+epsilon/3.0
        dist=Categorical(probs=probabilities[None])
        reward=torch.as_tensor(float(experience.reward)*100.0+float(experience.goal_reward_points),device=self.device,dtype=torch.float32)
        target=reward
        if experience.bootstrap_discount:
            if successor_values is None:
                raise ValueError("future-credit successor value is missing; replay must be retained")
            target=reward+float(experience.bootstrap_discount)*successor_values[
                0,int(experience.bootstrap_symbol_index)].float().detach()
        loss=chosen.sum()*0
        if not experience.portfolio_transition:
            advantage=target-predicted.detach()
            if experience.behavior_log_prob is not None and experience.trade_executed:
                old=torch.as_tensor(float(experience.behavior_log_prob),device=self.device,dtype=torch.float32)
                ratio=torch.exp((dist.log_prob(action)[0]-old).clamp(-20,20))
                loss=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)
            loss=loss+.5*nn.functional.smooth_l1_loss(predicted[None],target[None])
            loss=loss-.0005*dist.entropy().mean()
        else:
            if experience.portfolio_value_transition:
                account_reward=torch.as_tensor(100.0*float(experience.portfolio_reward or 0.0)+float(experience.portfolio_goal_reward_points),
                                               device=self.device,dtype=torch.float32)
                if experience.bootstrap_discount:
                    account_reward=account_reward+float(experience.bootstrap_discount)*successor_values[0].float().mean().detach()
                loss=loss+.25*nn.functional.smooth_l1_loss(values[0].float().mean(),account_reward)
            if allocations is not None and experience.trade_executed:
                selected=allocations[0,int(experience.symbol_index)].clamp_min(1e-7)
                loss=loss-.10*target.detach()*torch.log(selected)
        if experience.source.startswith("teacher"):
            loss=loss+nn.functional.cross_entropy(chosen[None],action)
        if not torch.isfinite(loss):
            self.metrics["nonfinite_updates"]+=1
            return None
        return loss

    def _train_candidate(self):
        return self._train_model("candidate")

    def _train_champion(self):
        return self._train_model("champion")

    def _credit_successor_values(self,model,experience,cache):
        """One gradient-free successor forward per shared input and update."""
        if not experience.bootstrap_discount:
            return None
        key=experience.bootstrap_window_key
        if key not in cache:
            inputs=getattr(experience,"_bootstrap_inputs",None)
            if inputs is None:
                raise ValueError("future-credit input is missing; replay retained")
            successor=Experience(**inputs,symbol_index=0,action=1,reward=0.0,
                                 timestamp=experience.timestamp)
            args,pstate,astate,mstate=self._pack([successor])
            was_training=model.training
            try:
                model.eval()
                with torch.no_grad():
                    if getattr(model,"_stockrl_uses_market_context",False):
                        output=model(*args,portfolio_state=pstate,account_state=astate,
                            return_allocation=True,multiscale_state=mstate,
                            **self._saved_goal_kwargs(successor,self.device),
                            **({"daily_history":torch.as_tensor(successor.daily_history[None],
                                   device=self.device,dtype=torch.float32)}
                               if successor.daily_history is not None else {}))
                    else:
                        output=model(*args)
                    cache[key]=output[1].detach()
            finally:
                model.train(was_training)
        return cache[key]

    def _train_model(self,learner):
        metrics=TrainingMetrics(self.metrics,learner)
        from datetime import datetime, timezone
        training_started=time.perf_counter()
        trigger_experience_count=int(metrics.get("paper_experiences_since_candidate",0))
        metrics["candidate_training"]=True
        metrics["candidate_skip_reason"] = None
        metrics["last_candidate_compute_seconds"]=0.0
        metrics["last_candidate_step_compute_seconds"]=0.0
        metrics["last_candidate_total_seconds"]=0.0
        metrics["last_candidate_peak_allocated_bytes"]=None
        metrics["last_candidate_peak_reserved_bytes"]=None
        self.metrics[learner+"_learning_gpu_segments"]=0
        self.metrics[learner+"_learning_gpu_segment_seconds_max"]=0.0
        with self.lock:
            source_champion=self.champion
        with self._gpu_work(learner+"_learning_setup"):
            if learner=="candidate":
                if self.candidate is None:
                    self._restore_candidate_checkpoint()
                with self.candidate_model_lock:
                    self.candidate=self.candidate.to(self.device)
                    candidate=self.candidate.train()
                model_version=self.candidate_version
            else:
                # Inference holds the immutable published model while its learner
                # works on an independent copy. Publish only a completed checkpoint.
                candidate=self._new_model_like(source_champion,self.device)
                candidate.load_state_dict(source_champion.state_dict())
                candidate.train()
                model_version=self.champion_training_version
        original=source_champion
        trainable_parameters=self._configure_candidate_trainables(candidate,learner)
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
        timeframe_samples=dict.fromkeys(TIMEFRAME_NAMES,0)
        long_context_samples=dict.fromkeys(LONG_CONTEXT_NAMES,0)
        daily_history_samples=0
        sampled_ids=set()
        # The SQLite queue is authoritative; the small RAM cache is not the
        # backlog. Each learner takes every eligible row once, oldest first.
        metrics["last_candidate_optimizer_steps"]=0
        metrics["last_candidate_samples_trained"]=0
        metrics["candidate_optimizer_steps_current"]=0
        metrics["candidate_samples_current"]=0
        metrics["candidate_window_forwards_current"]=0
        metrics["candidate_optimizer_steps_target"]=self.updates_per_candidate
        metrics["candidate_samples_target"]=self.updates_per_candidate*self.batch_size
        for update_ix in range(self.updates_per_candidate):
            if self.stop.is_set() or not self._wait_for_live_inference():
                break
            batch=self.replay.pending_batch(self.batch_size,self.candidate_replay_passes,
                exclude_row_ids=sampled_ids,learner=learner)
            if not batch: break
            sampled_ids.update(self.replay.row_ids_for(batch))
            malformed=[e for e in batch if (
                (e.goal_state is not None and np.shape(e.goal_state)!=(6,)) or
                (e.portfolio_state is not None and np.shape(e.portfolio_state)!=(e.features.shape[1],8)) or
                (e.account_state is not None and np.shape(e.account_state)!=(8,)) or
                (e.multiscale_state is not None and np.shape(e.multiscale_state) not in (
                    (e.features.shape[1],MULTISCALE_FEATURE_COUNT),(e.features.shape[1],96),
                    (e.features.shape[1],BASE_MULTISCALE_FEATURE_COUNT))))]
            if malformed:
                self.replay.quarantine(self.replay.row_ids_for(malformed),"saved portfolio input shape is incompatible")
                rejected={id(e) for e in malformed}
                batch=[e for e in batch if id(e) not in rejected]
                if not batch: continue
            candidate_sample_keys.update((e.origin_model,e.source,e.timestamp,e.symbol_index,e.action,e.portfolio_transition)
                                         for e in batch)
            ti=time.perf_counter(); opt.zero_grad(set_to_none=True); valid_samples=0; loss_values=[]
            step_segments=[]
            use_portfolio=getattr(candidate,"_stockrl_uses_market_context",False)
            # Preserve one optimizer step and the full minibatch gradient while
            # giving queued live inference priority between independent windows.
            successful_batch=[]
            successor_cache={}
            groups=OrderedDict()
            for experience in batch:
                key=getattr(experience,"_replay_window_key",None) or self.replay._window_key(experience)
                groups.setdefault(key,[]).append(experience)
            for group in groups.values():
                with self._learning_gpu_segment(learner+"_learning_step",step_segments):
                    args,pstate,astate,mstate=self._pack([group[0]])
                    if use_portfolio:
                        logits,values,allocations=candidate(*args,portfolio_state=pstate,
                            account_state=astate,return_allocation=True,multiscale_state=mstate,
                            **self._saved_goal_kwargs(group[0],self.device),
                            **({"daily_history":torch.as_tensor(group[0].daily_history[None],device=self.device,dtype=torch.float32)}
                               if getattr(group[0],"daily_history",None) is not None else {}))
                    else:
                        logits,values=candidate(*args); allocations=None
                    losses=[]
                    for experience in group:
                        successor_values=None
                        if experience.bootstrap_discount:
                            successor_values=self._credit_successor_values(candidate,experience,successor_cache)
                        loss=self._experience_loss(logits,values,allocations,experience,successor_values)
                        if loss is None:
                            continue
                        losses.append(loss)
                        successful_batch.append(experience)
                    if losses:
                        (torch.stack(losses).sum()/len(batch)).backward()
                        valid_samples+=len(losses)
                        loss_values.extend(torch.stack([loss.detach() for loss in losses]).cpu().tolist())
                    metrics["candidate_window_forwards_current"]+=1
                    del args,pstate,astate,mstate,logits,values,allocations,losses
            if not valid_samples:
                continue
            with self._learning_gpu_segment(learner+"_learning_step",step_segments):
                nn.utils.clip_grad_norm_(candidate.parameters(),1.0)
                with self.candidate_model_lock:
                    opt.step()
                    if self.device.type=="cuda": torch.cuda.synchronize(self.device)
            compute_elapsed.append(sum(step_segments))
            candidate_samples+=valid_samples
            metrics["teacher_examples_trained"]+=sum(e.source.startswith("teacher") for e in successful_batch)
            metrics["paper_examples_trained"]+=sum(not e.source.startswith("teacher") for e in successful_batch)
            used_experiences.extend(successful_batch)
            for experience in successful_batch:
                scale_input=experience.multiscale_state
                if getattr(experience,"daily_history",None) is not None and np.any(experience.daily_history[...,5]>0):
                    daily_history_samples+=1
                if scale_input is not None:
                    width=len(TIMEFRAME_FEATURE_NAMES)
                    for k,scale in enumerate(TIMEFRAME_NAMES):
                        if np.any(np.asarray(scale_input)[:,k*width+4]>0):
                            timeframe_samples[scale]+=1
                    if np.shape(scale_input)[-1]>=BASE_MULTISCALE_FEATURE_COUNT:
                        for k,name in enumerate(LONG_CONTEXT_NAMES):
                            column=BASE_MULTISCALE_FEATURE_COUNT+k*width+4
                            if column<np.shape(scale_input)[-1] and np.any(np.asarray(scale_input)[:,column]>0):
                                long_context_samples[name]+=1
            metrics["update_losses"].append(float(np.mean(loss_values)))
            metrics["update_losses"]=metrics["update_losses"][-2000:]
            if self.device.type=="cuda":
                torch.cuda.synchronize(self.device)
                metrics["last_candidate_peak_allocated_bytes"]=int(
                    torch.cuda.max_memory_allocated(self.device))
                metrics["last_candidate_peak_reserved_bytes"]=int(
                    torch.cuda.max_memory_reserved(self.device))
            elapsed.append(time.perf_counter()-ti); self.steps+=1
            metrics["candidate_optimizer_steps_current"]=len(elapsed)
            metrics["candidate_samples_current"]=candidate_samples
            metrics["last_candidate_compute_seconds"]=float(sum(compute_elapsed))
            metrics["last_candidate_step_compute_seconds"]=(
                float(sum(compute_elapsed)/len(compute_elapsed)) if compute_elapsed else 0.0)
            metrics["last_candidate_total_seconds"]=float(time.perf_counter()-training_started)
            self._write_metrics()
        metrics["update_seconds"].extend(elapsed); metrics["updates"]+=len(elapsed)
        metrics["last_candidate_update_seconds"]=float(sum(elapsed))
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
            metrics["last_candidate_peak_allocated_bytes"]=int(
                torch.cuda.max_memory_allocated(self.device))
            metrics["last_candidate_baseline_allocated_bytes"]=training_baseline_allocated
            metrics["last_candidate_peak_reserved_bytes"]=int(
                torch.cuda.max_memory_reserved(self.device))
        metrics["last_candidate_optimizer_steps"]=len(elapsed)
        metrics["last_candidate_samples_trained"]=candidate_samples
        metrics["last_candidate_unique_samples_trained"]=len(candidate_sample_keys)
        metrics["candidate_optimizer_steps_target"]=self.updates_per_candidate
        metrics["candidate_samples_target"]=self.updates_per_candidate*self.batch_size
        metrics["update_seconds"]=metrics["update_seconds"][-2000:]
        metrics["teacher_mix_probability"]=self.replay.teacher_fraction()
        if not elapsed:
            metrics["candidate_training"]=False
            if not self.stop.is_set():
                metrics["candidate_errors"]=int(metrics.get("candidate_errors",0))+1
                metrics["last_candidate_error"]="no finite optimizer update; replay retained for retry"
                self._schedule_candidate_retry()
            candidate.eval()
            if self.device.type=="cuda": candidate.to("cpu")
            self._write_metrics()
            del candidate,opt
            self._release_cuda_cache()
            return
        if not all(torch.isfinite(p).all() for p in candidate.parameters()):
            metrics["nonfinite_updates"]+=1
            metrics["rejections"]+=1
            self._schedule_candidate_retry()
            metrics["last_rejection_utc"]=datetime.now(timezone.utc).isoformat()
            history=metrics.setdefault("candidate_gate_history",[])
            history.append({"time_utc":metrics["last_rejection_utc"],"applied":False,
                            "reason":"가중치에 계산 불가능한 값이 발생해 적용하지 않음",
                            "candidate_score":None,"champion_score":None})
            metrics["candidate_gate_history"]=history[-20:]
            metrics["candidate_training"]=False
            if learner=="candidate": self._restore_candidate_checkpoint()
            self._write_metrics()
            del candidate,opt
            self._release_cuda_cache()
            return
        delta=sum((candidate.state_dict()[k].float()-v.detach().float()).abs().sum().item()
                  for k,v in original.state_dict().items())
        metrics["weight_delta_l1"].append(delta)
        metrics["weight_delta_l1"]=metrics["weight_delta_l1"][-2000:]
        metrics["last_update_utc"]=datetime.now(timezone.utc).isoformat()
        metrics["last_candidate_error"]=None
        self.state_dir.mkdir(exist_ok=True,parents=True)
        trained_replay_row_ids=self.replay.row_ids_for(used_experiences)
        uses={getattr(e,"_replay_row_id",self.replay.row_ids.get(id(e),id(e))):
              int(getattr(e,"_replay_training_uses",0))+1 for e in used_experiences}
        replay_commit={"uses":uses,"passes":self.candidate_replay_passes,"learner":learner,
                       "model_version":model_version+len(elapsed),
                       "candidate_version":model_version+len(elapsed)}
        if learner=="champion":
            staged=self.state_dir/"champion.learning.next"
            try:
                save_model(staged,candidate,self.cfg,step=self.steps,
                           temp_dir=self.state_dir,replay_commit=replay_commit)
                with self.lock:
                    if self.champion is not source_champion:
                        # A winning trial was promoted while this branch trained.
                        # Retry these rows against the new champion; do not ACK.
                        metrics["candidate_training"]=False
                        metrics["candidate_skip_reason"]="승급된 Champion에서 같은 경험을 다시 학습합니다"
                        return
                    os.replace(staged,self.champion_path)
                    candidate._stockrl_replay_commit=replay_commit
                    self.champion=candidate.eval()
                    self.champion.requires_grad_(False)
                    self.champion_training_version=replay_commit["model_version"]
            finally:
                staged.unlink(missing_ok=True)
        else:
            save_model(self.model_dir/"candidate.pt",candidate,self.cfg,step=self.steps,
                       temp_dir=self.state_dir,replay_commit=replay_commit)
            candidate._stockrl_replay_commit=replay_commit
        consumed=self.replay.acknowledge_training(uses,self.candidate_replay_passes,learner=learner)
        if learner=="candidate":
            with self.candidate_model_lock:
                self.candidate_version=replay_commit["model_version"]
                self.candidate_trained_replay_row_ids=set(trained_replay_row_ids)
                self.candidate_replay_uses={}
                self._atomic_json({"version":2,"candidate_version":self.candidate_version,
                    "queue_source":"replay.sqlite3","last_checkpoint_rows":trained_replay_row_ids},
                    self.candidate_lineage_path)
                self._publish_candidate_observer(candidate,self.candidate_version)
            metrics["candidate_live_status"]="candidate_updated"
            metrics["candidate_has_learning"]=True
        metrics["candidate_training"]=False
        if learner=="candidate" and not self.validation_active and not self.stop.is_set():
            self._begin_candidate_validation(candidate)
        if learner=="candidate" and self.device.type=="cuda": candidate.to("cpu")
        metrics["candidate_replay_rows_held_for_validation"]=0
        metrics["candidate_replay_rows_consumed"]=consumed
        metrics["last_candidate_window_forwards"]=metrics["candidate_window_forwards_current"]
        metrics["candidate_replay_rows_audit_status"]="tracked"
        metrics["candidate_completed_training_runs"]=int(metrics.get("candidate_completed_training_runs",0))+1
        self.last_train_replay_size=len(self.replay)
        metrics["last_train_replay_size"]=self.last_train_replay_size
        metrics["paper_experiences_since_candidate"]=max(
            0,int(metrics.get("paper_experiences_since_candidate",0))-trigger_experience_count)
        metrics["candidate_replay_since_last_update"]=int(
            metrics["paper_experiences_since_candidate"])
        metrics["candidate_stage"]=("sequential_paper_validation" if self.validation_active else
                                          "waiting_for_replay")
        metrics["last_candidate_total_seconds"]=float(time.perf_counter()-training_started)
        metrics["candidate_last_completed_round"]={
            "samples":candidate_samples,"unique_samples":len(candidate_sample_keys),
            "optimizer_steps":len(elapsed),
            "future_credit_samples":sum(e.credit_observations>0 for e in used_experiences),
            "successor_value_samples":sum(e.bootstrap_discount>0 for e in used_experiences),
            "goal_conditioned_samples":sum(e.goal_state is not None for e in used_experiences),
            "goal_bonus_samples":sum(bool(e.goal_reward_points or e.portfolio_goal_reward_points) for e in used_experiences),
            "legacy_short_reward_samples":sum(not e.credit_observations and not e.source.startswith("teacher") for e in used_experiences),
            "total_seconds":metrics["last_candidate_total_seconds"],
            "compute_seconds":metrics["last_candidate_compute_seconds"],
            "step_compute_seconds":metrics["last_candidate_step_compute_seconds"],
            "peak_allocated_bytes":metrics["last_candidate_peak_allocated_bytes"],
            "completed_utc":metrics["last_update_utc"],
            "model_version":replay_commit["model_version"]}
        metrics["candidate_last_completed_round"]["timeframe_samples"]=timeframe_samples
        metrics["candidate_last_completed_round"]["long_context_samples"]=long_context_samples
        metrics["candidate_last_completed_round"]["daily_history_samples"]=daily_history_samples
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
        with self.lock:
            champion_snapshot=self._new_model_like(self.champion,torch.device("cpu"))
            champion_snapshot.load_state_dict(self.champion.state_dict())
            champion_snapshot.eval(); champion_snapshot.requires_grad_(False)
            self.validation_champion=champion_snapshot
            source_sha=self._sha256_file(self.champion_path)
            source_champion_version=self.champion_training_version
            self.validation_promotion_epoch=int(self.metrics.get("promotions",0))
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
        self.validation_source_sha256=source_sha
        self._atomic_json({"status":"collecting","start_after":self.validation_start_after,
                           "bars":0,"source_champion_sha256":source_sha,
                           "source_candidate_version":self.candidate_version,
                           "source_champion_version":source_champion_version,
                           "source_promotion_epoch":self.validation_promotion_epoch,
                           "both_models_frozen":True,
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
        self.metrics["champion_validation_snapshot_version"]=source_champion_version
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

    def _collect_candidate_validation(self,panel,index,snapshot=None):
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
        if snapshot is None:
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
        while not self.stop.is_set() or ((self.state_dir/"daily_cycle.request").exists()
                                        and not self.validation_queue.empty()):
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
            champ_account.observe_goal(stamp)
            cand_account.observe_goal(stamp)
            window=self._window(panel,index)
            args=window
            champion=self.validation_champion
            validation_candidate=self.validation_candidate
            accounts=(("champion",champion,champ_account),
                      ("candidate",validation_candidate,cand_account))
            for model_name,model,account in accounts:
                # Candidate observation and frozen candidate validation share
                # one GPU inference slot so they cannot duplicate the 0.5B
                # model allocation at the same time.
                inference_lock=self._gpu_work("validation_"+model_name)
                with inference_lock:
                    snapshot_offloaded=(self.device.type=="cuda"
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
                                                          multiscale_state=mt,**self._daily_history_kwargs(panel,index,model_device),
                                                          **self._goal_kwargs(account,model_device))
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
            if self.validation_bars>=self.validation_window_bars and not self.daily_promotion:
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
        frozen_pair=bool(state.get("both_models_frozen"))
        if error is None and frozen_pair and state.get("source_promotion_epoch")!=int(self.metrics.get("promotions",0)):
            error="another promotion invalidated this frozen model pair"
        if error is None and not frozen_pair and state.get("source_champion_sha256")!=source_sha:
            error="champion changed during candidate validation"
        promoted=False
        if error is None:
            try:
                promoted=self._commit_candidate(candidate,float(candidate_score),float(champion_score),
                    self.validation_bars,state.get("source_champion_sha256",source_sha),trained_replay_row_ids,
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
            history=self.metrics.setdefault("candidate_gate_history",[])
            history.append({"time_utc":self.metrics["last_rejection_utc"],"applied":False,
                "reason":error,"candidate_score":candidate_score,"champion_score":champion_score,
                "candidate_version":snapshot_version,"champion_version":state.get("source_champion_version"),
                "bars":self.validation_bars,"required_bars":self.validation_window_bars,
                "candidate_profitable":candidate_score is not None and candidate_score>0,
                "candidate_beats_champion":candidate_score is not None and champion_score is not None and candidate_score>champion_score})
            self.metrics["candidate_gate_history"]=history[-20:]
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
        self.validation_champion=None
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
        if not self.stop.is_set() and self.candidate is not None and self.candidate_version>snapshot_version:
            self._begin_candidate_validation(self.candidate)

    def _commit_candidate(self,candidate,score_new:float,score_old:float,validation_bars:int,
                          source_champion_sha256:str,trained_replay_row_ids=None,
                          candidate_version:int|None=None,trial_state=None)->bool:
        """Stage a valid champion checkpoint, then atomically swap the reader."""
        frozen_pair=bool((trial_state or {}).get("both_models_frozen"))
        same_epoch=((trial_state or {}).get("source_promotion_epoch")==int(self.metrics.get("promotions",0)))
        source_valid=(same_epoch if frozen_pair else self._sha256_file(self.champion_path)==source_champion_sha256)
        promote=(validation_bars>=self.validation_window_bars
                 and source_valid
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
            self._atomic_json({"sha256":promoted_sha,"set_reason":"candidate passed profitable daily frozen-snapshot paper gate"},
                              baseline_next)
            with self.lock:
                if (frozen_pair and int(self.metrics.get("promotions",0))!=(trial_state or {}).get("source_promotion_epoch")) or (not frozen_pair and self._sha256_file(self.champion_path)!=source_champion_sha256):
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
                "model pair invalidated during comparison" if not source_valid else
                "candidate paper-account net return was not positive" if score_new<=0 else
                "candidate paper-account net return did not beat champion")
        history.append({"time_utc":self.metrics["last_promotion_utc" if promote else "last_rejection_utc"],
                        "applied":promote,"reason":reason,
                        "candidate_score":float(score_new),"champion_score":float(score_old),
                        "candidate_version":candidate_version,
                        "champion_version":(trial_state or {}).get("source_champion_version"),
                        "bars":validation_bars,"required_bars":self.validation_window_bars,
                        "candidate_profitable":bool(score_new>0),
                        "candidate_beats_champion":bool(score_new>score_old)})
        self.metrics["candidate_gate_history"]=history[-20:]
        self._atomic_json({"status":"promoted" if promote else "rejected",
                           "bars":validation_bars,"candidate_score":float(score_new),
                           "champion_score":float(score_old),"applied":promote,
                           "source_candidate_version":candidate_version,
                           "source_champion_version":(trial_state or {}).get("source_champion_version"),
                           "both_models_frozen":frozen_pair,
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
        scheduler=getattr(self,"candidate_live_inference_lock",None)
        if hasattr(scheduler,"snapshot"):
            metrics["gpu_scheduler"]=scheduler.snapshot()
        replay_stats=self.replay.stats(self.candidate_replay_passes)
        validation_snapshot_bytes=(sum(p.numel()*p.element_size()
            for p in self.validation_candidate.parameters()) if self.validation_candidate is not None else 0)
        metrics.update({"parameters":parameter_count(self.champion),"device":str(self.device),
          "shared_objective":{"mode":"net_equity_and_tenfold_goal_for_both_models",
              "target_multiple":self.operating_rules.get("goal_target_multiple",10.0),
              "win_bonus_points":self.operating_rules.get("goal_win_bonus_points",100.0),
              "live_accounts_preserved":not self.operating_rules.get("daily_reset_live_accounts",False),
              "goal_reward_in_competition_score":False,"shared_replay":True},
          "replay_persistence":"durable_fifo_shared_frames_sqlite",
          "learning_priority":metrics.get("gpu_scheduler",{}).get("policy","live_inference_first_then_complete_replay_coverage"),
          "replay_untrained_count":replay_stats["untrained"],
          "replay_pending_count":replay_stats.get("pending",0),
          "replay_database_path":str(self.replay.journal_path),
          "model_symbol_id_capacity":self.cfg.max_symbols,
          "replay_eligible_backlog":replay_stats["eligible"],
          "candidate_eligible_replay_count":replay_stats["model_remaining"]["candidate"],
          "candidate_untrained_replay_count":replay_stats["model_untrained"]["candidate"],
          "champion_eligible_replay_count":replay_stats["model_remaining"]["champion"],
          "champion_untrained_replay_count":replay_stats["model_untrained"]["champion"],
          "dual_learning_enabled":True,
          "shared_observation":self.replay.market_observation_stats(),
          "learning_experience_origins":"champion_and_candidate_own_account_outcomes",
          "learner_thread_alive":bool(getattr(self,"thread",None) and self.thread.is_alive()),
          "champion_learning_enabled":bool(not self.stop.is_set() and getattr(self,"thread",None) and self.thread.is_alive()),
          "champion_training_version":self.champion_training_version,
          "champion_batch_size":self.batch_size,
          "champion_optimizer_steps_target":self.updates_per_candidate,
          "champion_samples_target":self.updates_per_candidate*self.batch_size,
          "replay_quarantined_count":replay_stats["quarantined"],
          "replay_blocked_reasons":replay_stats.get("blocked_reasons",{}),
          "replay_completed_retained":replay_stats.get("completed_retained",0),
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
          "learning_policy":"one_checkpoint_confirmed_pass_per_model",
          "reward_credit":{"mode":"n_step_net_equity_with_successor_value",
              "observations":getattr(self,"reward_credit_observations",60),"discount":1.0,
              "duration_seconds":getattr(self,"reward_credit_seconds",3600),
              "score_unit":"one_point_per_one_percent_net_return",
          "legacy_short_reward_backlog_retained":True},
          "promotion_score_mode":"sequential_paper_account_net_return",
          "promotion_uses_sequential_paper_account":True,
          "promotion_gate_ready":not bool(metrics.get("promotion_blocked_reason")),
          "promotion_blocked_reason":metrics.get("promotion_blocked_reason"),
          "promotion_baseline_sha256":self.promotion_baseline_sha256,
          "candidate_learning_enabled":bool(not self.stop.is_set() and getattr(self,"thread",None) and self.thread.is_alive()),
          "candidate_start_ready":bool(replay_stats["model_remaining"]["candidate"]),
          "champion_start_ready":bool(replay_stats["model_remaining"]["champion"]),
          "candidate_every":self.candidate_interval,
          "candidate_min_replay":1,
          "candidate_batch_size":self.batch_size,
          "candidate_optimizer_steps_target":self.updates_per_candidate,
          "candidate_samples_target":self.updates_per_candidate*self.batch_size,
          "candidate_min_validation_dates":self.validation_window_bars,
          "promotion_schedule":"daily" if self.daily_promotion else "market_bars",
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
        # Each learner saves its checkpoint before acknowledging replay.
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
    if contextual:
        extra.update(OnlineGlobalAgent._daily_history_kwargs(panel,end_index,dev))
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
