"""Online-agent assembly and lifecycle. Implementations live in stockrl.online."""
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
from .global_transformer import ACTION_NAMES, GlobalMarketPanel, GlobalMarketTransformer, TransformerConfig, parameter_count, load_compatible_state_dict
from .paper_account import PaperAccount, _currency
from .account_diagnostics import summarize_policy
from .operating_rules import operating_rules
from .state_io import atomic_json
from .multiscale import MULTISCALE_FEATURE_COUNT, TIMEFRAME_NAMES, TIMEFRAME_FEATURE_NAMES, MULTISCALE_FEATURE_ORDER, BASE_MULTISCALE_FEATURE_COUNT, LONG_CONTEXT_NAMES
from .replay_store import GlobalReplayBuffer, ReplayStorageFull
from .online.checkpoint import _atomic_json, _atomic_save, load_model, save_model
from .online.data import Experience, IncrementalMarketCSV, MarketObservation, ONLINE_TRAINABLE_BLOCKS, PAPER_EXPLORATION_EPSILON, REWARD_DEFINITION, REWARD_VERSION, TrainingMetrics, parse_horizon
from .online.evaluation import benchmark_model
from .online.rewards import net_action_reward
from .online.validation import should_promote
from .online.observation import _ObservationMixin
from .online.rewards import _RewardMixin
from .online.learning import _LearningMixin
from .online.validation import _ValidationMixin
from .online.evaluation import _EvaluationMixin


class OnlineGlobalAgent(_ObservationMixin, _RewardMixin, _LearningMixin, _ValidationMixin, _EvaluationMixin):
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
        from .online.runtime_updates import RuntimeUpdates
        self.runtime_updates=RuntimeUpdates(self)

    def start(self):
        if self.validation_queue_thread.ident is None: self.validation_queue_thread.start()
        if self.candidate_live_thread.ident is None: self.candidate_live_thread.start()
        if self.thread.ident is None: self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread.is_alive(): self.thread.join(timeout=300)
        if self.validation_queue_thread.is_alive(): self.validation_queue_thread.join(timeout=300)
        if self.candidate_live_thread.is_alive(): self.candidate_live_thread.join(timeout=300)

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
