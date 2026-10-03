"""Live market observation, independent account inference and runtime controls."""
from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
import json, os, time, pickle
import numpy as np
import torch
from ..market_panel import ACTION_NAMES, GlobalMarketPanel
from ..paper_account import _currency
from ..account_diagnostics import summarize_policy
from ..multiscale import TIMEFRAME_NAMES, TIMEFRAME_FEATURE_NAMES, BASE_MULTISCALE_FEATURE_COUNT, LONG_CONTEXT_NAMES
from .checkpoint import _atomic_json
from .data import PAPER_EXPLORATION_EPSILON
from ..experience import IncrementalMarketCSV, MarketObservation, REWARD_VERSION

class _ObservationMixin:
    def _queue_candidate_live_observation(self,panel,index,paper_enabled,uniforms):
        if not self._model_enabled("candidate"):return None
        snapshot=MarketObservation(panel,index,self.window)
        self.replay.enqueue_market_observation(snapshot,bool(paper_enabled),tuple(float(x) for x in uniforms))
        return snapshot

    def _gpu_work(self,role):
        from contextlib import nullcontext
        scheduler=getattr(self,"candidate_live_inference_lock",None)
        if scheduler is None:
            return nullcontext()
        return scheduler.work(role) if hasattr(scheduler,"work") else scheduler

    def _run_modes(self):
        """Independent controls; collecting quotes never depends on these flags."""
        try:
            state=json.loads((self.state_dir/"autonomy.json").read_text(encoding="utf-8"))
        except (OSError,ValueError,AttributeError):
            state={}
        return {"observe_enabled":bool(state.get("observe_enabled",True)),
                "paper_enabled":bool(state.get("paper_enabled",state.get("enabled",True))),
                "learning_enabled":bool(state.get("learning_enabled",True))}

    @staticmethod
    def _has_tradable_update(panel,index):
        """Context-only quotes cannot submit stock orders and need no full policy pass."""
        return any(bool(panel.observed[index,j]) and
                   _currency(*panel.groups.get(symbol,("",""))) is not None
                   for j,symbol in enumerate(panel.symbols))

    def _candidate_live_worker(self):
        """Run an independent observational paper account for current candidate weights."""
        while not self.stop.is_set():
            if not self._model_enabled("candidate"):
                self.stop.wait(.25)
                continue
            with self.role_locks["candidate"]:
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
                modes=self._run_modes()
                paper_enabled=bool(data["paper_enabled"]) and modes["paper_enabled"]
                uniforms=data["uniforms"]
                full_inference=(modes["observe_enabled"] and bool(uniforms)
                                and self._has_tradable_update(panel,index))
                if full_inference and self.candidate_live_model is None:
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
                    if not full_inference:
                        self.replay.commit_observer(stamp,account.state,self.candidate_portfolio_pending)
                        committed=True
                        account.save()
                        reason="context_only" if modes["observe_enabled"] else "judgment_paused"
                        key="candidate_"+reason+"_updates"
                        self.metrics[key]=int(self.metrics.get(key,0))+1
                        self.metrics["candidate_live_last_timestamp"]=stamp
                        self.metrics["candidate_live_status"]=reason
                        self.metrics["candidate_live_error"]=None
                        try:
                            state=json.loads(self.candidate_live_state_path.read_text(encoding="utf-8"))
                        except (OSError,ValueError):
                            state={}
                        # Preserve the last genuine action and its own timestamp.
                        state.setdefault("last_full_decision_timestamp",state.get("last_timestamp"))
                        state.update({"status":reason,"last_timestamp":stamp,"error":None,
                            "candidate_training":bool(self.metrics.get("candidate_training")),
                            "inference_skipped_reason":reason,
                            "context_only_updates":self.metrics.get("candidate_context_only_updates",0),
                            "observation_profile":{**observation_profile,
                                "inference_seconds":0.0,"total_seconds":time.perf_counter()-observation_started},
                            "gpu_scheduler":(self.candidate_live_inference_lock.snapshot()
                                if hasattr(self.candidate_live_inference_lock,"snapshot") else {})})
                        self._atomic_json(state,self.candidate_live_state_path)
                        continue
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
                    window=self._window(panel,index,self.candidate_live_model)
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
                        "last_timestamp":stamp,"last_full_decision_timestamp":stamp,
                        "candidate_version":observer_version,
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

    def _window(self,panel,index,model=None):
        contextual=getattr(model if model is not None else self._runtime_model(),"_stockrl_uses_market_context",getattr(self,"_runtime_contextual",False))
        return panel.window(index,self.window,include_context=contextual)

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
                self._sync_models()
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
                        symbol_map=getattr(self._runtime_model(),"_stockrl_symbol_map",getattr(self,"_runtime_symbol_map",None)),
                        # Keep closed-session indices, futures, yields, and other
                        # reference markets in the action set using their last
                        # known quote. The dashboard still marks their quote as
                        # stale; dropping them here hid their BUY/HOLD/SELL
                        # output entirely whenever their venue was closed.
                        recent_timestamps=None, active_stale_seconds=604800,
                        raw_frame=raw_frame)
                    if (self._model_enabled("champion") and self._model_enabled("candidate") and not self.validation_active
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
                    if self.stop.is_set() or (self.state_dir/"stop.request").exists(): break
                    self._sync_models()
                    self.current_market_timestamp=str(panel.dates[ti])
                    # Autonomy is distinct from the model's HOLD action. When
                    # disabled, inference and counterfactual learning continue,
                    # but directional paper positions are left unchanged.
                    mode_state=self._run_modes()
                    paper_enabled=mode_state["paper_enabled"] and self._model_enabled("champion")
                    observe_enabled=mode_state["observe_enabled"] and self._model_enabled("champion")
                    self.metrics["autonomy_enabled"]=paper_enabled
                    self.metrics["paper_enabled"]=paper_enabled
                    self.metrics["observe_enabled"]=observe_enabled
                    filled_orders=[]
                    if self._model_enabled("champion"):
                        filled_orders=self.paper_account.process_bar(panel,ti,paper_enabled)
                        portfolio_pending=self._mature_portfolio(portfolio_pending,panel,ti,filled_orders)
                        pending=self._mature(pending,panel,ti)
                    if not observe_enabled or not self._has_tradable_update(panel,ti):
                        reason="context_only" if observe_enabled else "judgment_paused"
                        key="champion_"+reason+"_updates"
                        self.metrics[key]=int(self.metrics.get(key,0))+1
                        self.metrics["champion_inference_skipped_reason"]=reason
                        # Candidate must still mark its account and mature earlier
                        # actions at this timestamp; it uses the same CPU fast path.
                        uniforms=([self.policy_rng.random() for _ in panel.symbols]
                                  if mode_state["observe_enabled"] and self._has_tradable_update(panel,ti) else ())
                        self._queue_candidate_live_observation(panel,ti,mode_state["paper_enabled"],uniforms)
                        cursor=str(panel.dates[ti]); self.metrics["observations"]+=1
                        self.metrics["last_market_timestamp"]=cursor
                        self.metrics["pending_experiences"] = len(pending)+len(portfolio_pending)
                        self.replay.save_pending(pending,portfolio_pending)
                        self.paper_account.save()
                        _atomic_json({"last_timestamp":cursor},cursor_path)
                        continue
                    pstate,astate=self.paper_account.model_inputs(panel,ti)
                    logits,values,allocation=self._infer(panel,ti,pstate,astate)
                    self.metrics["champion_last_full_decision_timestamp"]=str(panel.dates[ti])
                    self.metrics["champion_inference_skipped_reason"]=None
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
                        self.metrics["daily_history_input_status_utc"]=stamp
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
                        self.metrics["multiscale_input_status_utc"]=stamp
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
