"""Frozen model competition and promotion."""
from __future__ import annotations
from collections import deque
from pathlib import Path
import json, os, time, queue
import hashlib
import numpy as np
import torch
from ..global_transformer import ACTION_NAMES
from ..paper_account import _currency
from ..state_io import atomic_json
from .checkpoint import load_model, save_model
from .data import MarketObservation

def should_promote(candidate_score:float, champion_score:float, minimum_delta:float=0.0)->bool:
    """Finite, strict promotion gate shared by online control and its checks."""
    return bool(np.isfinite(candidate_score) and np.isfinite(champion_score)
                and candidate_score>0 and candidate_score>champion_score+minimum_delta)


class _ValidationMixin:
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
        if not self._run_modes()["observe_enabled"] or not self._has_tradable_update(panel,index):
            return
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
            if not self._run_modes()["observe_enabled"]:
                self.stop.wait(.25)
                continue
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
