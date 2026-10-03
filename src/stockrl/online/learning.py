"""Ordered replay training, shared forwards and saved model publication."""
from __future__ import annotations
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict
import os, time, gc
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from ..global_transformer import GlobalMarketTransformer
from ..multiscale import MULTISCALE_FEATURE_COUNT, TIMEFRAME_NAMES, TIMEFRAME_FEATURE_NAMES, BASE_MULTISCALE_FEATURE_COUNT, LONG_CONTEXT_NAMES
from .checkpoint import load_model, save_model
from .data import ONLINE_TRAINABLE_BLOCKS, PAPER_EXPLORATION_EPSILON, TrainingMetrics
from ..experience import Experience
from .losses import shared_experience_losses
from .prefix_cache import FrozenPrefixCache, prefix_input, install_prefix_forward
from .optimizer_state import restore_optimizer, remember_optimizer
from .replay_updates import install_replay_updates

class _LearningMixin:
    def _refresh_runtime_update_driver(self):
        """Update the small boundary controller without replacing its state."""
        updater=getattr(self,"runtime_updates",None)
        if updater is None: return
        from pathlib import Path
        import types
        from . import runtime_updates
        path=Path(runtime_updates.__file__)
        stamp=path.stat().st_mtime_ns
        if getattr(updater,"_driver_stamp",None)==stamp: return
        try:
            module=types.ModuleType(runtime_updates.__name__)
            module.__dict__.update(__package__=runtime_updates.__package__,__file__=str(path))
            exec(compile(path.read_text(encoding="utf-8"),str(path),"exec"),module.__dict__)
            updater.__class__=module.RuntimeUpdates
            updater._driver_stamp=stamp
            updater.accepted_signature=updater.signature()
        except Exception as exc:
            self.metrics["runtime_updates"].update(status="rejected",error=f"runtime controller: {exc}"[:800])

    def _new_model_like(self, source, device):
        contextual=bool(getattr(source,"_stockrl_uses_market_context",False))
        source_dtype=next(source.parameters()).dtype
        cfg=getattr(source,"_stockrl_cfg",self.cfg)
        # Every caller immediately loads an existing state_dict. Allocate its
        # final storage directly instead of initializing 0.5B random FP32
        # parameters and then converting/discarding them before that copy.
        with torch.device("meta"):
            if contextual:
                from ..market_training import ContextConditionedTransformer
                backbone=GlobalMarketTransformer(cfg)
                if device.type=="cuda" or source_dtype==torch.float16: backbone=backbone.half()
                model=ContextConditionedTransformer(backbone)
                model.context_policy.float(); model.context_value.float()
                model._stockrl_uses_market_context=True
                model._stockrl_symbol_map=dict(source._stockrl_symbol_map)
            else:
                model=GlobalMarketTransformer(cfg)
                if device.type=="cuda" or source_dtype==torch.float16: model=model.half()
        model._stockrl_cfg=cfg
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
            candidate._stockrl_cfg=cfg
            self.candidate=candidate
            commit=getattr(candidate,"_stockrl_replay_commit",{})
            self.replay.acknowledge_training(commit.get("uses",{}),self.candidate_replay_passes)
            self.candidate_version=max(self.candidate_version,int(commit.get("candidate_version",0)))
        if self.device.type=="cuda": self.candidate.to("cpu")
        self.metrics["candidate_checkpoint_loaded"]=True

    def _pack(self,batch):
        def tensor(key,dtype=None):
            a=np.stack([getattr(x,key) for x in batch]); t=torch.as_tensor(a,device=self.device)
            return t.to(dtype) if dtype else t
        feature_dtype=torch.float16 if self.device.type=="cuda" else torch.float32
        args=(tensor("features",feature_dtype),tensor("symbol_ids",torch.long),tensor("market_ids",torch.long),
              tensor("asset_ids",torch.long),tensor("valid_mask",torch.bool))
        if getattr(self._runtime_model(),"_stockrl_uses_market_context",False):
            from ..market_training import CONTEXT_FEATURES
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
        wait_started=time.perf_counter()
        with self._gpu_work(role):
            learner=role.split("_",1)[0]
            self.metrics[learner+"_learning_gpu_wait_seconds_round"]=float(
                self.metrics.get(learner+"_learning_gpu_wait_seconds_round",0))+time.perf_counter()-wait_started
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
            updater=getattr(self,"runtime_updates",None)
            if updater is not None and updater.pending_change() and (
                self.metrics.get("candidate_training") or self.metrics.get("champion_training")):
                self.metrics["runtime_updates"]["status"]="pending_batch_save"
                self.metrics["learning_wait_reason"]="설정·학습 코드 변경: 현재 배치를 저장한 뒤 재적용"
                return False
            if not self._run_modes()["learning_enabled"]:
                self.metrics["learning_wait_reason"]="사용자가 replay 학습을 껐습니다."
                return False
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
            updater=getattr(self,"runtime_updates",None)
            if updater is not None:
                if updater.poll(self): self._write_metrics()
            if not self._run_modes()["learning_enabled"]:
                self.metrics["learning_wait_reason"]="사용자가 replay 학습을 껐습니다."
                self.stop.wait(.5)
                continue
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
            remaining={role:count if self._model_enabled(role) else 0 for role,count in remaining.items()}
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
                with self.role_locks[learner]:
                    if not self._model_enabled(learner):continue
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
                    if not self._model_requests()[learner]:continue
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
                with torch.no_grad(), prefix_input(model,successor,getattr(self,"_frozen_prefix_cache",None)):
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
        global prefix_input
        self._refresh_runtime_update_driver()  # includes elapsed reward settlement updates
        from . import prefix_cache as prefix_module
        prefix_stamp=os.stat(prefix_module.__file__).st_mtime_ns
        if getattr(prefix_module,"_hot_source_stamp",None)!=prefix_stamp:
            import importlib
            importlib.reload(prefix_module)
            prefix_module._hot_source_stamp=prefix_stamp
        with self._gpu_work(learner+"_learning_setup"):
            prefix_input=prefix_module.install_prefix_forward()
        install_replay_updates(self.replay)  # Keep live counters aligned with durable rows.
        if not hasattr(self,"_frozen_prefix_cache"):
            self._frozen_prefix_cache=FrozenPrefixCache()
        # Retain current/successor encodings for both independent accounts.
        # Each insertion still checks available VRAM before keeping a tensor.
        self._frozen_prefix_cache.max_bytes=512*1024*1024
        self._frozen_prefix_cache.max_entries=8
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
        self.metrics[learner+"_learning_gpu_wait_seconds_round"]=0.0
        setup_started=time.perf_counter()
        with self.lock:
            source_champion=self.champion if learner=="champion" else self.candidate
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
        trainable_parameters=self._configure_candidate_trainables(candidate,learner)
        original_parameters={name:value.detach().to("cpu",copy=True) for name,value in candidate.named_parameters() if value.requires_grad}
        opt=torch.optim.AdamW(
            trainable_parameters,lr=self.lr,weight_decay=.01,
            eps=1e-4 if self.device.type=="cuda" else 1e-8,
            **({"fused":True} if self.device.type=="cuda" else {"foreach":False}))
        metrics["candidate_optimizer_state_resumed"]=restore_optimizer(self,learner,opt,model_version)
        metrics["candidate_optimizer_backend"]="fused_adamw" if self.device.type=="cuda" else "adamw"
        metrics["candidate_loss_backend"]="shared_window_vectorized"
        setup_seconds=time.perf_counter()-setup_started
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            training_baseline_allocated=int(torch.cuda.memory_allocated(self.device))
        else:
            training_baseline_allocated=0
        elapsed=[]; compute_elapsed=[]
        batch_load_seconds=0.0; all_losses=[]; gradient_norms=[]
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
            if self.stop.is_set() or not self._model_requests()[learner] or not self._wait_for_live_inference():
                break
            load_started=time.perf_counter()
            batch=self.replay.pending_batch(self.batch_size,self.candidate_replay_passes,
                exclude_row_ids=sampled_ids,learner=learner)
            batch_load_seconds+=time.perf_counter()-load_started
            if not batch: break
            sampled_ids.update(self.replay.row_ids_for(batch))
            malformed_by_reason={}
            for experience in batch:
                symbols=experience.features.shape[1]
                shapes={
                    "goal_state":(np.shape(experience.goal_state),(6,)),
                    "portfolio_state":(np.shape(experience.portfolio_state),(symbols,8)),
                    "account_state":(np.shape(experience.account_state),(8,)),
                    "multiscale_state":(np.shape(experience.multiscale_state),None),
                }
                issues=[]
                for name,(actual,expected) in shapes.items():
                    if getattr(experience,name) is None:
                        continue
                    valid=(actual==expected) if expected is not None else actual in (
                        (symbols,MULTISCALE_FEATURE_COUNT),(symbols,96),
                        (symbols,BASE_MULTISCALE_FEATURE_COUNT))
                    if not valid:
                        issues.append(f"{name} expected {expected if expected is not None else 'supported multiscale shape'} got {actual}")
                if issues:
                    reason="saved portfolio input shape is incompatible: "+"; ".join(issues)
                    malformed_by_reason.setdefault(reason,[]).append(experience)
            malformed=[experience for group in malformed_by_reason.values() for experience in group]
            if malformed:
                for reason,group in malformed_by_reason.items():
                    self.replay.quarantine(self.replay.row_ids_for(group),reason)
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
                with self._learning_gpu_segment(learner+"_learning_step",step_segments), prefix_input(candidate,group[0],self._frozen_prefix_cache):
                    args,pstate,astate,mstate=self._pack([group[0]])
                    if use_portfolio:
                        logits,values,allocations=candidate(*args,portfolio_state=pstate,
                            account_state=astate,return_allocation=True,multiscale_state=mstate,
                            **self._saved_goal_kwargs(group[0],self.device),
                            **({"daily_history":torch.as_tensor(group[0].daily_history[None],device=self.device,dtype=torch.float32)}
                               if getattr(group[0],"daily_history",None) is not None else {}))
                    else:
                        logits,values=candidate(*args); allocations=None
                    successors=[]
                    for experience in group:
                        successor_values=None
                        if experience.bootstrap_discount:
                            successor_values=self._credit_successor_values(candidate,experience,successor_cache)
                        successors.append(successor_values)
                    losses,accepted,rejected=shared_experience_losses(
                        logits,values,allocations,group,successors)
                    self.metrics["nonfinite_updates"]+=rejected
                    successful_batch.extend(accepted)
                    if accepted:
                        (losses.sum()/len(batch)).backward()
                        valid_samples+=len(accepted)
                        loss_values.extend(losses.detach().cpu().tolist())
                    metrics["candidate_window_forwards_current"]+=1
                    del args,pstate,astate,mstate,logits,values,allocations,losses
            if not valid_samples:
                continue
            with self._learning_gpu_segment(learner+"_learning_step",step_segments):
                gradient_norm=nn.utils.clip_grad_norm_(trainable_parameters,1.0,foreach=True)
                with self.candidate_model_lock:
                    opt.step()
                    if self.device.type=="cuda": torch.cuda.synchronize(self.device)
                gradient_norms.append(float(gradient_norm.detach().cpu()))
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
            all_losses.extend(loss_values)
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
        if not torch.stack([torch.isfinite(p).all() for p in candidate.parameters()]).all().item():
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
        with torch.no_grad():
            delta_tensor=torch.zeros((),device=self.device,dtype=torch.float32)
            current=dict(candidate.named_parameters())
            for k,v in original_parameters.items():
                delta_tensor.add_((current[k].float()-v.to(self.device).float()).abs().sum())
        del original_parameters
        delta=delta_tensor.item()
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
        checkpoint_started=time.perf_counter()
        if learner=="champion":
            staged=self.state_dir/"champion.learning.next"
            try:
                save_model(staged,candidate,getattr(candidate,"_stockrl_cfg",self.cfg),step=self.steps,
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
            save_model(self.model_dir/"candidate.pt",candidate,getattr(candidate,"_stockrl_cfg",self.cfg),step=self.steps,
                       temp_dir=self.state_dir,replay_commit=replay_commit)
            candidate._stockrl_replay_commit=replay_commit
        checkpoint_seconds=time.perf_counter()-checkpoint_started
        optimizer_state_started=time.perf_counter()
        remember_optimizer(self,learner,opt,replay_commit["model_version"])
        optimizer_state_seconds=time.perf_counter()-optimizer_state_started
        acknowledge_started=time.perf_counter()
        consumed=self.replay.acknowledge_training(uses,self.candidate_replay_passes,learner=learner)
        acknowledge_seconds=time.perf_counter()-acknowledge_started
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
        if learner=="candidate" and self._model_enabled("champion") and self._model_enabled("candidate") and not self.validation_active and not self.stop.is_set():
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
            "setup_seconds":setup_seconds,
            "batch_load_seconds":batch_load_seconds,
            "gpu_wait_seconds":self.metrics.get(learner+"_learning_gpu_wait_seconds_round",0.0),
            "window_forwards":metrics.get("candidate_window_forwards_current",0),
            "loss_mean":float(np.mean(all_losses)) if all_losses else None,
            "loss_min":float(np.min(all_losses)) if all_losses else None,
            "loss_max":float(np.max(all_losses)) if all_losses else None,
            "gradient_norm_mean":float(np.mean(gradient_norms)) if gradient_norms else None,
            "replay_rows_deleted":consumed,
            "checkpoint_seconds":checkpoint_seconds,
            "optimizer_state_seconds":optimizer_state_seconds,
            "replay_acknowledge_seconds":acknowledge_seconds,
            "optimizer_state_resumed":metrics["candidate_optimizer_state_resumed"],
            "compute_seconds":metrics["last_candidate_compute_seconds"],
            "samples_per_compute_second":candidate_samples/max(metrics["last_candidate_compute_seconds"],1e-9),
            "samples_per_total_second":candidate_samples/max(metrics["last_candidate_total_seconds"],1e-9),
            "optimizer_backend":metrics.get("candidate_optimizer_backend"),
            "attention_backend":"sdpa" if getattr(candidate,"backbone",candidate).blocks[0]._stockrl_sdpa_enabled else "mha",
            "loss_backend":metrics.get("candidate_loss_backend"),
            "frozen_prefix_cache":self._frozen_prefix_cache.snapshot(),
            "step_compute_seconds":metrics["last_candidate_step_compute_seconds"],
            "peak_allocated_bytes":metrics["last_candidate_peak_allocated_bytes"],
            "peak_reserved_bytes":metrics["last_candidate_peak_reserved_bytes"],
            "completed_utc":metrics["last_update_utc"],
            "model_version":replay_commit["model_version"]}
        metrics["candidate_last_completed_round"]["timeframe_samples"]=timeframe_samples
        metrics["candidate_last_completed_round"]["long_context_samples"]=long_context_samples
        metrics["candidate_last_completed_round"]["daily_history_samples"]=daily_history_samples
        self.candidate_retry_attempts=0; self.candidate_retry_after=0.0
        self._write_metrics()
        del candidate,opt
