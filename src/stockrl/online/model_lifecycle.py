"""Independent model residency inside the existing shared online runtime."""
import gc
import threading
import torch
from .checkpoint import load_model, save_model, _atomic_json, _atomic_save


class _ModelLifecycleMixin:
    def _init_model_lifecycle(self, independent):
        self.independent_models=independent
        self.role_locks={role:threading.RLock() for role in ("champion","candidate")}
        self._failed_model_requests={}
        self.model_states={role:{"status":"stopped","loaded":False,"error":None}
                           for role in self.role_locks}

    def _model_requests(self):
        if not self.independent_models:return {role:True for role in self.role_locks}
        try:
            import json
            data=json.loads((self.state_dir/"autonomy.json").read_text(encoding="utf-8"))
        except (OSError,ValueError):data={}
        return {role:bool(data.get(role+"_enabled",False)) for role in self.role_locks}

    def _model_enabled(self,role):
        if not self.independent_models:return True
        return (self._model_requests()[role] and getattr(self,role,None) is not None
                and self.model_states[role]["status"]=="running")

    def _runtime_model(self):
        return self.champion if self.champion is not None else self.candidate

    def _sync_models(self):
        if not self.independent_models:return
        requests=self._model_requests()
        for role,wanted in requests.items():
            model=getattr(self,role)
            state=self.model_states[role]
            if wanted and model is not None:continue
            if wanted and state["status"]=="error" and self._failed_model_requests.get(role)==self._model_request_versions().get(role):continue
            if not wanted and model is None:
                if state["status"]!="error":state.update(status="stopped",loaded=False)
                continue
            if not wanted:
                state["status"]="saving"
                self._write_metrics()
            # A training step or live decision finishes before releasing weights.
            if not self.role_locks[role].acquire(blocking=False):continue
            try:
                if wanted:
                    state.update(status="loading",error=None,loaded=False)
                    self._write_metrics()
                    path=self.model_dir/(role+".pt")
                    if not path.is_file():raise FileNotFoundError(f"{role} checkpoint missing: {path}")
                    loaded,cfg=load_model(path,self.device,self.instrument_config)
                    self.cfg=cfg
                    loaded._stockrl_cfg=cfg
                    setattr(self,role,loaded)
                    commit=getattr(loaded,"_stockrl_replay_commit",{})
                    self.replay.acknowledge_training(commit.get("uses",{}),self.candidate_replay_passes,learner=role)
                    if role=="candidate":
                        self.candidate_version=max(self.candidate_version,int(commit.get("candidate_version",0)))
                        self._publish_candidate_observer(loaded,self.candidate_version)
                        if self.device.type=="cuda":loaded.to("cpu")
                    else:self.champion_training_version=max(self.champion_training_version,int(commit.get("model_version",0)))
                    optimizer_path=self.model_dir/(role+".optimizer.pt")
                    if optimizer_path.exists():
                        self.__dict__.setdefault("_optimizer_states",{})[role]=torch.load(optimizer_path,map_location="cpu",weights_only=False)
                    state.update(status="running",loaded=True,error=None)
                else:
                    self._unload_model(role)
                    state.update(status="stopped",loaded=False,error=None)
            except Exception as exc:
                state.update(status="error",error=f"{type(exc).__name__}: {exc}",loaded=getattr(self,role) is not None)
                # Failed starts require another explicit request, not a reload loop.
                self._failed_model_requests[role]=self._model_request_versions().get(role)
            finally:self.role_locks[role].release()
            self._write_metrics()

    def _model_request_versions(self):
        try:
            import json
            return json.loads((self.state_dir/"autonomy.json").read_text(encoding="utf-8")).get("model_request_versions",{})
        except (OSError,ValueError):return {}

    def _unload_model(self,role):
        # Cancel the trial without a promotion; long-term accounts are preserved.
        with self.validation_lock:
            self.validation_active=False
            self.validation_generation+=1
            self.validation_champion=None
            self.validation_candidate=None
        self._atomic_json({"status":"paused","reason":role+" stopped","bars":self.validation_bars},self.validation_state_path)
        model=getattr(self,role)
        commit=getattr(model,"_stockrl_replay_commit",{})
        save_model(self.model_dir/(role+".pt"),model,getattr(model,"_stockrl_cfg",self.cfg),replay_commit=commit,temp_dir=self.state_dir)
        optimizer=getattr(self,"_optimizer_states",{}).pop(role,None)
        if optimizer is not None:_atomic_save(optimizer,self.model_dir/(role+".optimizer.pt"),temp_dir=self.state_dir)
        if role=="candidate":
            with self.candidate_live_model_lock:
                self.candidate_live_model=None
                self.candidate_live_model_version=None
            self.candidate_live_account.save()
            self.replay.save_pending_kind("candidate_portfolio",self.candidate_portfolio_pending)
        else:self.paper_account.save()
        setattr(self,role,None)
        cache=getattr(self,"_frozen_prefix_cache",None)
        if cache is not None:
            # Encodings can retain device tensors belonging to the stopped model.
            del self._frozen_prefix_cache
        del cache
        del model
        gc.collect()
        if self.device.type=="cuda":torch.cuda.empty_cache()

    def _model_residency(self):
        result={}
        for role in self.role_locks:
            model=getattr(self,role,None)
            models=[model]
            if role=="candidate":models.append(getattr(self,"candidate_live_model",None))
            models.append(getattr(self,"validation_"+role,None))
            cpu=gpu=0;seen=set()
            for module in models:
                if module is None:continue
                for tensor in list(module.parameters())+list(module.buffers()):
                    storage=tensor.untyped_storage()
                    key=(str(tensor.device),storage.data_ptr())
                    if key in seen:continue
                    seen.add(key)
                    if tensor.device.type=="cuda":gpu+=storage.nbytes()
                    else:cpu+=storage.nbytes()
            state=dict(self.model_states[role])
            if not self.independent_models:state.update(status="running" if model is not None else "stopped",loaded=model is not None)
            state.update(loaded=model is not None,ram_weight_bytes=cpu,gpu_weight_bytes=gpu,
                         compute_device=str(self.device),
                         device=str(next(model.parameters()).device) if model is not None else None,
                         last_decision=self.metrics.get("champion_last_full_decision_timestamp" if role=="champion" else "candidate_live_last_timestamp"))
            result[role]=state
        return result
