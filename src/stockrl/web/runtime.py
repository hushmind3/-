"""Feed/agent lifecycle and independent runtime controls."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from ..paths import ensure_project_path, validate_model_dir
from ..operating_rules import operating_rules
from .health import _json
from .resources import ROOT

from .status import _StatusMixin
from .accounts import _AccountResetMixin

class Supervisor(_StatusMixin, _AccountResetMixin):
    def __init__(self, runtime: Path, device: str, candidate_every: int, fee: float,
                 horizon: str = "1m", config: str = "configs/live_symbols.json",
                 initial_champion: str | None = None, model_dir: str | Path | None = None,
                 settings_dir: str | Path | None = None):
        self.runtime = ensure_project_path(runtime, "runtime")
        self.device = device
        self.candidate_every, self.fee = candidate_every, fee
        self.config = Path(config)
        if not self.config.is_absolute():
            self.config = ROOT / self.config
        self.initial_champion = Path(initial_champion) if initial_champion else None
        if self.initial_champion is not None and not self.initial_champion.is_absolute():
            self.initial_champion = ROOT / self.initial_champion
        requested_model_dir = model_dir or os.environ.get("STOCKRL_MODEL_DIR")
        self.model_dir = validate_model_dir(requested_model_dir)
        self.lock = threading.RLock()
        self.profile: Path | None = None
        self.mode = "live"
        self.children: dict[str, subprocess.Popen] = {}
        self.run_requested = False
        self.stopping = False
        self.agent_reload_pending = False
        self.daily_cycle_pending=False
        self.account_reset_lock=threading.Lock()
        self.operating_rules=operating_rules()
        self.restart_request: tuple[str, str | None] | None = None
        self.log_tail: list[str] = []
        self.log_handles = {}
        self._market_row_cache = {"path": None, "offset": 0, "lines": 0}
        self._latest_csv_cache = {}
        self._gpu_snapshot = {"sampled": 0.0}
        self.settings_path = (ensure_project_path(settings_dir, "settings") / "web_settings.json"
                              if settings_dir else ROOT / "configs" / "local" / "web_settings.json")
        ensure_project_path(self.settings_path, "settings")
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings = _json(self.settings_path)
        self.mode = settings.get("mode", "live")
        self.horizon = settings.get("horizon", horizon)
        self.autonomy_enabled = bool(settings.get("paper_enabled", settings.get("autonomy_enabled", True)))
        self.observe_enabled = bool(settings.get("observe_enabled", True))
        self.learning_enabled = bool(settings.get("learning_enabled", True))
        self.model_enabled={role:bool(settings.get(role+"_enabled",False)) for role in ("champion","candidate")}
        self.model_request_versions=dict(settings.get("model_request_versions",{}))
        self.model_families=dict(settings.get("model_families",{}))
        self.moe_model_workers={}
        self.idle_agent_shutdown=False
        self.idle_agent_deadline=None
        from .workers import adopt
        adopt(self)
        self.worker = threading.Thread(target=self._monitor, daemon=True, name="web-supervisor")
        self.worker.start()

    def _log(self, line: str):
        self.log_tail.append(line)
        self.log_tail = self.log_tail[-30:]

    def _spawn(self, name: str, args: list[str], log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists() and log_path.stat().st_size >= 8*1024*1024:
            log_path.write_text("",encoding="utf-8")
        handle = log_path.open("a", encoding="utf-8", buffering=1)
        self.log_handles[name] = handle
        self.children[name] = subprocess.Popen(args, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._log(f"{time.strftime('%H:%M:%S')} started: {name} (PID {self.children[name].pid})")

    def _launch(self, name: str):
        assert self.profile is not None
        if name in ("champion","candidate"):
            from .workers import AttachedWorker
            worker=self._moe_model_worker(name)
            result=worker.start()
            if not result.get("ok"):raise RuntimeError(result.get("error"))
            self.children[name]=AttachedWorker(worker.read(worker.record))
            return
        data = self.profile / "market.csv"
        stop = self.profile / "feed.stop"
        state = self.profile / "agent"
        if name == "feed":
            if self.mode == "mock":
                source = ROOT / "data/global_market_daily.csv"
                args = [sys.executable, "-u", "-m", "stockrl", "mock-feed", "--source", str(source),
                        "--output", str(data), "--bars", "24", "--interval-seconds", "0.25", "--stop-file", str(stop)]
            else:
                args = [sys.executable, "-u", "-m", "stockrl", "live-feed", "--config",
                        str(self.config), "--output", str(data), "--poll-seconds", "15",
                        "--stop-file", str(stop)]
            self._spawn(name, args, self.profile / "logs" / "feed.log")
            return
        args = [sys.executable, "-u", "-m", "stockrl", "global-online", "--data", str(data),
                "--independent-models",
                "--state-dir", str(state), "--model-dir", str(self.model_dir), "--follow", "--poll-seconds", "1", "--window", "128",
                "--initial-lookback-bars", "8",
                "--candidate-every", str(self.candidate_every),
                "--batch-size", str(self.operating_rules["training_batch_size"]),
                "--updates", str(self.operating_rules["training_optimizer_steps"]),
                "--fee", str(self.fee), "--horizon", self.horizon,
                "--device", self._device()]
        seed = self.initial_champion or (self.model_dir / "champion.pt")
        if self.mode=="live":
            args.extend(["--instrument-config",str(self.config)])
        if seed.is_file():
            args.extend(["--initial-champion", str(seed)])
        self._spawn(name, args, self.profile / "logs" / "agent.log")

    def _device(self):
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def start(self, mode: str = "live", horizon: str | None = None) -> dict:
        if mode not in ("live", "mock"):
            return {"error": "mode must be live or mock"}
        from ..global_online import parse_horizon
        try:
            parse_horizon(horizon or self.horizon)
        except ValueError as exc:
            return {"error": str(exc)}
        with self.lock:
            if self.stopping:
                return {"error": "System is stopping; wait for completion."}
            if self.run_requested:
                return {"ok": True, "message": "System is already running."}
            self.mode, self.profile = mode, self.runtime / mode
            # Bundled mock history is daily, so one source bar is its meaningful horizon.
            self.horizon = "1bar" if mode == "mock" else (horizon or self.horizon)
            self.profile.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(json.dumps({"mode": mode, "horizon": self.horizon,
                "autonomy_enabled":self.autonomy_enabled,"paper_enabled":self.autonomy_enabled,
                "observe_enabled":self.observe_enabled,"learning_enabled":self.learning_enabled}, ensure_ascii=False, indent=2), encoding="utf-8")
            self._write_autonomy()
            for path in (self.profile / "feed.stop", self.profile / "agent" / "stop.request"):
                path.unlink(missing_ok=True)
            self.run_requested = True
            self.stopping = False
            self.children.clear()
            self.model_enabled={role:False for role in self.model_enabled}
            self._write_autonomy()
            self._launch("feed")
            return {"ok": True, "message": "Market feed started; both models remain unloaded."}

    def set_model(self,role,enabled):
        if role not in ("champion","candidate"):return {"error":"Unknown model role"}
        with self.lock:
            if self.stopping:return {"error":"System is saving; wait for completion."}
            if enabled and not self.run_requested:
                result=self.start(self.mode,self.horizon)
                if not result.get("ok"):return result
            state=_json((self.profile or self.runtime/self.mode)/"agent"/"metrics.json").get("models",{}).get(role,{})
            if self.model_families.get(role)=="trading_moe":state=self._moe_model_worker(role).status()
            if self.model_enabled[role]==bool(enabled) and state.get("status") not in ("error",) and (not enabled or state.get("status")!="stopped"):
                return {"ok":True,"already_requested":True}
            self.model_enabled[role]=bool(enabled)
            self.model_request_versions[role]=time.time_ns()
            if enabled:
                import torch
                try:metadata=torch.load(self.model_dir/(role+".pt"),map_location="cpu",weights_only=False,mmap=True)
                except Exception as exc:
                    self.model_enabled[role]=False
                    self._write_autonomy()
                    return {"error":f"{role}.pt could not be read: {exc}"}
                self.model_families[role]="trading_moe" if "feature_sizes" in metadata.get("config",{}) else "legacy"
                del metadata
            self._write_autonomy()
            if enabled:
                if self.model_families.get(role)=="trading_moe":
                    self._launch(role)
                    return {"ok":True,"role":role,"requested":True,"message":"Named TradingMoE model load requested"}
                self.idle_agent_shutdown=False
                agent=self.children.get("agent")
                if agent is None or agent.poll() is not None:
                    (self.profile/"agent"/"stop.request").unlink(missing_ok=True)
                    self._launch("agent")
            elif self.model_families.get(role)=="trading_moe":
                self._moe_model_worker(role).stop()
            return {"ok":True,"role":role,"requested":bool(enabled),
                    "message":"Load requested" if enabled else "Save and unload requested"}

    def _moe_model_worker(self,role):
        from .trading_moe import TradingMoELifecycle
        state=(self.profile or self.runtime/self.mode)/"agent"/(role+"_moe")
        worker=self.moe_model_workers.get(role)
        if worker is None or worker.state!=state:
            worker=TradingMoELifecycle(checkpoint=self.model_dir/(role+".pt"),state=state)
            self.moe_model_workers[role]=worker
        return worker

    def _write_autonomy(self):
        profile=self.profile or (self.runtime/self.mode)
        target=profile/"agent"/"autonomy.json"
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"enabled":self.autonomy_enabled,
                                          "paper_enabled":self.autonomy_enabled,
                                          "observe_enabled":self.observe_enabled,
                                          "learning_enabled":self.learning_enabled,
                                          **{role+"_enabled":value and self.model_families.get(role)!="trading_moe" for role,value in self.model_enabled.items()},
                                          "model_request_versions":self.model_request_versions},ensure_ascii=False),encoding="utf-8")
        temporary.replace(target)
        settings=_json(self.settings_path)
        settings.update({"mode":self.mode,"horizon":self.horizon,"autonomy_enabled":self.autonomy_enabled,
                         "paper_enabled":self.autonomy_enabled,"observe_enabled":self.observe_enabled,
                         "learning_enabled":self.learning_enabled})
        settings.update({role+"_enabled":value for role,value in self.model_enabled.items()})
        settings["model_request_versions"]=self.model_request_versions
        settings["model_families"]=self.model_families
        self.settings_path.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding="utf-8")

    def set_autonomy(self, enabled: bool) -> dict:
        with self.lock:
            self.autonomy_enabled=bool(enabled)
            self._write_autonomy()
            return {"ok":True,"autonomy_enabled":self.autonomy_enabled,
                    "paper_enabled":self.autonomy_enabled,"observe_enabled":self.observe_enabled}

    def set_modes(self, paper_enabled=None, observe_enabled=None, learning_enabled=None) -> dict:
        with self.lock:
            if paper_enabled is not None:
                self.autonomy_enabled = bool(paper_enabled)
            if observe_enabled is not None:
                self.observe_enabled = bool(observe_enabled)
            if learning_enabled is not None:
                self.learning_enabled = bool(learning_enabled)
            self._write_autonomy()
            return {"ok": True, "paper_enabled": self.autonomy_enabled,
                    "observe_enabled": self.observe_enabled,"learning_enabled":self.learning_enabled}

    def restart(self, mode: str, horizon: str | None = None) -> dict:
        from ..global_online import parse_horizon
        if mode not in ("live", "mock"):
            return {"error": "mode must be live or mock"}
        try:
            parse_horizon(horizon or self.horizon)
        except ValueError as exc:
            return {"error": str(exc)}
        with self.lock:
            if self.stopping:
                return {"error": "System is stopping; wait for completion."}
            if not self.run_requested:
                return self.start(mode, horizon)
            self.restart_request = (mode, horizon)
            self.stop(keep_restart=True)
            return {"ok": True, "message": "Restart requested; current state is being saved."}

    def reload_feed(self) -> None:
        """Apply credential/provider changes without interrupting the model."""
        with self.lock:
            if self.run_requested and self.mode == "live" and self.profile:
                (self.profile / "feed.stop").touch()
                self._log(f"{time.strftime('%H:%M:%S')} market feed reloading after provider change")

    def reload_agent(self) -> dict:
        """Gracefully reload model code/settings while keeping feed and web alive."""
        with self.lock:
            if not self.run_requested or not self.profile:
                return {"error": "Start the system before reloading the model."}
            if self.stopping:
                return {"error": "System shutdown is already in progress."}
            agent = self.children.get("agent")
            if agent is None or agent.poll() is not None:
                return {"error": "Model process is not running."}
            stop_request = self.profile / "agent" / "stop.request"
            if stop_request.exists():
                return {"error": "Model reload is already in progress."}
            agent_metrics = _json(self.profile / "agent" / "metrics.json")
            try:
                updated_rules = operating_rules()
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                return {"error": f"Training settings were not applied: {exc}"}
            self.operating_rules = updated_rules
            if agent_metrics.get("candidate_validation_active"):
                self.agent_reload_pending = True
                self._log(f"{time.strftime('%H:%M:%S')} model reload queued until the active promotion trial finishes")
                return {"ok": True, "pending": True,
                        "message": "Model reload queued until the active promotion trial finishes."}
            stop_request.touch()
            self._log(f"{time.strftime('%H:%M:%S')} model reload requested; replay/account state will be saved, feed stays running")
            return {"ok": True, "message": "Model is saving state and will restart; feed and web stay running."}

    def stop(self, keep_restart: bool = False):
        with self.lock:
            if not keep_restart:
                self.restart_request = None
            if self.stopping:
                return {"ok": True, "message": "System is already stopping."}
            self.run_requested = False
            self.stopping = True
            self.model_enabled={role:False for role in self.model_enabled}
            self._write_autonomy()
            for role in ("champion","candidate"):
                if self.model_families.get(role)=="trading_moe":self._moe_model_worker(role).stop()
            self.agent_reload_pending = False
            if self.profile:
                (self.profile / "feed.stop").touch()
                (self.profile / "agent" / "stop.request").parent.mkdir(parents=True, exist_ok=True)
                (self.profile / "agent" / "stop.request").touch()
            threading.Thread(target=self._stop_children, daemon=True).start()
        return {"ok": True, "message": "Stopping; learning state is being saved."}

    def _stop_children(self):
        for name, proc in list(self.children.items()):
            try:
                proc.wait(timeout=90 if name in ("agent","champion","candidate") else 8)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        with self.lock:
            for handle in self.log_handles.values():
                try:
                    handle.close()
                except OSError:
                    pass
            self.log_handles.clear()
            self.children.clear()
            self.stopping = False
            request = self.restart_request
            self.restart_request = None
            self._log("Stopped; learning state saved.")
            if request is not None:
                result = self.start(*request)
                self._log("Restarted." if result.get("ok") else f"Restart failed: {result.get('error')}")

    def _monitor(self):
        while True:
            time.sleep(2)
            if self.run_requested and all(self.model_enabled.values()) and "trading_moe" not in self.model_families.values() and not self.stopping and self.mode=="live" and not self.daily_cycle_pending:
                cycle=self._daily_cycle_status()
                if cycle["session_key"]!=cycle["current_session_key"] or cycle.get("pending_record"):
                    self.daily_cycle_pending=True
                    def run_cycle(key=cycle["current_session_key"],daily=not cycle.get("pending_record") or cycle["pending_record"].get("reason")=="scheduled_daily"):
                        try:
                            result=self.reset_paper_accounts(daily=daily,session_key=key)
                            self._log("Daily competition completed: "+str(result))
                        finally:
                            self.daily_cycle_pending=False
                    threading.Thread(target=run_cycle,daemon=True,name="daily-competition").start()
            with self.lock:
                if not self.run_requested or not self.profile:
                    continue
                agent=self.children.get("agent")
                legacy_requested=any(enabled and self.model_families.get(role)!="trading_moe" for role,enabled in self.model_enabled.items())
                if agent is not None and agent.poll() is None and not legacy_requested:
                    models=_json(self.profile/"agent"/"metrics.json").get("models",{})
                    if models and all(not state.get("loaded") and state.get("status")=="stopped" for state in models.values()):
                        self.idle_agent_shutdown=True
                        (self.profile/"agent"/"stop.request").touch()
                        if self.idle_agent_deadline is None:self.idle_agent_deadline=time.monotonic()+5
                        # Both roles have already saved and unloaded. Reclaim an
                        # idle coordinator's CUDA context if old code is catching up.
                        elif time.monotonic()>=self.idle_agent_deadline:agent.terminate()
                else:self.idle_agent_deadline=None
                if self.agent_reload_pending and not self.stopping:
                    agent = self.children.get("agent")
                    if agent is not None and agent.poll() is None:
                        agent_metrics = _json(self.profile / "agent" / "metrics.json")
                        if not agent_metrics.get("candidate_validation_active"):
                            stop_request = self.profile / "agent" / "stop.request"
                            if not stop_request.exists():
                                stop_request.touch()
                            self.agent_reload_pending = False
                            self._log(f"{time.strftime('%H:%M:%S')} queued model reload started after promotion trial")
                for name, proc in list(self.children.items()):
                    code = proc.poll()
                    if code is None:
                        continue
                    self._log(f"{time.strftime('%H:%M:%S')} {name} exited ({code}); restarting")
                    self.children.pop(name, None)
                    handle = self.log_handles.pop(name, None)
                    if handle:
                        handle.close()
                    if name == "feed" and self.mode == "mock" and code == 0:
                        continue
                    if name=="agent" and self.idle_agent_shutdown:
                        self.idle_agent_shutdown=False
                        self.idle_agent_deadline=None
                        if not legacy_requested:continue
                    if name in ("champion","candidate") and not self.model_enabled[name]:continue
                    if name=="agent" and not legacy_requested:continue
                    if name == "feed":
                        (self.profile / "feed.stop").unlink(missing_ok=True)
                    time.sleep(.1)
                    self._launch(name)
