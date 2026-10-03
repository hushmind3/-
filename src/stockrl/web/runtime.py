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
        settings = _json(self.settings_path) or _json(ROOT / "configs" / "web_settings.default.json")
        self.mode = settings.get("mode", "live")
        self.horizon = settings.get("horizon", horizon)
        self.autonomy_enabled = bool(settings.get("paper_enabled", settings.get("autonomy_enabled", True)))
        self.observe_enabled = bool(settings.get("observe_enabled", True))
        self.learning_enabled = bool(settings.get("learning_enabled", True))
        self.model_enabled={role:bool(settings.get(role+"_enabled",False)) for role in ("champion","candidate")}
        self.model_request_versions=dict(settings.get("model_request_versions",{}))
        self.model_families={role:"trading_moe" for role in self.model_enabled}
        self.moe_model_workers={}
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
        raise ValueError("Unknown runtime role: " + name)


    def start(self, mode: str = "live", horizon: str | None = None) -> dict:
        if mode not in ("live", "mock"):
            return {"error": "mode must be live or mock"}
        from ..experience import parse_horizon
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

    def set_model(self, role, enabled):
        if role not in self.model_enabled:return {"error":"Unknown model role"}
        assembly=getattr(self,"assembly_orchestrator",None)
        if role=="candidate" and assembly is not None and assembly.worker is not None:
            if assembly.worker.runner_script=="run_assembly_trial.py" and assembly.worker.process():
                if not enabled:return assembly.trial(False)
                return {"error":"Candidate 조립 시험 실행 중입니다. 시험을 정지한 뒤 운영 모델을 시작하세요."}
        with self.lock:
            if self.stopping:return {"error":"System is saving; wait for completion."}
            if enabled and not self.run_requested:
                result=self.start(self.mode,self.horizon)
                if not result.get("ok"):return result
            worker=self._moe_model_worker(role)
            if role=="candidate" and not worker.process():
                worker.runner_script="run_native_vertical_trading.py"
                worker.extra_args=[]
                recipe=ROOT/"runtime/assembly/current_recipe.json"
                if recipe.is_file() and _json(recipe).get("enabled_experts"):
                    worker.extra_args=["--recipe",str(recipe)]
                if assembly is not None and assembly.worker is worker:assembly.worker=None
            state=worker.status()
            if enabled and state.get("alive"):
                if state.get("stop_requested"):return {"error":"Model is saving; wait before starting."}
                self.model_enabled[role]=True
                self._write_autonomy()
                return {"ok":True,"already_requested":True}
            self.model_enabled[role]=bool(enabled)
            self.model_request_versions[role]=time.time_ns()
            self._write_autonomy()
            if enabled:
                try:self._launch(role)
                except (RuntimeError,OSError,ValueError) as exc:
                    self.model_enabled[role]=False
                    self._write_autonomy()
                    return {"error":str(exc)}
            else:worker.stop()
            return {"ok":True,"role":role,"requested":bool(enabled),
                    "message":"TradingMoE load requested" if enabled else "Save and unload requested"}

    def _moe_model_worker(self,role):
        from .trading_moe import TradingMoELifecycle
        state=(self.profile or self.runtime/self.mode)/"agent"/(role+"_moe")
        worker=self.moe_model_workers.get(role)
        if worker is None or worker.state!=state:
            worker=TradingMoELifecycle(checkpoint=self.model_dir/"champion.pt",state=state)
            self.moe_model_workers[role]=worker
        if role=="champion":
            recipe=ROOT/"runtime/assembly/champion_recipe.json"
            if recipe.is_file() and _json(recipe).get("enabled_experts"):
                worker.extra_args=["--recipe",str(recipe)]
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
                                          **{role+"_enabled":value for role,value in self.model_enabled.items()},
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
        from ..experience import parse_horizon
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
        # Retained endpoint: model code is applied by explicit save/stop/start.
        return {"error":"TradingMoE 코드 변경은 해당 모델을 저장 후 정지하고 다시 시작하면 적용됩니다."}

    def stop(self, keep_restart: bool = False):
        assembly=getattr(self,"assembly_orchestrator",None)
        if assembly:
            assembly.start(False)
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
                self._moe_model_worker(role).stop()
            if self.profile:
                (self.profile / "feed.stop").touch()
            threading.Thread(target=self._stop_children, daemon=True).start()
        return {"ok": True, "message": "Stopping; learning state is being saved."}

    def _stop_children(self):
        for name, proc in list(self.children.items()):
            try:
                proc.wait(timeout=90 if name in ("champion","candidate") else 8)
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
            with self.lock:
                if not self.run_requested or self.stopping or not self.profile:continue
                for name,proc in list(self.children.items()):
                    code=proc.poll()
                    if code is None:continue
                    self.children.pop(name,None)
                    handle=self.log_handles.pop(name,None)
                    if handle:handle.close()
                    if name in self.model_enabled:
                        # A failed load must be visible, never an automatic reload loop.
                        self.model_enabled[name]=False
                        self._write_autonomy()
                        self._log(f"{name} stopped ({code}); saved state retained")
                        continue
                    if name!="feed" or (self.mode=="mock" and code==0):continue
                    (self.profile/"feed.stop").unlink(missing_ok=True)
                    self._launch("feed")
