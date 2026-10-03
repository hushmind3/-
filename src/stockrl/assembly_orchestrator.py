"""Persistent recipe search; the existing Candidate lifecycle owns execution.

No torch import, checkpoint copy, account implementation or Transformer promotion.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import threading
import uuid

from .expert_registry import atomic_json, ensure_registry
from .paths import DEFAULT_MODEL_DIR, PROJECT_ROOT


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, fallback=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return deepcopy(fallback if fallback is not None else {})


def expert_metadata(entry):
    """Preserve native applicability; never infer universes from probe symbols."""
    crypto = entry.get("backend") == "macrophft"
    stock = entry.get("backend") == "stock_policy" or bool(entry.get("universe"))
    return {"id": entry["id"], "name": entry.get("name", entry["id"]),
            "role": "policy" if crypto or stock else "market",
            "description": entry.get("role", ""),
            "universe": ["ETHUSDT"] if crypto else entry.get("universe"),
            "input_shapes": entry.get("probe", {}).get("input_shapes", {}),
            "version": hashlib.sha256(json.dumps({"files":entry.get("files"),
                "source":entry.get("source"),"pinned_models":entry.get("pinned_models"),
                "universe":entry.get("universe"),"probe_inputs":entry.get("probe",{}).get("input_shapes")},
                sort_keys=True).encode()).hexdigest(),
            "eligible": entry.get("verified") is True and bool(entry.get("probe",{}).get("input_shapes"))}


class AssemblyOrchestrator:
    def __init__(self, supervisor=None, *, directory=None, registry=None, checkpoint=None, background=True):
        self.supervisor = supervisor
        if supervisor is not None:supervisor.assembly_orchestrator=self
        self.directory = Path(directory or PROJECT_ROOT / "runtime/assembly")
        self.registry = Path(registry or PROJECT_ROOT / "runtime/trading_moe/registry.json")
        self.checkpoint = Path(checkpoint or DEFAULT_MODEL_DIR / "champion.pt")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.state = read(self.directory / "state.json", {"enabled":False,
            "settings":{"auto_replace":True,"auto_promote":False,"detect_experts":True},
            "experiments":0,"promotions":0,"rejections":0,"generation":0,
            "registry_versions":{},"new_experts":[],"message":"자동조립 정지"})
        self.queue = read(self.directory / "queue.json", [])
        self.current = read(self.directory / "current_recipe.json")
        self.champion = read(self.directory / "champion_recipe.json")
        self.experts = []
        self.worker = None
        self.scan_registry(force=True)
        if supervisor is not None:
            previous=supervisor._moe_model_worker("candidate")
            command=previous.read(previous.record).get("command",[])
            if any("run_assembly_trial.py" in str(part) for part in command):self._candidate_worker()
        if background:
            self.thread = threading.Thread(target=self._loop, daemon=True, name="moe-assembly")
            self.thread.start()

    def _persist(self):
        self.state["updated_at"] = now()
        for name, data in (("state",self.state),("queue",self.queue),("current_recipe",self.current),
                           ("champion_recipe",self.champion)):
            atomic_json(self.directory / (name + ".json"), data)

    def _event(self, kind, recipe, reason, **data):
        with (self.directory / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time":now(),"event":kind,"candidate_id":recipe.get("candidate_id"),
                "parent_id":recipe.get("parent_id"),"mutation":recipe.get("mutation_description"),
                "reason":reason,**data}, ensure_ascii=False) + "\n")

    def scan_registry(self, force=False):
        if not force and not self.state["settings"]["detect_experts"]:
            return
        ensure_registry(self.registry)
        document = read(self.registry)
        entries = [expert_metadata(e) for e in document.get("experts", [])]
        previous = self.state["registry_versions"]
        versions = {e["id"]:e["version"] for e in entries}
        added = [e["id"] for e in entries if e["id"] not in previous] if previous else []
        changed = [key for key in versions if key in previous and versions[key] != previous[key]]
        self.experts = entries
        self.state["registry_versions"] = versions
        if added or changed:
            self.state["new_experts"] = list(dict.fromkeys(self.state["new_experts"] + added + changed))
            self._event("registry_changed", {}, "새 expert 또는 원본 revision 감지", experts=added + changed)
        if force or added or changed:
            self._persist()

    def _base_identity(self):
        stat = self.checkpoint.stat()
        signature = [str(self.checkpoint),stat.st_size,stat.st_mtime_ns]
        if self.state.get("base_signature") == signature:
            if not self.champion:
                with self.lock:self._seed_champion()
            return
        self.state["message"] = "공용 checkpoint hash 확인 중 · PT 복사 없음"
        digest = hashlib.sha256()
        with self.checkpoint.open("rb") as handle:
            for chunk in iter(lambda:handle.read(8*1024**2), b""):
                if self.closed.is_set():return
                digest.update(chunk)
        with self.lock:
            self.state.update(base_signature=signature, base_hash=digest.hexdigest())
            self.state["message"]="공용 checkpoint 준비 완료 · 자동조립 " + ("실행" if self.state["enabled"] else "정지")
            self._seed_champion()
            self._persist()

    def _seed_champion(self):
        if self.champion:
            return
        if not self.state.get("base_hash"):
            raise ValueError("공용 checkpoint hash 확인 중입니다. 잠시 후 다시 요청하세요.")
        eligible = [e for e in self.experts if e["eligible"]]
        if not any(e["role"] == "market" for e in eligible):
            raise ValueError("사용 가능한 시장 expert가 없습니다.")
        self.champion = {"candidate_id":"champion-base","parent_id":None,
            "base_checkpoint":str(self.checkpoint),"base_checkpoint_hash":self.state["base_hash"],
            "enabled_experts":[e["id"] for e in eligible],
            "expert_roles":{e["id"]:e["role"] for e in eligible},
            "symbol_applicability":{e["id"]:e["universe"] for e in eligible},
            "native_inputs":{e["id"]:e["input_shapes"] for e in eligible},
            "refresh_seconds":{e["id"]:7200 if e["role"] == "market" else 60 for e in eligible},
            "cache_interval_seconds":60,"market_routing":{"top_k":0,"temperature":1.0},
            "policy_routing":{"top_k":0,"temperature":1.0},
            "controller_variant":"vertical_native_prior_v1","mutation_description":"현재 Champion 기본 조립",
            "created_at":now(),"evaluation_state":"champion","trainable_state":None}
        self._persist()

    def generate(self):
        with self.lock:
            self._seed_champion()
            candidate = deepcopy(self.champion)
            candidate.update(candidate_id="asm-" + uuid.uuid4().hex[:12],parent_id=self.champion["candidate_id"],
                created_at=now(),evaluation_state="queued",scores={},trainable_state=None)
            available = {e["id"]:e for e in self.experts if e["eligible"]}
            enabled = [key for key in candidate["enabled_experts"] if key in available]
            self.state["generation"] += 1
            rng = random.Random(candidate["candidate_id"])
            newcomers = [key for key in self.state["new_experts"] if key in available and key not in enabled]
            kind = "enable" if newcomers else ["toggle","router","refresh","policy"][self.state["generation"] % 4]
            if kind in ("enable","toggle","policy"):
                options = newcomers or [key for key,e in available.items() if kind != "policy" or e["role"] == "policy"]
                # Keep at least one usable market expert and one existing policy.
                options = [key for key in options if key not in enabled or sum(available[k]["role"] == available[key]["role"] for k in enabled) > 1]
                if not options:kind = "router"
                else:
                    key = rng.choice(options)
                    if key in enabled:enabled.remove(key);description = key + " OFF"
                    else:enabled.append(key);description = key + " ON"
            if kind == "router":
                settings = candidate["market_routing"]
                settings["top_k"] = rng.choice([1,2,4])
                settings["temperature"] = rng.choice([.75,1.0,1.25])
                description = "시장 router top-k=" + str(settings["top_k"]) + " / 온도=" + str(settings["temperature"])
            if kind == "refresh":
                key = rng.choice(enabled)
                candidate["refresh_seconds"][key] = rng.choice([60,300,1800,3600])
                description = key + " refresh=" + str(candidate["refresh_seconds"][key]) + "초"
            candidate["enabled_experts"] = enabled
            for key,e in available.items():
                candidate["expert_roles"][key]=e["role"]
                candidate["symbol_applicability"][key]=e["universe"]
                candidate["native_inputs"][key]=e["input_shapes"]
                candidate["refresh_seconds"].setdefault(key,7200 if e["role"] == "market" else 60)
            candidate["mutation_description"] = description
            self.queue.append(candidate)
            self._event("generated", candidate, "Champion에서 소규모 mutation 생성")
            if not self.current:self._install_next()
            self._persist()
            return candidate

    def _install_next(self):
        if not self.queue:return
        self.current = self.queue.pop(0)
        self.current["evaluation_state"] = "ready"
        self.state["message"] = "새 Candidate 장착 · 시험 대기"
        self._event("installed", self.current, "Candidate 슬롯 recipe 교체 · 전체 PT 복사 없음")

    def next(self, reason="사용자가 현재 후보 탈락 요청"):
        with self.lock:
            if self.worker and self.worker.process():
                self.worker.stop()
                self.state["pending_next_reason"] = reason
                self._persist()
                return {"ok":True,"message":"시험 저장·정지 후 다음 후보로 교체합니다."}
            if self.current:
                self.current["evaluation_state"] = "rejected"
                self.state["rejections"] += 1
                self._archive(reason)
                self.current = {}
            if not self.queue and self.state["settings"]["auto_replace"]:self.generate()
            self._install_next()
            self._persist()
            return {"ok":True,"message":self.state["message"]}

    def _archive(self, reason):
        recipe = deepcopy(self.current)
        path = self.directory / "recipes" / (recipe["candidate_id"] + ".json")
        path.parent.mkdir(exist_ok=True)
        atomic_json(path, recipe)
        self._event(recipe["evaluation_state"], recipe, reason, scores=recipe.get("scores",{}),recipe_path=str(path))
        if self.worker:
            source=(self.worker.state/"assembly"/recipe["candidate_id"]).resolve()
            destination=(PROJECT_ROOT/"휴지통/assembly-experiments"/recipe["candidate_id"]).resolve()
            if source.is_relative_to(PROJECT_ROOT.resolve()) and destination.is_relative_to(PROJECT_ROOT.resolve()) and source.is_dir():
                destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists():shutil.move(str(source),str(destination))

    def _candidate_worker(self):
        if self.supervisor is None:raise ValueError("Candidate lifecycle가 연결되지 않았습니다.")
        worker = self.supervisor._moe_model_worker("candidate")
        if worker.process():
            command=worker.read(worker.record).get("command",[])
            if worker is not self.worker and not any("run_assembly_trial.py" in str(part) for part in command):
                raise ValueError("Candidate가 이미 실행 중입니다. 기존 모델을 먼저 정지하세요.")
            worker.runner_script="run_assembly_trial.py"
            worker.extra_args=["--assembly-root",str(self.directory)]
            worker.checkpoint=self.checkpoint
            self.worker=worker
            return worker
        if self.supervisor.model_enabled.get("candidate") and self.supervisor.model_families.get("candidate") != "trading_moe":
            raise ValueError("기존 Candidate가 실행 중입니다. 자동실험과 동시에 사용하지 않습니다.")
        worker.checkpoint = self.checkpoint
        worker.runner_script = "run_assembly_trial.py"
        worker.extra_args = ["--assembly-root",str(self.directory)]
        self.supervisor.model_families["candidate"] = "trading_moe"
        self.worker = worker
        return worker

    def owns_candidate_slot(self):
        return self.worker is not None and self.worker.runner_script=="run_assembly_trial.py"

    def trial(self, enabled):
        with self.lock:
            if not enabled:
                self.state["trial_paused"] = True
                if self.worker:self.worker.stop()
                self._persist()
                return {"ok":True,"message":"현재 시험 정지 요청 · 성적과 작은 state 저장"}
            if not self.current:self.generate()
            if self.current.get("base_checkpoint_hash") != self.state.get("base_hash"):
                raise ValueError("공용 PT가 변경됐습니다. 이전 hash의 recipe를 새 PT 성적으로 평가하지 않습니다.")
            worker = self._candidate_worker()
            if worker.process():return {"ok":True,"already_running":True}
            self.state["trial_paused"] = False
            self.current["evaluation_state"] = "replay"
            self.current["reason"] = None
            self.state["message"] = "현재 Candidate 시험 시작 · 같은 조건으로 Champion과 비교"
            (self.directory/"results"/(self.current["candidate_id"]+".json")).unlink(missing_ok=True)
            result = worker.start()
            if not result.get("ok"):
                self.current["evaluation_state"] = "blocked"
                self.current["reason"] = result.get("error")
            else:
                self.state["experiments"] += 1
                self.state["trial_candidate_id"] = self.current["candidate_id"]
                self.supervisor.model_enabled["candidate"] = True
                self.supervisor._write_autonomy()
            self._persist()
            return result

    def settings(self, payload):
        with self.lock:
            for key,value in payload.items():
                if key not in self.state["settings"] or not isinstance(value,bool):raise ValueError("설정은 허용된 ON/OFF 값만 받습니다.")
            self.state["settings"].update(payload)
            self._persist()
            return {"ok":True}

    def start(self, enabled):
        with self.lock:
            self.state["enabled"] = enabled
            self.state["message"] = "자동조립 실행 · 후보 생성과 순차 시험" if enabled else "자동조립 정지 · 진행 중 시험은 별도 제어"
            if enabled:self.state["trial_paused"] = False
            self._persist()
            return {"ok":True}

    def tick(self):
        with self.lock:
            self.scan_registry()
            if self.current:
                result = read(self.directory / "results" / (self.current["candidate_id"] + ".json"))
                if result and (not self.worker or not self.worker.process()) and (self.current["evaluation_state"] in ("replay","paper","blocked") or self.current["evaluation_state"]=="qualified" and self.state["settings"]["auto_promote"]):
                    self.current.update(scores=result.get("scores",{}),reason=result.get("reason"),evaluation_state=result["state"],
                        trainable_state=result.get("trainable_state"))
                    self.state["trial_candidate_id"] = None
                    if self.supervisor:
                        self.supervisor.model_enabled["candidate"]=False
                        self.supervisor._write_autonomy()
                    if result["state"] == "qualified":
                        if self.state["settings"]["auto_promote"]:
                            self.current["evaluation_state"]="promoted"
                            self.champion=deepcopy(self.current)
                            self.state["promotions"]+=1
                            self._archive("같은 시점·비용·초기 자금의 paper 비교 통과 · Champion recipe 승격")
                            self.current={}
                        else:self.state["message"]="비교 통과 · 자동 승격 OFF · 후보 유지"
                    elif result["state"] == "rejected":
                        self.state["rejections"]+=1
                        self._archive(result.get("reason","비교 탈락"));self.current={}
                    if not self.current and self.state["settings"]["auto_replace"]:
                        if not self.queue:self.generate()
                        self._install_next()
                    self._persist()
                elif self.current["evaluation_state"] in ("replay","paper") and self.worker and not self.worker.process():
                    self.current.update(evaluation_state="blocked",reason="시험 worker가 결과 저장 없이 종료됐습니다. 시험 시작으로 재개할 수 있습니다.")
                    self._persist()
                elif self.worker and self.worker.process():
                    stage=self.worker.status().get("evaluation_stage")
                    if stage in ("replay","paper") and self.current["evaluation_state"]!=stage:
                        self.current["evaluation_state"]=stage;self._persist()
            if self.state.get("pending_next_reason") and (not self.worker or not self.worker.process()):
                reason=self.state.pop("pending_next_reason");self.next(reason)
            if self.state["enabled"] and self.state.get("base_hash"):
                if not self.current and self.state["experiments"] and not self.state["settings"]["auto_replace"]:return
                while len(self.queue)<2:self.generate()
                if not self.current and self.state["settings"]["auto_replace"]:self._install_next();self._persist()
                if self.current.get("evaluation_state")=="ready" and not self.state.get("trial_paused"):
                    if not self.worker or not self.worker.process():self.trial(True)

    def _loop(self):
        while not self.closed.is_set():
            try:
                self._base_identity()
                self.tick()
            except Exception as exc:
                with self.lock:
                    self.state["message"] = str(exc)
                    self._persist()
            self.closed.wait(2)

    def close(self):
        self.closed.set()

    def status(self):
        with self.lock:
            lines=[]
            path=self.directory/"history.jsonl"
            if path.exists():
                with path.open("rb") as handle:
                    handle.seek(max(0,path.stat().st_size-64000))
                    for line in handle.read().splitlines()[-50:]:
                        try:lines.append(json.loads(line))
                        except ValueError:pass
            candidate=deepcopy(self.current)
            worker=self.worker.status() if self.worker else {}
            return {"ok":True,**deepcopy(self.state),"champion":deepcopy(self.champion),
                "candidate":candidate,"queue":deepcopy(self.queue),"history":lines[::-1],
                "experts":deepcopy(self.experts),"worker":worker,"checkpoint_copies":0,
                "candidate_state_bytes":Path(candidate["trainable_state"]).stat().st_size if candidate.get("trainable_state") and Path(candidate["trainable_state"]).is_file() else 0}
