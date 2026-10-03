"""Persistent recipe search; the existing Candidate lifecycle owns execution.

No torch import, checkpoint copy, account implementation or Transformer promotion.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import threading
import uuid

from .expert_registry import atomic_json, ensure_registry
from .paths import DEFAULT_MODEL_DIR, PROJECT_ROOT, PROJECT_TRASH_DIR
from .state_io import read_json as read


def now():
    return datetime.now(timezone.utc).isoformat()



def recipe_fingerprint(recipe):
    """Assembly identity, independent of IDs, scores, ordering and file paths."""
    enabled = sorted(set(recipe.get("enabled_experts", [])))
    roles = recipe.get("expert_roles", {})
    document = {"base":recipe.get("base_checkpoint_hash"), "enabled":enabled,
        "roles":{k:roles.get(k) for k in enabled},
        "universes":{k:sorted(recipe.get("symbol_applicability", {}).get(k) or []) for k in enabled},
        "versions":{k:v for k,v in sorted(recipe.get("expert_versions", {}).items()) if k in enabled},
        "market":{k:(int(v) if k=="top_k" else float(v)) for k,v in recipe.get("market_routing", {}).items()},
        "policy":{k:(int(v) if k=="top_k" else float(v)) for k,v in recipe.get("policy_routing", {}).items()},
        # Policy refresh and cache_interval are not consumed by the paired trial.
        "refresh":{k:recipe.get("refresh_seconds", {}).get(k, 7200) for k in enabled if roles.get(k)=="market"},
        "controller":recipe.get("controller_variant")}
    return hashlib.sha256(json.dumps(document,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def mutation_key(operation):
    return json.dumps(operation,sort_keys=True,separators=(",",":"))


def mutation_description(operation):
    if operation["kind"]=="toggle":return operation["expert"]+(" ON" if operation["enabled"] else " OFF")
    if operation["kind"]=="refresh":return f'{operation["expert"]} cache 유효기간={operation["value"]}초'
    group,field=operation["field"].split(".")
    return ("시장" if group=="market_routing" else "정책")+" router "+("top-k" if field=="top_k" else "온도")+"="+str(operation["value"])


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
        self.state.setdefault("expert_trials", {})
        self.history_stats = {"mutations":{}, "experts":{}, "families":{}}
        self._history_signature = None
        self._seen_fingerprints = set()
        self._evaluated_fingerprints = set()
        self._successful_operations = []
        self.experts = []
        self.worker = None
        self.scan_registry(force=True)
        if self.champion:
            self._history()
            self._prune_queue()
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
                "reason":reason,"fingerprint":recipe.get("fingerprint"),
                "generation_reason":recipe.get("generation_reason"),
                "mutation_operations":recipe.get("mutation_operations", []),
                "probed_experts":recipe.get("probed_experts", []),
                **data}, ensure_ascii=False) + "\n")

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
        for key in self.state["new_experts"] + added + changed:
            if key not in versions:continue
            trial = self.state["expert_trials"].get(key)
            if not trial or trial["version"] != versions[key]:
                self.state["expert_trials"][key] = {"version":versions[key],"status":"untested","candidate_id":None}
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

    def _history(self):
        """Read journals only when changed; terminal results count once per candidate."""
        paths = [self.directory/"history.jsonl"] + sorted((self.directory/"recipes").glob("*.json"))
        signature = [(str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in paths if p.exists()]
        if signature == self._history_signature:return
        records = {}
        seen = set()
        evaluated = set()
        journal = paths[0]
        if journal.exists():
            with journal.open(encoding="utf-8") as handle:
                for line in handle:
                    try:event=json.loads(line)
                    except ValueError:continue
                    if event.get("fingerprint"):seen.add(event["fingerprint"])
                    if event.get("event") in ("qualified","promoted","rejected"):
                        records[event.get("candidate_id")] = event
                        if event.get("fingerprint"):evaluated.add(event["fingerprint"])
        for path in paths[1:]:
            recipe = read(path)
            if not recipe:continue
            # Reconstruct legacy recipe identity using the currently pinned revision.
            recipe.setdefault("expert_versions", {k:self.state["registry_versions"].get(k) for k in recipe.get("enabled_experts", [])})
            seen.add(recipe_fingerprint(recipe))
            evaluated.add(recipe_fingerprint(recipe))
            records[recipe.get("candidate_id")] = {**records.get(recipe.get("candidate_id"), {}),**recipe}
        stats = {"mutations":{}, "experts":{}, "families":{}}
        successes = {}
        for record in records.values():
            self._record_expert_outcome(record)
            phase = record.get("scores", {}).get("paper") or record.get("scores", {}).get("replay") or {}
            delta = phase.get("delta")
            if not isinstance(delta,(int,float)) or not math.isfinite(delta):continue
            candidate, champion = phase.get("candidate", {}), phase.get("champion", {})
            samples = min(candidate.get("decisions",0),champion.get("decisions",0))
            window = str((candidate.get("first_as_of"),candidate.get("last_as_of"),phase.get("source")))
            operations = record.get("mutation_operations") or self._legacy_operations(record)
            # Mixed mutations share credit; this is an association, not causal proof.
            for operation in operations:
                key = mutation_key(operation)
                targets = [(stats["mutations"],key),(stats["families"],operation["kind"])]
                if operation.get("expert"):targets.append((stats["experts"],operation["expert"]))
                for table, label in targets:
                    item = table.setdefault(label,{"count":0,"delta_sum":0.,"improved":0,"worsened":0,"windows":{}})
                    item["count"] += 1
                    item["delta_sum"] += max(-.02,min(.02,delta))/max(1,len(operations))
                    item["improved"] += int(delta>0);item["worsened"] += int(delta<0)
                    item["windows"][window] = max(item["windows"].get(window,0),samples)
                if delta>0:successes[key] = operation
        for table in stats.values():
            for item in table.values():
                windows = item.pop("windows")
                item["samples"] = sum(windows.values())
                item["windows"] = len(windows)
                item["mean_delta"] = item["delta_sum"]/item["count"]
                item["confidence"] = min(1.,item["samples"]/256)*min(1.,item["windows"]/8)
                item["selection_weight"] = math.exp(max(-math.log(4),min(math.log(4),item["mean_delta"]*item["confidence"]/.001)))
        self.history_stats = stats
        self._successful_operations = [op for key,op in successes.items() if stats["mutations"][key]["mean_delta"]>0]
        self._seen_fingerprints = seen
        self._evaluated_fingerprints = evaluated
        self._history_signature = signature

    def _prune_queue(self):
        """Migrate stale queued recipes without touching the current worker/recipe."""
        baseline=self._candidate_base()
        occupied={recipe_fingerprint(baseline)} | self._evaluated_fingerprints
        if self.current:
            current=deepcopy(self.current)
            current.setdefault("expert_versions",{k:self.state["registry_versions"].get(k) for k in current["enabled_experts"]})
            occupied.add(recipe_fingerprint(current))
        remaining=[]
        for recipe in self.queue:
            recipe.setdefault("expert_versions",{k:self.state["registry_versions"].get(k) for k in recipe["enabled_experts"]})
            fingerprint=recipe_fingerprint(recipe)
            operations=recipe.get("mutation_operations") or self._legacy_operations(recipe)
            ineffective=bool(operations) and all(op["kind"]=="refresh" and recipe.get("expert_roles",{}).get(op["expert"])=="policy" for op in operations)
            if fingerprint not in occupied and not ineffective:
                occupied.add(fingerprint);remaining.append(recipe);continue
            recipe.update(evaluation_state="rejected",fingerprint=fingerprint)
            reason="대기열 정리: 이미 시험/예약한 조합 또는 평가 동작이 바뀌지 않는 설정"
            path=self.directory/"recipes"/(recipe["candidate_id"]+".json");path.parent.mkdir(exist_ok=True)
            atomic_json(path,recipe)
            self._record_expert_outcome(recipe)
            self._event("rejected",recipe,reason,scores=recipe.get("scores",{}),recipe_path=str(path))
            self.state["rejections"]+=1
            self._seen_fingerprints.add(fingerprint)
        if len(remaining)!=len(self.queue):
            self.queue=remaining;self._persist()

    def _mutation_weight(self, operation):
        exact=self.history_stats["mutations"].get(mutation_key(operation))
        if exact:return exact["selection_weight"]
        family=self.history_stats["families"].get(operation["kind"],{})
        expert=self.history_stats["experts"].get(operation.get("expert"),{})
        # An evaluated exact recipe is excluded. Transfer weak evidence to
        # untested settings in the same mutation family / expert instead.
        return math.sqrt(family.get("selection_weight",1.)*expert.get("selection_weight",1.))

    def _legacy_operations(self, record):
        text = record.get("mutation_description") or record.get("mutation") or ""
        words = text.split()
        if len(words)==2 and words[1] in ("ON","OFF"):
            return [{"kind":"toggle","expert":words[0],"enabled":words[1]=="ON"}]
        routing=re.search(r"router top-k=(\d+) / 온도=([\d.]+)",text)
        if routing:
            return [{"kind":"router","field":"market_routing.top_k","value":int(routing[1])},
                {"kind":"router","field":"market_routing.temperature","value":float(routing[2])}]
        refresh=re.fullmatch(r"(\S+) refresh=(\d+)초",text)
        if refresh:return [{"kind":"refresh","expert":refresh[1],"value":int(refresh[2])}]
        return []

    def _record_expert_outcome(self, recipe):
        outcome = recipe.get("evaluation_state") or recipe.get("event")
        if outcome not in ("qualified","promoted","rejected"):return
        probes=recipe.get("probed_experts") or [op["expert"] for op in self._legacy_operations(recipe)
            if op.get("enabled") and op.get("expert") in self.state["expert_trials"]]
        for key in probes:
            trial = self.state["expert_trials"].get(key)
            version = recipe.get("expert_versions", {}).get(key)
            if trial and (not version or trial["version"]==version) and trial.get("candidate_id") in (None,recipe.get("candidate_id")):
                trial.update(status="tested" if outcome=="qualified" else outcome,candidate_id=recipe.get("candidate_id"))

    def _candidate_base(self):
        candidate = deepcopy(self.champion)
        candidate.update(candidate_id="asm-"+uuid.uuid4().hex[:12],parent_id=self.champion["candidate_id"],
            created_at=now(),evaluation_state="queued",scores={},trainable_state=None,
            mutation_operations=[],probed_experts=[])
        for e in self.experts:
            key=e["id"]
            for name,value in (("expert_roles",e["role"]),("symbol_applicability",e["universe"]),
                               ("native_inputs",e["input_shapes"]),("expert_versions",e["version"])):
                candidate.setdefault(name,{})[key]=value
            candidate["refresh_seconds"].setdefault(key,7200 if e["role"]=="market" else 60)
        return candidate

    def _mutation_pool(self, recipe):
        # The current paired runner trades ETH. Unsupported stock inputs remain
        # registered but cannot produce useful ON/OFF or routing trials here.
        symbols = set(self.state.get("evaluation_context", {}).get("symbols", ["ETHUSDT"]))
        usable = {e["id"]:e for e in self.experts if e["eligible"] and
            (e["role"]=="market" or not e["universe"] or symbols.intersection(e["universe"]))}
        enabled = set(recipe["enabled_experts"])
        operations = []
        for key,e in usable.items():
            count = sum(k in enabled and v["role"]==e["role"] for k,v in usable.items())
            if key not in enabled or count>1:
                operations.append({"kind":"toggle","expert":key,"enabled":key not in enabled})
        for role,kind in (("market","router"),("policy","policy")):
            count = sum(k in enabled and e["role"]==role for k,e in usable.items())
            if count<2:continue
            name=role+"_routing"; settings=recipe[name]
            for field,values in (("top_k",[0]+[k for k in (1,2,4) if k<count]),("temperature",[.75,1.,1.25])):
                for value in values:
                    if field=="top_k":
                        old=int(settings.get(field,0))
                        if (old if 0<old<count else count)==(value if value else count):continue
                    elif settings.get("top_k")==1:continue
                    if settings.get(field)!=value:
                        operations.append({"kind":kind,"field":name+"."+field,"value":value})
        # Refresh only when the observed cache ages cross the proposed threshold.
        ages = self.state.get("evaluation_context", {}).get("market_ages", {})
        for key,values in ages.items():
            if key not in enabled or key not in usable or usable[key]["role"]!="market":continue
            old=recipe["refresh_seconds"].get(key,7200)
            for value in (60,300,1800,3600,7200):
                if value!=old and any((age<=old)!=(age<=value) for age in values):
                    operations.append({"kind":"refresh","expert":key,"value":value})
        return operations

    def _apply_operations(self, base, operations):
        recipe=deepcopy(base);enabled=set(recipe["enabled_experts"])
        for op in operations:
            if op["kind"]=="toggle":
                if op["enabled"]:enabled.add(op["expert"])
                else:enabled.discard(op["expert"])
            elif op["kind"]=="refresh":recipe["refresh_seconds"][op["expert"]]=op["value"]
            else:
                group,field=op["field"].split(".");recipe[group][field]=op["value"]
        recipe["enabled_experts"]=sorted(enabled)
        recipe["mutation_operations"]=deepcopy(operations)
        return recipe

    def _valid_combination(self, recipe):
        symbols=set(self.state.get("evaluation_context", {}).get("symbols",["ETHUSDT"]))
        usable=[e for e in self.experts if e["id"] in recipe["enabled_experts"] and e["eligible"] and
            (e["role"]=="market" or not e["universe"] or symbols.intersection(e["universe"]))]
        if not all(any(e["role"]==role for e in usable) for role in ("market","policy")):return False
        for op in recipe.get("mutation_operations",[]):
            if "field" not in op:continue
            group,field=op["field"].split(".")
            count=sum(e["role"]==group.removesuffix("_routing") for e in usable)
            selected=int(recipe[group].get("top_k",0))
            effective=selected if 0<selected<count else count
            if field=="temperature" and effective<2:return False
            if field=="top_k":
                old=int(self.champion[group].get("top_k",0))
                if effective==(old if 0<old<count else count):return False
        return True

    def generate(self):
        with self.lock:
            self._seed_champion();self._history();self._prune_queue()
            base=self._candidate_base();pool=self._mutation_pool(base)
            rng=random.Random(uuid.uuid4().hex)
            # Current/queued assemblies are reserved too, including legacy recipes.
            occupied=self._seen_fingerprints | {recipe_fingerprint(base)}
            for recipe in [self.current]+self.queue:
                if recipe:
                    recipe.setdefault("expert_versions",{k:self.state["registry_versions"].get(k) for k in recipe["enabled_experts"]})
                    occupied.add(recipe_fingerprint(recipe))
            choices=[]
            for op in pool:
                recipe=self._apply_operations(base,[op]);fingerprint=recipe_fingerprint(recipe)
                if fingerprint in occupied:continue
                trial=self.state["expert_trials"].get(op.get("expert"),{})
                new=op["kind"]=="toggle" and op["enabled"] and trial.get("status")=="untested" and not trial.get("candidate_id")
                weight=self._mutation_weight(op)
                choices.append((recipe,"new-expert-probe" if new else None,weight))
            probes=[c for c in choices if c[1]]
            explore=rng.random()<.25
            # Limit successful two-operation combinations to 15% of non-probes.
            if not probes and rng.random()<.15:
                successful=[op for op in self._successful_operations if op in pool]
                rng.shuffle(successful)
                for i,first in enumerate(successful[:12]):
                    for second in successful[i+1:12]:
                        if (first.get("field") or first.get("expert"))==(second.get("field") or second.get("expert")):continue
                        recipe=self._apply_operations(base,[first,second])
                        if recipe_fingerprint(recipe) in occupied:continue
                        # Recheck role preservation and applicability after both edits.
                        if not self._valid_combination(recipe):continue
                        choices=[(recipe,"successful-mutation-combination",1.)];break
                    if choices and choices[0][1]=="successful-mutation-combination":break
            # New combinations remain reachable even when all one-step recipes
            # have been seen; bound the random pair search instead of enumerating 2^N.
            if not probes and (not choices or explore) and len(pool)>1:
                for _ in range(64):
                    operations=rng.sample(pool,2)
                    if len({op.get("field") or op.get("expert") for op in operations})<2:continue
                    recipe=self._apply_operations(base,operations)
                    if self._valid_combination(recipe) and recipe_fingerprint(recipe) not in occupied:
                        choices.append((recipe,"exploration",1.));break
            if probes:chosen=rng.choice(probes)
            elif not choices:
                self.state["message"]="새로 시험할 유효한 조합이 없습니다 · 같은 후보를 반복하지 않습니다."
                self._persist()
                raise ValueError(self.state["message"])
            elif explore:chosen=rng.choice(choices)
            else:chosen=rng.choices(choices,weights=[c[2] for c in choices],k=1)[0]
            candidate,reason,_=chosen
            candidate["generation_reason"]=reason or ("exploration" if explore or not self.history_stats["mutations"] else "history-guided")
            candidate["selection_weights"]={mutation_key(c[0]["mutation_operations"][0]):c[2] for c in choices
                if len(c[0]["mutation_operations"])==1}
            candidate["exploration_probability"]=.25
            candidate["fingerprint"]=recipe_fingerprint(candidate)
            candidate["mutation_description"]="; ".join(mutation_description(op) for op in candidate["mutation_operations"])
            if candidate["generation_reason"]=="new-expert-probe":
                key=candidate["mutation_operations"][0]["expert"]
                candidate["probed_experts"]=[key]
                self.state["expert_trials"][key]["candidate_id"]=candidate["candidate_id"]
            self.state["generation"]+=1
            self.queue.append(candidate)
            self._event("generated",candidate,candidate["generation_reason"])
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
        self._record_expert_outcome(recipe)
        path = self.directory / "recipes" / (recipe["candidate_id"] + ".json")
        path.parent.mkdir(exist_ok=True)
        atomic_json(path, recipe)
        self._event(recipe["evaluation_state"], recipe, reason, scores=recipe.get("scores",{}),recipe_path=str(path))
        if self.worker:
            source=(self.worker.state/"assembly"/recipe["candidate_id"]).resolve()
            destination=(PROJECT_TRASH_DIR/"assembly-experiments"/recipe["candidate_id"]).resolve()
            if source.is_relative_to(PROJECT_ROOT.resolve()) and destination.is_relative_to(PROJECT_TRASH_DIR.resolve()) and source.is_dir():
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
                if self.supervisor is not None:
                    self.supervisor.model_enabled["candidate"]=False
                    self.supervisor._write_autonomy()
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
                    if result.get("evaluation_context"):self.state["evaluation_context"]=result["evaluation_context"]
                    self._record_expert_outcome(self.current)
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
                        else:
                            self.state["message"]="비교 통과 · 자동 승격 OFF · 후보 유지"
                            self._event("qualified",self.current,self.state["message"],scores=self.current.get("scores",{}))
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
                while len(self.queue)<2:
                    try:self.generate()
                    except ValueError:break
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
            self._history()
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
                "history_stats":deepcopy(self.history_stats),
                "candidate_state_bytes":Path(candidate["trainable_state"]).stat().st_size if candidate.get("trainable_state") and Path(candidate["trainable_state"]).is_file() else 0}
