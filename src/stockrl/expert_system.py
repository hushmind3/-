"""Standalone capability routing and typed evidence fusion, outside live trading.

The experts are pretrained. The common trading head is NOT pretrained: actual
frozen diagnostic projections/attention/heads run, with non-executable outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import math
import os
import subprocess
import sys
import tempfile
import time
import threading
import uuid
import errno
from contextlib import contextmanager

from .gpu_scheduler import FairGpuScheduler
from .expert_registry import atomic_json, utc_now
from .paths import GPU_OWNER_LOCK, expert_weight_path


@contextmanager
def registry_owner(path, *, wait=False, on_wait=None):
    """Serialize GPU work; singleton worker locks still reject duplicate owners."""
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        waiting_started = None
        while True:
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if not wait or exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise RuntimeError("another TradingMoE wrapper owns the GPU execution lock") from exc
                if waiting_started is None:
                    waiting_started = time.monotonic()
                    if on_wait:on_wait(True, 0.0)
                time.sleep(.05)
        try:
            if waiting_started is not None and on_wait:
                on_wait(False, time.monotonic()-waiting_started)
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    modality: str
    weight_bytes: int
    priority: int


EXPERTS = (
    ExpertSpec("timesfm", "daily_excess_return", 79_237_216, 0),
    ExpertSpec("kronos", "OHLCV", 425_074_536, 1),
    ExpertSpec("toto", "multivariate", 1_250_738_432, 2),
    ExpertSpec("macrophft", "native_ETH_policy", 182_028, 3),
    ExpertSpec("marketgpt", "ITCH", 377_170_944, 4),
    ExpertSpec("exaone", "numeric_series", 809_278_080, 5),
    ExpertSpec("chronos", "daily_excess_return", 184_616_960, 6),
    ExpertSpec("timemoe", "numeric_series", 906_393_600, 7),
    ExpertSpec("fincast", "numeric_series", 3_965_747_840, 8),
)


def select_experts(inputs, top_k=2, weight_budget_bytes=4 * 1024**3, specs=EXPERTS, diversify_modalities=True):
    """Deterministic capability/cost admission, not a learned or calibrated gate.

    Weight budget is an admission ceiling per sequential expert, not proof that
    an inference fits. Measured working memory + device headroom is also needed.
    """
    if top_k < 1 or weight_budget_bytes < 1:
        raise ValueError("top_k and memory budget must be positive")
    chosen, used_modalities = [], set()
    for spec in sorted(specs, key=lambda item: item.priority):
        if (spec.name not in inputs or spec.weight_bytes > weight_budget_bytes
                or diversify_modalities and spec.modality in used_modalities):
            continue
        chosen.append(spec)
        used_modalities.add(spec.modality)
        if len(chosen) == top_k:
            break
    return chosen


def validate_snapshot(snapshot):
    symbols = snapshot["symbols"]
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("snapshot must have unique symbols")
    if not snapshot.get("as_of"):
        raise ValueError("snapshot as_of is required")
    currencies = snapshot["currencies"]
    current = snapshot["current_weights"]
    if set(currencies) != set(symbols) or set(current) != set(symbols):
        raise ValueError("currency and position maps must match symbol identities")
    for symbol in symbols:
        if currencies[symbol] not in ("KRW", "USD"):
            raise ValueError("existing project uses independent KRW and USD ledgers")
        if not math.isfinite(current[symbol]) or current[symbol] < 0:
            raise ValueError("current weights must be finite and nonnegative")
    for currency in set(currencies.values()):
        total = sum(current[s] for s in symbols if currencies[s] == currency)
        if total > 1 + 1e-9:
            raise ValueError("weights exceed their currency account equity")
    for name, data in snapshot["expert_inputs"].items():
        if data.get("as_of") != snapshot["as_of"]:
            raise ValueError(f"{name} input belongs to another as_of snapshot")
        observed = data.get("symbols", [])
        if not observed or len(set(observed)) != len(observed) or not set(observed).issubset(symbols):
            raise ValueError(f"{name} input symbols must be a unique subset of the snapshot")


def evidence_tokens(packets):
    """Keep native outputs and units in separate tokens; do not average forecasts.

    A later trained per-modality projection consumes these tokens. Forecast
    quantiles and policy Q values retain different schemas and horizons.
    """
    return [{"expert":p["expert"], "symbols":p["symbols"], "units":p["units"],
        "layout":p["layout"], "horizon":p["horizon"], "sampling_seconds":p["sampling_seconds"],
        "as_of":p["as_of"], "native_output":p["native_output"], "frozen":p["frozen"]}
        for p in packets]


def pending_head_output(snapshot):
    """A head readiness sentinel; HOLD is not a prediction or learned strategy."""
    symbols = snapshot["symbols"]
    weights = dict(snapshot["current_weights"])
    currencies = snapshot["currencies"]
    cash = {c: 1 - sum(weights[s] for s in symbols if currencies[s] == c)
            for c in sorted(set(currencies.values()))}
    return {"policy_status":"fusion_head_not_trained", "executable":False,
        "actions":{s:"HOLD" for s in symbols}, "target_weights":weights,
        "cash_weights_by_currency":cash, "selected_symbols":[s for s in symbols if weights[s] > 0],
        "position_replacements":[], "reason":"native inference verified; joint trading head needs subsequent training"}


class TradingMoE:
    """One GPU owner, sequential native experts, CPU evidence, explicit readiness.

    This research class launches disposable workers for dependency isolation.
    A future live implementation must use in-process/persistent backends owned
    by the SAME scheduler as the live agent. Per-process schedulers do not
    arbitrate against unrelated live processes. No integration is done here.
    """
    def __init__(self, artifact_root, device="cpu", scheduler=None, top_k=2,
                 weight_budget_bytes=4*1024**3, registry_path=None, adapters_enabled=True,
                 diversify_modalities=True):
        self.root = Path(artifact_root).resolve()
        self.device = device
        self.scheduler = scheduler or FairGpuScheduler()
        self.top_k = top_k
        self.weight_budget_bytes = weight_budget_bytes
        self.adapters_enabled = adapters_enabled
        self.diversify_modalities = diversify_modalities
        self.fusion_head = None
        self.fusion_signature = None
        self.fusion_checkpoint = None
        self.registry_path = Path(registry_path) if registry_path else Path(__file__).resolve().parents[2] / "runtime/trading_moe/registry.json"
        self.lock = threading.Lock()
        with registry_owner(GPU_OWNER_LOCK):
            self._register_catalog()

    def _register_catalog(self):
        catalog = json.loads((self.root / "expert_catalog.json").read_text(encoding="utf-8"))
        previous = {}
        if self.registry_path.is_file():
            from .expert_registry import read_registry
            status = read_registry(self.registry_path)
            if any(e["active"] for e in status["experts"]):
                raise RuntimeError("another TradingMoE worker already owns this registry")
            saved = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if saved.get("artifact_root") == str(self.root):
                previous = {e["id"]:e for e in saved["experts"]}
        capabilities = {s.name:s for s in EXPERTS}
        self.registry = catalog
        self.registry["artifact_root"] = str(self.root)
        self.specs = []
        for entry in catalog["experts"]:
            if not entry.get("verified") or not entry.get("frozen"):
                raise ValueError(f"unverified/unfrozen expert cannot register: {entry['id']}")
            backend = capabilities[entry["backend"]]
            self.specs.append(ExpertSpec(entry["id"], backend.modality, entry["weight_bytes"], backend.priority))
            entry.update(loaded=False, active=False, router_selected=False,
                last_inference_seconds=None, last_used_at=None, error=None, worker=None)
            entry["raw_output_path"] = str(self.root / "verification" / (entry["id"] + ".json"))
            entry["raw_output_origin"] = "independent_verification"
            old = previous.get(entry["id"], {})
            if old.get("files") == entry["files"]:
                for key in ("last_inference_seconds", "last_used_at", "router_selected", "raw_output_path",
                            "raw_output_origin", "last_peak_vram_bytes", "last_input_shapes", "last_output_shape",
                            "last_timings", "raw_output_sha256"):
                    if key in old:
                        entry[key] = old[key]
        if previous and saved.get("pipeline"):
            self.registry["pipeline"] = saved["pipeline"]
            self.fusion_checkpoint = saved["pipeline"].get("fusion_checkpoint")
        # The legacy diagnostic wrapper does not own the registered PT policies.
        # Refreshing its original catalog must not erase the runtime's additions.
        for entry in previous.values():
            if entry.get("backend")=="stock_policy":
                self.registry["experts"].append(entry)
        self._publish()

    def _publish(self):
        self.registry["updated_at"] = utc_now()
        atomic_json(self.registry_path, self.registry)

    def _verify_originals(self, entry):
        for artifact in entry["files"]:
            if artifact.get("archive"):
                continue
            path = expert_weight_path(self.root / artifact["path"])
            if path.stat().st_size != artifact["bytes"]:
                raise ValueError(f"original file size changed: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8*1024**2), b""):
                    digest.update(chunk)
            if digest.hexdigest() != artifact["sha256"]:
                raise ValueError(f"original checkpoint hash changed: {path}")

    def infer(self, snapshot):
        with self.lock, registry_owner(GPU_OWNER_LOCK):
            try:
                return self._infer(snapshot)
            except BaseException as exc:
                self.registry["recent_error"] = str(exc)
                self.registry.setdefault("pipeline", {}).update(stage="error", error=str(exc))
                self._publish()
                raise

    def _infer(self, snapshot):
        inference_started = time.perf_counter()
        run_id = uuid.uuid4().hex
        self.registry["pipeline"] = {"run_id":run_id, "stage":"input_adapter", "started_at":utc_now(),
            "as_of":snapshot.get("as_of"), "input_authenticity":snapshot.get("input_authenticity"),
            "adapter_status":"enabled" if self.adapters_enabled else "disabled_raw_review_phase",
            "fusion_head_status":"untrained_diagnostic" if self.adapters_enabled else "not_trained",
            "training_performed":False}
        validate_snapshot(snapshot)
        known = {s.name for s in self.specs}
        if set(snapshot["expert_inputs"]) - known:
            raise ValueError("input references an unregistered expert")
        input_adapters = {name:{"symbols":data["symbols"], "as_of":data["as_of"],
            "units":data.get("units"), "input_authenticity":data.get("input_authenticity"),
            "provided_fields":sorted(data)} for name,data in snapshot["expert_inputs"].items()}
        self.registry["pipeline"]["stage"] = "router"
        selected = select_experts(snapshot["expert_inputs"], self.top_k, self.weight_budget_bytes, self.specs,
                                  self.diversify_modalities)
        if not selected:
            raise ValueError("no expert has an available compatible modality within the budget")
        packets = []
        by_id = {e["id"]:e for e in self.registry["experts"]}
        for entry in by_id.values():
            entry["router_selected"] = entry["id"] in {s.name for s in selected}
        self.registry["recent_error"] = None
        self.registry["pipeline"].update(stage="experts", selected_experts=[s.name for s in selected], completed_experts=0)
        self._publish()
        for spec in selected:
            entry = by_id[spec.name]
            round_trip_started = time.perf_counter()
            self._verify_originals(entry)
            with tempfile.TemporaryDirectory(prefix="stockrl-expert-") as temp:
                folder = Path(temp)
                inp, out = folder / "input.json", folder / "output.json"
                data = dict(snapshot["expert_inputs"][spec.name])
                if entry["variant"]:
                    data["variant"] = entry["variant"]
                inp.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
                venv = "venv-toto" if entry["backend"] == "toto" else "venv"
                python = self.root / venv / "Scripts/python.exe"
                if not python.is_file():
                    raise FileNotFoundError(f"isolated expert Python missing: {python}")
                env = os.environ.copy()
                env.update(PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                           PYTHONUTF8="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
                with self.scheduler.work("research_expert_" + spec.name):
                    import psutil
                    status_path = folder / "worker.json"
                    process_started = time.perf_counter()
                    worker = subprocess.Popen([str(python), "-m", "stockrl.expert_backends", entry["backend"],
                        "--root", str(self.root), "--input", str(inp), "--output", str(out),
                        "--device", self.device, "--status", str(status_path), "--expert-id", spec.name],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env)
                    entry["worker"] = {"pid":worker.pid, "created_at":psutil.Process(worker.pid).create_time(),
                        "status_path":str(status_path)}
                    self._publish()
                    try:
                        stdout, stderr = worker.communicate(timeout=300)
                        if worker.returncode:
                            raise RuntimeError(f"{spec.name}: {stderr[-2000:]}")
                    except BaseException as exc:
                        try:
                            for child in psutil.Process(worker.pid).children(recursive=True):
                                child.kill()
                        except psutil.Error:
                            pass
                        worker.kill()
                        worker.communicate()
                        entry["error"] = self.registry["recent_error"] = str(exc)
                        raise
                    finally:
                        entry["worker"] = None
                        self._publish()
                    process_seconds = time.perf_counter() - process_started
                    packet = json.loads(out.read_text(encoding="utf-8"))
                    if packet["parameters"] != entry["parameters"] or packet["parameter_bytes"] != entry["weight_bytes"]:
                        raise ValueError(f"registered parameter schema changed: {spec.name}")
                    if packet["as_of"] != snapshot["as_of"] or packet["symbols"] != data["symbols"]:
                        raise ValueError("native packet provenance differs from supplied expert input")
                    raw_path = self.root / "inference/runs" / run_id / (spec.name + ".json")
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_bytes = out.read_bytes()
                    raw_path.write_bytes(raw_bytes)  # Verbatim worker output, before adaptation.
                    entry["raw_output_path"] = str(raw_path)
                    entry["raw_output_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
                    entry["raw_output_origin"] = "runtime_inference"
                timings = {key:packet[key] for key in ("cold_load_seconds", "gpu_transfer_seconds", "forward_seconds")}
                timings.update(process_seconds=process_seconds, round_trip_seconds=time.perf_counter()-round_trip_started)
                packets.append({**packet, "expert":spec.name, "timings":timings})
                entry["last_timings"] = timings
                entry["last_inference_seconds"] = packet["forward_seconds"]
                entry["last_used_at"] = utc_now()
                entry["last_peak_vram_bytes"] = packet["peak_allocated_bytes"]
                entry["last_input_shapes"] = packet["input_shapes"]
                entry["last_output_shape"] = packet["output_shape"]
                entry["error"] = None
                self.registry["pipeline"]["completed_experts"] = len(packets)
                self._publish()
        self.registry["pipeline"]["stage"] = "output_adapter"
        self._publish()
        adaptation_started = time.perf_counter()
        adapted = adapter_features(packets, snapshot["symbols"]) if self.adapters_enabled else []
        adaptation_seconds = time.perf_counter()-adaptation_started
        self.registry["pipeline"]["stage"] = "fusion"
        self._publish()
        fusion_started = time.perf_counter()
        fusion, trading = self._fuse(adapted, snapshot) if adapted else ({"status":"disabled"}, pending_head_output(snapshot))
        fusion_seconds = time.perf_counter()-fusion_started
        result = {"schema":"frozen_heterogeneous_experts_v2", "run_id":run_id, "as_of":snapshot["as_of"],
                "selected_experts":[s.name for s in selected], "router_status":"deterministic_untrained",
                "input_adapters":input_adapters, "evidence_tokens":evidence_tokens(packets), "trading_output":trading,
                "shared_representation":adapted, "fusion_output":fusion,
                "adapter_status":"connected" if self.adapters_enabled else "disabled_raw_review_phase",
                "pipeline_timings":{"adapter_seconds":adaptation_seconds, "fusion_seconds":fusion_seconds,
                    "total_seconds":time.perf_counter()-inference_started},
                "profiles":[{k:v for k,v in p.items() if k != "native_output"} for p in packets],
                "training_performed":False, "live_integration":False}
        result_path = self.root / "inference/runs" / run_id / "TradingMoE.json"
        atomic_json(result_path, result)
        self.registry["pipeline"].update(stage="complete", completed_at=utc_now(), result_path=str(result_path),
            timings=result["pipeline_timings"], fusion_shapes=fusion.get("shapes"),
            executable=trading["executable"], fusion_checkpoint=self.fusion_checkpoint)
        self._publish()
        return result

    def _fuse(self, adapted, snapshot):
        """Execute real projections/attention/heads, with frozen diagnostic weights.

        No pretrained joint head exists. Untrained outputs are kept observable,
        but are marked non-executable and never connected to paper/live accounts.
        """
        import numpy as np
        import torch
        sizes = {a["expert"]:a["shape"][1]+3 for a in adapted}
        signature = hashlib.sha256(json.dumps(sizes, sort_keys=True).encode()).hexdigest()[:16]
        checkpoint = expert_weight_path(self.root / "fusion" / (signature + ".untrained.pt"))
        if self.fusion_signature != signature:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(2026)
                self.fusion_head = build_fusion_head(dict(sorted(sizes.items())))
            if checkpoint.is_file():
                self.fusion_head.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            else:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.fusion_head.state_dict(), checkpoint)
            self.fusion_head.requires_grad_(False).eval()
            self.fusion_signature = signature
            self.fusion_checkpoint = {"path":str(checkpoint.relative_to(self.root)), "sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "schema":sizes, "trained":False, "frozen":True, "device":"cpu"}
        values, coverage = {}, []
        for a in adapted:
            x = np.asarray(a["features"], dtype=np.float32)
            scale = np.maximum(np.sqrt(np.mean(x.astype(np.float64)**2, axis=-1, keepdims=True)), 1e-6)
            meta = np.column_stack([np.log1p(scale[:,0]),
                np.full(len(x), np.log1p(a["horizon"])), np.full(len(x), np.log1p(a["sampling_seconds"] or 0))])
            features = np.concatenate([x/scale, meta], axis=-1).astype(np.float32)
            available = np.asarray(a["coverage_mask"], dtype=bool)
            features[~available] = 0
            values[a["expert"]] = torch.tensor(features[None])
            coverage.append(available)
        symbols = snapshot["symbols"]
        market = torch.zeros((1,len(symbols),16))
        for n,symbol in enumerate(symbols):
            currency = snapshot["currencies"][symbol]
            market[0,n,:4] = torch.tensor([snapshot["current_weights"][symbol],
                1-sum(snapshot["current_weights"][s] for s in symbols if snapshot["currencies"][s] == currency),
                float(currency == "USD"), float(currency == "KRW")])
        mask = torch.tensor(~np.asarray(coverage).T)[None]
        before = [p._version for p in self.fusion_head.parameters()]
        with torch.inference_mode():
            output = self.fusion_head(values, market, key_padding_mask=mask, allow_untrained=True)
        if before != [p._version for p in self.fusion_head.parameters()] or any(p.requires_grad for p in self.fusion_head.parameters()):
            raise RuntimeError("diagnostic fusion weights mutated")
        coverage_any = (~mask[0]).any(-1).tolist()
        fusion = {"status":"untrained_diagnostic", "trained":False, "device":"cpu", "frozen":True,
            "shapes":{k:list(v.shape) for k,v in output.items()}, "coverage_mask":coverage_any,
            "native_head_output":{k:v.tolist() for k,v in output.items()}}
        return fusion, decode_trading_output(output, snapshot, coverage_any)

    def save_manifest(self, path):
        files = [{k:v for k,v in entry.items() if k not in
            ("worker", "active", "loaded", "router_selected", "error", "last_inference_seconds", "last_used_at",
             "raw_output_path", "raw_output_origin", "last_peak_vram_bytes", "last_input_shapes", "last_output_shape",
             "last_timings", "raw_output_sha256")}
            for entry in self.registry["experts"]]
        payload = {"schema":"frozen_heterogeneous_experts_v1", "expert_artifacts":files,
            "unavailable":self.registry["unavailable"],
            "router":{"top_k":self.top_k, "weight_budget_bytes":self.weight_budget_bytes,
                      "status":"deterministic_untrained", "diversify_modalities":self.diversify_modalities},
            "fusion_head_status":"untrained_diagnostic" if self.adapters_enabled else "not_trained",
            "fusion_checkpoint":self.fusion_checkpoint,
            "experts_frozen":True, "weights_merged":False, "originals_embedded":False,
            "packaging":"manifest references immutable originals; self-contained packaging is a separate step"}
        atomic_json(path, payload)
        return payload

    @classmethod
    def from_manifest(cls, path, artifact_root, **kwargs):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["experts_frozen"] is not True or payload["weights_merged"] is not False:
            raise ValueError("manifest is not an independent frozen expert model")
        expected = [(e["id"], e["files"]) for e in payload["expert_artifacts"]]
        catalog = json.loads((Path(artifact_root) / "expert_catalog.json").read_text(encoding="utf-8"))
        actual = [(e["id"], e["files"]) for e in catalog["experts"]]
        if expected != actual:
            raise ValueError("manifest artifacts differ from independently verified catalog")
        for name in ("top_k", "weight_budget_bytes", "diversify_modalities"):
            if name in payload["router"]:
                kwargs.setdefault(name, payload["router"][name])
        kwargs.setdefault("adapters_enabled", payload.get("fusion_head_status") == "untrained_diagnostic")
        return cls(artifact_root, **kwargs)


def adapter_features(packets, symbols=None):
    """Native evidence -> per-symbol features, retaining modality/units/masks.

    This is structural adaptation, not trained projection or price averaging.
    Sampling/quantile axes are flattened only within their own expert token.
    """
    import numpy as np
    shared = []
    for packet in packets:
        values = np.asarray(packet["native_output"], dtype=np.float32)
        if packet["layout"] == "nine_quantiles,batch,variate,horizon":
            values = values[:, 0].transpose(1, 0, 2)
        count = len(packet["symbols"])
        if values.shape[0] != count:
            raise ValueError(f"expert output cannot align with symbol axis: {packet['expert']}")
        features = values.reshape(count, -1)
        target_symbols = symbols or packet["symbols"]
        aligned = np.zeros((len(target_symbols), features.shape[-1]), dtype=np.float32)
        coverage = np.zeros(len(target_symbols), dtype=bool)
        for n,symbol in enumerate(packet["symbols"]):
            target = target_symbols.index(symbol)
            aligned[target] = features[n]
            coverage[target] = True
        if not np.isfinite(aligned).all():
            raise ValueError("nonfinite native evidence cannot enter fusion")
        shared.append({"expert":packet["expert"], "symbols":target_symbols, "source_symbols":packet["symbols"],
            "units":packet["units"], "horizon":packet["horizon"], "as_of":packet["as_of"],
            "sampling_seconds":packet.get("sampling_seconds"),
            "features":aligned.tolist(), "shape":list(aligned.shape), "coverage_mask":coverage.tolist(),
            "mask":np.broadcast_to(coverage[:,None], aligned.shape).tolist()})
    return shared


def decode_trading_output(output, snapshot, coverage):
    """Diagnostic head output -> the project's independent currency contract."""
    import torch
    symbols, currencies = snapshot["symbols"], snapshot["currencies"]
    weights = dict(snapshot["current_weights"])
    scores = output["allocation_scores"][0]
    cash_score = output["cash_scores"][0,0]
    cash = {}
    for currency in sorted(set(currencies.values())):
        available = [n for n,s in enumerate(symbols) if currencies[s] == currency and coverage[n]]
        held = sum(weights[s] for n,s in enumerate(symbols) if currencies[s] == currency and not coverage[n])
        allocation = torch.softmax(torch.cat([scores[available],cash_score[None]]), dim=-1) * (1-held)
        for n,value in zip(available,allocation[:-1]):
            weights[symbols[n]] = float(value)
        cash[currency] = float(allocation[-1])
    actions = {s:("BUY" if weights[s]-snapshot["current_weights"][s] > 1e-6 else
                  "SELL" if weights[s]-snapshot["current_weights"][s] < -1e-6 else "HOLD") for s in symbols}
    entering = [s for s in symbols if snapshot["current_weights"][s] == 0 and weights[s] > 1e-6]
    leaving = [s for s in symbols if snapshot["current_weights"][s] > 0 and weights[s] <= 1e-6]
    return {"policy_status":"untrained_fusion_diagnostic", "executable":False,
        "actions":actions, "target_weights":weights, "cash_weights_by_currency":cash,
        "selected_symbols":[s for s in symbols if weights[s] > 1e-6],
        "position_replacements":[{"from":a,"to":b} for a,b in zip(leaving,entering)],
        "action_probabilities":torch.softmax(output["policy_logits"][0],-1).tolist(),
        "action_order":["SELL","HOLD","BUY"], "values":output["value"][0].tolist(),
        "reason":"End-to-end diagnostic inference; joint head has no trained checkpoint"}


def build_fusion_head(expert_feature_sizes, market_features=16, width=64):
    """Real projections/cross-attention/heads; diagnostic execution is explicit.

    Do not run this randomly initialized module as a pretrained trading agent.
    Native feature flattening/uncertainty calibration requires explicit schemas.
    """
    import torch
    from torch import nn

    class EvidenceFusionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.projections = nn.ModuleDict({name:nn.Linear(size,width) for name,size in expert_feature_sizes.items()})
            self.market_projection = nn.Linear(market_features,width)
            self.cross_attention = nn.MultiheadAttention(width,4,batch_first=True)
            self.policy = nn.Linear(width,3)
            self.value = nn.Linear(width,1)
            self.allocation = nn.Linear(width,1)
            self.cash = nn.Linear(width,1)
            self.trained = False

        def forward(self, expert_features, market_state, key_padding_mask=None, allow_untrained=False, expert_gates=None):
            if not self.trained and not allow_untrained:
                raise RuntimeError("fusion head has no trained checkpoint; execution is disabled")
            query = self.market_projection(market_state)
            tokens = torch.stack([self.projections[name](features) for name,features in expert_features.items()],dim=-2)
            if expert_gates is not None:
                tokens = tokens * expert_gates.unsqueeze(-1)
            batch,symbols,experts,dim = tokens.shape
            mask = key_padding_mask.reshape(batch*symbols,experts) if key_padding_mask is not None else None
            unavailable = mask.all(-1) if mask is not None else torch.zeros(batch*symbols,dtype=torch.bool)
            if mask is not None:
                mask = mask.clone()
                mask[unavailable,0] = False  # Avoid all-masked attention NaNs; zero this row below.
            attended,attention = self.cross_attention(query.reshape(batch*symbols,1,dim),
                tokens.reshape(batch*symbols,experts,dim),tokens.reshape(batch*symbols,experts,dim),
                key_padding_mask=mask, need_weights=True)
            attended = attended.masked_fill(unavailable[:,None,None],0)
            attention = attention.masked_fill(unavailable[:,None,None],0)
            hidden = query + attended.reshape(batch,symbols,dim)
            asset_scores = self.allocation(hidden).squeeze(-1)
            cash_score = self.cash(hidden.mean(1))
            result = {"policy_logits":self.policy(hidden), "value":self.value(hidden).squeeze(-1),
                "allocation_scores":asset_scores, "cash_scores":cash_score,
                "shared_latent":hidden, "expert_attention":attention.reshape(batch,symbols,experts)}
            if any(not torch.isfinite(value).all() for value in result.values()):
                raise ValueError("fusion returned nonfinite values")
            return result

    return EvidenceFusionHead()
