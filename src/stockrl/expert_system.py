"""Standalone capability routing and typed evidence fusion, outside live trading.

The experts are pretrained. The common trading head is NOT pretrained: this
phase exposes its architecture and validates HOLD/current-weight outputs until
an explicitly authorized subsequent training phase supplies a verified head.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
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

from .gpu_scheduler import FairGpuScheduler
from .expert_registry import atomic_json, utc_now


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


def select_experts(inputs, top_k=2, weight_budget_bytes=4 * 1024**3, specs=EXPERTS):
    """Deterministic capability/cost admission, not a learned or calibrated gate.

    Weight budget is an admission ceiling per sequential expert, not proof that
    an inference fits. Measured working memory + device headroom is also needed.
    """
    if top_k < 1 or weight_budget_bytes < 1:
        raise ValueError("top_k and memory budget must be positive")
    chosen, used_modalities = [], set()
    for spec in sorted(specs, key=lambda item: item.priority):
        if (spec.name not in inputs or spec.weight_bytes > weight_budget_bytes
                or spec.modality in used_modalities):
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
        if data.get("symbols") != symbols:
            raise ValueError(f"{name} input symbol order differs")


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
                 weight_budget_bytes=4*1024**3, registry_path=None, adapters_enabled=False):
        self.root = Path(artifact_root).resolve()
        self.device = device
        self.scheduler = scheduler or FairGpuScheduler()
        self.top_k = top_k
        self.weight_budget_bytes = weight_budget_bytes
        self.adapters_enabled = adapters_enabled
        self.registry_path = Path(registry_path) if registry_path else Path(__file__).resolve().parents[2] / "runtime/trading_moe/registry.json"
        self.lock = threading.Lock()
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
                            "raw_output_origin", "last_peak_vram_bytes", "last_input_shapes", "last_output_shape"):
                    if key in old:
                        entry[key] = old[key]
        self._publish()

    def _publish(self):
        self.registry["updated_at"] = utc_now()
        atomic_json(self.registry_path, self.registry)

    def _verify_originals(self, entry):
        for artifact in entry["files"]:
            if artifact.get("archive"):
                continue
            path = self.root / artifact["path"]
            if path.stat().st_size != artifact["bytes"]:
                raise ValueError(f"original file size changed: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8*1024**2), b""):
                    digest.update(chunk)
            if digest.hexdigest() != artifact["sha256"]:
                raise ValueError(f"original checkpoint hash changed: {path}")

    def infer(self, snapshot):
        with self.lock:
            try:
                return self._infer(snapshot)
            except BaseException as exc:
                self.registry["recent_error"] = str(exc)
                self._publish()
                raise

    def _infer(self, snapshot):
        validate_snapshot(snapshot)
        selected = select_experts(snapshot["expert_inputs"], self.top_k, self.weight_budget_bytes, self.specs)
        if not selected:
            raise ValueError("no expert has an available compatible modality within the budget")
        packets = []
        by_id = {e["id"]:e for e in self.registry["experts"]}
        for entry in by_id.values():
            entry["router_selected"] = entry["id"] in {s.name for s in selected}
        self.registry["recent_error"] = None
        self._publish()
        for spec in selected:
            entry = by_id[spec.name]
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
                    packet = json.loads(out.read_text(encoding="utf-8"))
                    if packet["parameters"] != entry["parameters"] or packet["parameter_bytes"] != entry["weight_bytes"]:
                        raise ValueError(f"registered parameter schema changed: {spec.name}")
                    raw_path = self.root / "inference" / (spec.name + ".json")
                    atomic_json(raw_path, packet)
                    entry["raw_output_path"] = str(raw_path)
                    entry["raw_output_origin"] = "runtime_inference"
                if packet["as_of"] != snapshot["as_of"] or packet["symbols"] != snapshot["symbols"]:
                    raise ValueError("native packet provenance differs from the supplied market")
                packets.append(packet)
                packet["expert"] = spec.name
                entry["last_inference_seconds"] = packet["forward_seconds"]
                entry["last_used_at"] = utc_now()
                entry["last_peak_vram_bytes"] = packet["peak_allocated_bytes"]
                entry["last_input_shapes"] = packet["input_shapes"]
                entry["last_output_shape"] = packet["output_shape"]
                self._publish()
        return {"schema":"frozen_heterogeneous_experts_v1", "as_of":snapshot["as_of"],
                "selected_experts":[s.name for s in selected], "router_status":"deterministic_untrained",
                "evidence_tokens":evidence_tokens(packets), "trading_output":pending_head_output(snapshot),
                "shared_representation":adapter_features(packets) if self.adapters_enabled else [],
                "adapter_status":"structural_only" if self.adapters_enabled else "disabled_raw_review_phase",
                "profiles":[{k:v for k,v in p.items() if k != "native_output"} for p in packets],
                "training_performed":False, "live_integration":False}

    def save_manifest(self, path):
        files = [{k:v for k,v in entry.items() if k not in
            ("worker", "active", "loaded", "router_selected", "error", "last_inference_seconds", "last_used_at",
             "raw_output_path", "raw_output_origin", "last_peak_vram_bytes", "last_input_shapes", "last_output_shape")}
            for entry in self.registry["experts"]]
        payload = {"schema":"frozen_heterogeneous_experts_v1", "expert_artifacts":files,
            "unavailable":self.registry["unavailable"],
            "router":{"top_k":self.top_k, "weight_budget_bytes":self.weight_budget_bytes,
                      "status":"deterministic_untrained"}, "fusion_head_status":"not_trained",
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
        return cls(artifact_root, **kwargs)


def adapter_features(packets):
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
        shared.append({"expert":packet["expert"], "symbols":packet["symbols"],
            "units":packet["units"], "horizon":packet["horizon"], "as_of":packet["as_of"],
            "features":features.tolist(), "shape":list(features.shape),
            "mask":np.isfinite(features).tolist()})
    return shared


def build_fusion_head(expert_feature_sizes, market_features=16, width=64):
    """Architecture only: trained projections/cross-attention/trading heads later.

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

        def forward(self, expert_features, market_state):
            if not self.trained:
                raise RuntimeError("fusion head has no trained checkpoint; execution is disabled")
            query = self.market_projection(market_state)
            tokens = torch.stack([self.projections[name](features) for name,features in expert_features.items()],dim=-2)
            batch,symbols,experts,dim = tokens.shape
            attended,_ = self.cross_attention(query.reshape(batch*symbols,1,dim),
                tokens.reshape(batch*symbols,experts,dim),tokens.reshape(batch*symbols,experts,dim),need_weights=False)
            hidden = query + attended.reshape(batch,symbols,dim)
            asset_scores = self.allocation(hidden).squeeze(-1)
            cash_score = self.cash(hidden.mean(1))
            return self.policy(hidden),self.value(hidden).squeeze(-1),torch.softmax(torch.cat([asset_scores,cash_score],dim=-1),dim=-1)

    return EvidenceFusionHead()
