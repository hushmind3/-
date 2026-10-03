"""TradingMoE runtime registry. Read-only status never imports torch or loads weights."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from .state_io import atomic_json as _atomic_json
from .paths import EXPERT_ASSETS_DIR, expert_weight_path


def atomic_json(path, payload):
    # Reuse the project's Windows-sharing retries and serialized writer.
    _atomic_json(payload, path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def ensure_registry(path):
    """Seed portable expert metadata without carrying process state or loading PT."""
    path=Path(path)
    template=EXPERT_ASSETS_DIR/"registry.template.json"
    if path.exists() or not template.exists():return
    document=json.loads(template.read_text(encoding="utf-8"))
    document["artifact_root"]=str(EXPERT_ASSETS_DIR)
    for entry in document["experts"]:
        entry["raw_output_path"]=str(EXPERT_ASSETS_DIR/entry["raw_output_path"])
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(path,document)


def read_registry(path):
    """Current residency comes from a matching live worker, never historical peaks.

    Checkpoint paths and process samples are local; no GPU probe, model import,
    API download or checkpoint hashing occurs in a dashboard request.
    """
    import psutil
    ensure_registry(path)
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"registered":False, "experts":[], "totals":{}, "error":None}
    root = Path(document["artifact_root"])
    rows = []
    for entry in document["experts"]:
        row = {k:v for k,v in entry.items() if k not in ("probe", "files", "worker", "raw_output_path")}
        row["output_shape"] = entry.get("last_output_shape", entry["probe"]["output_shape"])
        row["input_shapes"] = entry.get("last_input_shapes", entry["probe"]["input_shapes"])
        row["verified_forward_seconds"] = entry["probe"]["forward_seconds"]
        files = [expert_weight_path(root / f["path"]) for f in entry["files"] if not f.get("archive")]
        row["checkpoint_bytes"] = sum(p.stat().st_size for p in files if p.is_file())
        row["checkpoint_present"] = bool(files) and all(p.is_file() for p in files)
        row.update(loaded=False, active=False, location="disk", ram_bytes=0, vram_bytes=0)
        if not row["checkpoint_present"]:
            row["location"] = "missing"
        worker = entry.get("worker") or {}
        try:
            process = psutil.Process(worker["pid"])
            if abs(process.create_time() - worker["created_at"]) > .01:
                raise ValueError("worker PID was reused")
            row.update(active=True, runtime_stage="loading", ram_bytes=process.memory_info().rss)
            sample = json.loads(Path(worker["status_path"]).read_text(encoding="utf-8"))
            if sample["expert_id"] != entry["id"]:
                raise ValueError("status belongs to another expert")
            # On Windows a venv python.exe launcher owns a separate native
            # Python child. Authenticate ancestry, then measure that child.
            native = psutil.Process(sample["pid"])
            if native.pid != worker["pid"] and not any(p.pid == worker["pid"] for p in native.parents()):
                raise ValueError("sample process is not the launched worker")
            process = native
            loaded = sample["stage"] in ("loaded", "inference", "completed")
            row.update(loaded=loaded, active=sample["stage"] in ("loading", "loaded", "inference"),
                location=sample.get("location", ("GPU VRAM" if sample["device"].startswith("cuda") else "CPU RAM") if loaded else "disk"),
                ram_bytes=process.memory_info().rss, vram_bytes=sample.get("vram_bytes"),
                vram_reserved_bytes=sample.get("vram_reserved_bytes"),
                runtime_stage=sample["stage"], worker_pid=process.pid, sampled_at=sample.get("updated_at"))
        except (KeyError, OSError, ValueError, psutil.Error):
            pass
        rows.append(row)
    totals = {"expert_count":len(rows), "parameters":sum(r["parameters"] for r in rows),
        "checkpoint_bytes":sum(r["checkpoint_bytes"] for r in rows),
        "weight_bytes":sum(r["weight_bytes"] for r in rows),
        "ram_bytes":sum(r["ram_bytes"] for r in rows),
        "vram_bytes":None if any(r["vram_bytes"] is None for r in rows) else sum(r["vram_bytes"] for r in rows),
        "active_parameters":sum(r["parameters"] for r in rows if r["active"])}
    return {"registered":True, "experts":rows, "totals":totals,
        "updated_at":document.get("updated_at"), "unavailable":document.get("unavailable", []),
        "router_status":"deterministic_untrained", "fusion_head_status":document.get("pipeline", {}).get("fusion_head_status", "not_trained"),
        "pipeline":{k:v for k,v in document.get("pipeline", {}).items() if k != "result_path"},
        "training_performed":False, "live_integration":False,
        "recent_error":document.get("recent_error"), "error":None}


def read_raw_output(path, expert_id):
    """Only registry-owned paths can be read; callers cannot supply file paths."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    entry = next(e for e in document["experts"] if e["id"] == expert_id)
    target = Path(entry["raw_output_path"]).resolve()
    root = Path(document["artifact_root"]).resolve()
    if not target.is_relative_to(root):
        raise ValueError("raw output outside artifact root")
    packet = json.loads(target.read_text(encoding="utf-8"))
    return {"expert_id":entry["id"], "origin":entry["raw_output_origin"], "packet":packet}


def read_fusion_output(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    target = Path(document["pipeline"]["result_path"]).resolve()
    if not target.is_relative_to(Path(document["artifact_root"]).resolve()):
        raise ValueError("fusion output outside artifact root")
    return json.loads(target.read_text(encoding="utf-8"))
