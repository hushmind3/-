"""Register independently verified frozen experts; optionally run a raw-only probe.

No live account, replay, optimizer or training. A shared GPU must have no other
agent owner during this isolated probe. The registry is read by the dashboard.
"""
from pathlib import Path
import argparse
import json
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stockrl.expert_system import TradingMoE
from stockrl.expert_registry import read_registry, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    model = TradingMoE(args.root, device=args.device, registry_path=args.registry)
    manifest = args.root / "TradingMoE.manifest.json"
    model.save_manifest(manifest)
    if args.probe:
        data = json.loads((args.root / "verification/daily_excess.json").read_text(encoding="utf-8"))
        snapshot = {"symbols":data["symbols"], "as_of":data["as_of"],
            "currencies":{s:"USD" for s in data["symbols"]},
            "current_weights":{s:0.0 for s in data["symbols"]},
            "expert_inputs":{"timesfm":data, "toto":data}}
        stopped = threading.Event()
        trace = []
        def sample_residency():
            while not stopped.is_set():
                status = read_registry(model.registry_path)
                trace.append({"at":time.time(), "active":[{"id":e["id"], "location":e["location"],
                    "loaded":e["loaded"], "vram_bytes":e["vram_bytes"]} for e in status["experts"] if e["active"]]})
                stopped.wait(.1)
        sampler = threading.Thread(target=sample_residency, daemon=True)
        sampler.start()
        try:
            result = model.infer(snapshot)
        finally:
            stopped.set()
            sampler.join()
            atomic_json(args.root / "verification/residency_trace.json", trace)
        maximum = max(len(t["active"]) for t in trace)
        if maximum > 1:
            raise RuntimeError("more than one expert ran concurrently")
        print(json.dumps({"max_concurrent_experts":maximum,
            "observed_gpu_residency":any("GPU" in e["location"] for t in trace for e in t["active"]),
            "samples":len(trace)}))
        if result["training_performed"] or result["trading_output"]["executable"]:
            raise RuntimeError("probe must not train or execute a random trading head")
        atomic_json(args.root / "verification/TradingMoE.raw_probe.json", result)
        print(json.dumps({"selected":result["selected_experts"], "raw_shapes":
            {p["expert"]:p["output_shape"] for p in result["profiles"]},
            "adapter_status":result["adapter_status"], "training_performed":False}))
    status = read_registry(model.registry_path)
    print(json.dumps({"registry":str(model.registry_path), "totals":status["totals"],
        "all_unloaded":all(not e["loaded"] and not e["active"] for e in status["experts"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
