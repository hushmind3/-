"""Register frozen experts or verify end-to-end adapter/router/fusion inference.

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


def all_expert_snapshot(root):
    """Same-as-of real stock inputs + explicitly synthetic ITCH/ETH schema probes."""
    import pandas as pd
    daily = json.loads((root / "verification/daily_excess.json").read_text(encoding="utf-8"))
    as_of = daily["as_of"]
    frame = pd.read_csv(Path(__file__).resolve().parents[1] / "data/global_market_daily.csv")
    rows = frame[(frame.symbol == "AAPL") & (frame.date <= as_of)].sort_values("date").tail(128)
    if len(rows) != 128 or str(rows.date.iloc[-1]) != as_of:
        raise ValueError("cannot construct a real same-as-of AAPL history")
    bars = rows.rename(columns={"date":"timestamp"}).copy()
    bars["amount"] = 0.0
    prices = {"symbols":["AAPL"], "as_of":as_of, "series":[rows.close.tolist()], "units":"price",
        "horizon":1, "sampling_seconds":86400, "frequency_id":1, "input_authenticity":"real_daily_price"}
    candles = {"symbols":["AAPL"], "as_of":as_of, "horizon":1, "sampling_seconds":86400,
        "bars":[bars[["timestamp","open","high","low","close","volume","amount"]].to_dict("records")],
        "future_timestamps":[str((pd.Timestamp(as_of)+pd.offsets.BDay()).date())], "amount_observed":False,
        "input_authenticity":"real_OHLCV_native_missing_amount_zero"}
    itch = json.loads((root / "verification/itch_probe.json").read_text(encoding="utf-8"))
    itch.update(as_of=as_of, sampling_seconds=1)
    policy = json.loads((root / "verification/policy_probe.json").read_text(encoding="utf-8"))
    policy.update(as_of=as_of, sampling_seconds=1)
    inputs = {"timesfm":daily, "chronos":daily, "toto":daily, "kronos":candles,
        "fincast":prices, "exaone":prices, "timemoe":prices, "marketgpt":itch}
    for regime in ("slope", "vol"):
        for label in (1,2,3):
            inputs[f"macrophft_{regime}_{label}"] = policy
    symbols = ["AAPL", "MSFT", "ETHUSDT"]
    return {"as_of":as_of, "symbols":symbols, "currencies":{s:"USD" for s in symbols},
        "current_weights":{s:0.0 for s in symbols}, "expert_inputs":inputs,
        "input_authenticity":"mixed_real_and_synthetic_schema_probe_not_market_validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-all", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    model = TradingMoE(args.root, device=args.device, registry_path=args.registry,
        top_k=14 if args.probe_all else 2, diversify_modalities=not args.probe_all)
    manifest = args.root / "TradingMoE.manifest.json"
    model.save_manifest(manifest)
    if args.probe or args.probe_all or args.snapshot:
        data = json.loads((args.root / "verification/daily_excess.json").read_text(encoding="utf-8"))
        snapshot = {"symbols":data["symbols"], "as_of":data["as_of"],
            "currencies":{s:"USD" for s in data["symbols"]},
            "current_weights":{s:0.0 for s in data["symbols"]},
            "expert_inputs":{"timesfm":data, "toto":data}}
        if args.probe_all:
            snapshot = all_expert_snapshot(args.root)
        if args.snapshot:
            snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
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
        atomic_json(args.root / "verification/TradingMoE.integrated_probe.json", result)
        model.save_manifest(manifest)
        print(json.dumps({"selected":result["selected_experts"], "raw_shapes":
            {p["expert"]:p["output_shape"] for p in result["profiles"]},
            "adapter_status":result["adapter_status"], "fusion_shapes":result["fusion_output"]["shapes"],
            "timings":{p["expert"]:p["timings"] for p in result["profiles"]},
            "pipeline_seconds":result["pipeline_timings"]["total_seconds"], "training_performed":False}))
    status = read_registry(model.registry_path)
    print(json.dumps({"registry":str(model.registry_path), "totals":status["totals"],
        "all_unloaded":all(not e["loaded"] and not e["active"] for e in status["experts"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
