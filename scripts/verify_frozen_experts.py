"""Run independent native inference and record compact results, without training.

Each CUDA worker finishes and exits before the next starts. This script never
starts the live agent, touches its replay, or registers the TradingMoE wrapper.
Input fixtures must already exist under the audited artifact root/verification.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "verification"
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
               PYTHONUTF8="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    work = [("chronos", "daily_excess"), ("timesfm", "daily_excess"),
            ("kronos", "bars"), ("exaone", "prices"), ("timemoe", "prices"),
            ("toto", "daily_excess"), ("fincast", "prices"), ("marketgpt", "itch_probe")]
    for regime in ("slope", "vol"):
        for label in (1, 2, 3):
            name = f"macrophft_{regime}_{label}"
            data = json.loads((out / "policy_probe.json").read_text(encoding="utf-8"))
            data["variant"] = f"{regime}/{label}"
            (out / (name + ".input.json")).write_text(json.dumps(data), encoding="utf-8")
            work.append((name, name + ".input"))
    summary_path = out / "independent_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    failures = []
    for name, input_name in work:
        expert = "macrophft" if name.startswith("macrophft_") else name
        if args.only and name not in args.only:
            continue
        python = root / ("venv-toto" if expert == "toto" else "venv") / "Scripts/python.exe"
        target = out / (name + ".json")
        started = time.perf_counter()
        try:
            proc = subprocess.run([str(python), "-m", "stockrl.expert_backends", expert,
                "--root", str(root), "--input", str(out / (input_name + ".json")),
                "--output", str(target), "--device", args.device], env=env,
                capture_output=True, text=True, encoding="utf-8", timeout=300)
            (out / (name + ".log")).write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
            if proc.returncode == 0:
                data = json.loads(target.read_text(encoding="utf-8"))
                row = {k:v for k,v in data.items() if k != "native_output"}
                row.update(name=name, status="ok")
            else:
                row = {"name":name, "status":"error", "error":proc.stderr[-2500:]}
        except Exception as exc:
            row = {"name":name, "status":"error", "error":str(exc)}
        row["process_seconds"] = time.perf_counter() - started
        summary = [previous for previous in summary if previous["name"] != name] + [row]
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if row["status"] != "ok":
            failures.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ("rss_start_bytes","rss_loaded_bytes","rss_end_bytes")}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
