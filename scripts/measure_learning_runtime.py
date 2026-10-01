"""Observe real completed learning batches; never run extra training or alter modes.

Writes one bounded JSON report, replacing it as each new batch completes.
The process runs independently of Codex and exits after --rounds per model.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
from datetime import datetime, timezone


FIELDS = (
    "completed_utc", "model_version", "samples", "unique_samples", "optimizer_steps",
    "total_seconds", "compute_seconds", "checkpoint_seconds", "optimizer_state_seconds",
    "replay_acknowledge_seconds", "optimizer_state_resumed", "samples_per_compute_second",
    "samples_per_total_second", "peak_allocated_bytes", "frozen_prefix_cache",
)


def measure(args):
    started = time.monotonic()
    seen = {}
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "target_steps": args.steps, "rounds_per_model": args.rounds,
              "baseline": {}, "rounds": {"champion": [], "candidate": []}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    last_write = 0.0
    while time.monotonic() - started < args.timeout:
        changed = False
        try:
            metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
            for role in report["rounds"]:
                value = metrics.get(role + "_last_completed_round", {})
                key = value.get("completed_utc")
                if not key or seen.get(role) == key:
                    continue
                row = {field: value[field] for field in FIELDS if field in value}
                if role not in seen:
                    report["baseline"][role] = row
                elif len(report["rounds"][role]) < args.rounds and value.get("optimizer_steps") == args.steps:
                    report["rounds"][role].append(row)
                seen[role] = key
                changed = True
            report["errors"] = {role: metrics.get("last_" + role + "_error") for role in report["rounds"]}
            report["applied_steps"] = metrics.get("runtime_updates", {}).get("applied_rules", {}).get("training_optimizer_steps")
            if all(len(rows) >= args.rounds for rows in report["rounds"].values()):
                report["status"] = "complete"
            report.pop("read_error", None)
        except (OSError, ValueError) as exc:
            report["read_error"] = str(exc)
        if changed or time.monotonic() - last_write >= 30 or report["status"] == "complete":
            report["updated_utc"] = datetime.now(timezone.utc).isoformat()
            report["summary"] = {}
            for role, rows in report["rounds"].items():
                total = sum(row.get("total_seconds", 0) for row in rows)
                samples = sum(row.get("samples", 0) for row in rows)
                report["summary"][role] = {
                    "completed_batches": len(rows), "samples": samples,
                    "samples_per_total_second": samples / total if total else None,
                    "median_batch_seconds": statistics.median(row["total_seconds"] for row in rows) if rows else None,
                    "checkpoint_seconds": sum(row.get("checkpoint_seconds", 0) for row in rows),
                }
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            last_write = time.monotonic()
        if report["status"] == "complete":
            return
        time.sleep(args.poll)
    report["status"] = "timeout"
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=root / "runtime/markets/korea/live/agent/metrics.json")
    parser.add_argument("--output", type=Path, default=root / "runtime/learning_measurement.json")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--poll", type=float, default=5)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 100 or args.steps < 1 or args.poll < 1 or args.timeout <= 0:
        parser.error("rounds must be 1..100; steps >= 1; poll >= 1; timeout > 0")
    measure(args)
