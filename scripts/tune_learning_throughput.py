"""Compare live saved-batch sizes automatically, with no extra replay passes.

This changes only training_optimizer_steps through the existing hot reload.
Models, accounts, observation and replay data remain live. One JSON report is
overwritten, and the original setting is restored on failure. Changing the
setting manually stops the experiment instead of overriding that change.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import statistics
import time

from measure_learning_runtime import FIELDS


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".next")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def change_steps(path, value, expected):
    rules = read(path)
    if rules["training_optimizer_steps"] != expected:
        raise RuntimeError("The optimizer setting was changed outside this measurement; leaving it intact.")
    rules["training_optimizer_steps"] = value
    write(path, rules)


def summarize(phase):
    summary = {}
    for role, rows in phase["rounds"].items():
        samples = sum(row["samples"] for row in rows)
        seconds = sum(row["total_seconds"] for row in rows)
        summary[role] = {"batches": len(rows), "samples": samples,
            "examples_per_second": samples/seconds if seconds else None,
            "median_batch_seconds": statistics.median(row["total_seconds"] for row in rows) if rows else None}
        for field in ("compute_seconds", "setup_seconds", "batch_load_seconds", "gpu_wait_seconds",
                      "checkpoint_seconds", "optimizer_state_seconds", "replay_acknowledge_seconds",
                      "window_forwards", "replay_rows_deleted"):
            summary[role][field] = sum(row.get(field, 0) or 0 for row in rows)
        losses = [row["loss_mean"] for row in rows if row.get("loss_mean") is not None]
        summary[role]["loss_mean"] = statistics.mean(losses) if losses else None
        summary[role]["loss_min"] = min((row["loss_min"] for row in rows if row.get("loss_min") is not None), default=None)
        summary[role]["loss_max"] = max((row["loss_max"] for row in rows if row.get("loss_max") is not None), default=None)
        summary[role]["peak_vram_bytes"] = max((row.get("peak_allocated_bytes", 0) or 0 for row in rows), default=0)
    phase["summary"] = summary
    total_samples = sum(item["samples"] for item in summary.values())
    total_seconds = sum(row["total_seconds"] for rows in phase["rounds"].values() for row in rows)
    phase["combined_examples_per_second"] = total_samples/total_seconds if total_seconds else None
    if phase.get("measured_started"):
        phase["wall_seconds"] = time.monotonic()-phase["measured_started"]
        phase["wall_examples_per_second"] = total_samples/max(phase["wall_seconds"], .001)


def run(args):
    original = read(args.rules)["training_optimizer_steps"]
    expected = original
    report = {"status": "running", "started_utc": now(), "original_steps": original,
              "samples_per_model_per_phase": args.samples, "phases": [],
              "limitations": "Live inputs and inference load vary between phases. Loss changes are not a proof of improved profitability."}
    if args.baseline and args.baseline.exists():
        report["before_optimization"] = read(args.baseline)
    try:
        for steps in args.steps:
            phase = {"steps": steps, "status": "applying", "rounds": {"champion": [], "candidate": []},
                     "warmup": {}, "started_utc": now(), "missed_rounds": {"champion": 0, "candidate": 0}}
            report["phases"].append(phase)
            initial = read(args.metrics)
            seen = {role: initial.get(role+"_last_completed_round", {}).get("completed_utc") for role in phase["rounds"]}
            versions = {}
            change_steps(args.rules, steps, expected)
            expected = steps
            deadline = time.monotonic()+args.phase_timeout
            last_write = 0.0
            while time.monotonic() < deadline:
                if read(args.rules)["training_optimizer_steps"] != expected:
                    raise RuntimeError("Optimizer steps changed manually; measurement stopped.")
                try:
                    metrics = read(args.metrics)
                except (OSError, ValueError):
                    time.sleep(args.poll)
                    continue
                applied = metrics.get("runtime_updates", {})
                if applied.get("status") == "rejected":
                    raise RuntimeError(applied.get("error", "runtime update rejected"))
                report["latest_errors"] = {role: metrics.get("last_"+role+"_error") for role in phase["rounds"]}
                if any(report["latest_errors"].values()):
                    raise RuntimeError(str(report["latest_errors"]))
                dirty = False
                for role, rows in phase["rounds"].items():
                    current = metrics.get(role+"_last_completed_round", {})
                    key = current.get("completed_utc")
                    if not key or key == seen[role]:
                        continue
                    seen[role] = key
                    if current.get("optimizer_steps") != steps or current.get("loss_mean") is None:
                        continue
                    row = {field: current[field] for field in FIELDS if field in current}
                    if role in versions and current["model_version"]-versions[role] > steps:
                        phase["missed_rounds"][role] += max(0, (current["model_version"]-versions[role])//steps-1)
                    versions[role] = current["model_version"]
                    if role not in phase["warmup"]:
                        phase["warmup"][role] = row
                    elif len(phase["warmup"]) == 2 and sum(item["samples"] for item in rows) < args.samples:
                        rows.append(row)
                    if len(phase["warmup"]) == 2 and not phase.get("measured_started"):
                        phase["measured_started"] = time.monotonic()
                        phase["status"] = "measuring"
                    dirty = True
                if all(sum(row["samples"] for row in rows) >= args.samples for rows in phase["rounds"].values()):
                    phase["status"] = "complete"
                if dirty or time.monotonic()-last_write >= 30:
                    summarize(phase)
                    report["updated_utc"] = now()
                    write(args.output, report)
                    last_write = time.monotonic()
                if phase["status"] == "complete":
                    summarize(phase)
                    phase.pop("measured_started", None)
                    break
                time.sleep(args.poll)
            else:
                raise TimeoutError(f"Steps={steps} did not produce enough completed batches within {args.phase_timeout}s.")
        if any(sum(phase["missed_rounds"].values()) for phase in report["phases"]):
            raise RuntimeError("Some completed batches were missed; no optimum selected from incomplete observations.")
        best = max(report["phases"], key=lambda phase: phase["combined_examples_per_second"])
        report["recommended_steps"] = best["steps"]
        selected = best["steps"] if args.apply_best else original
        change_steps(args.rules, selected, expected)
        expected = selected
        report["status"] = "applying_best"
        report["recommended_steps"] = best["steps"]
        report["selected_steps"] = selected
        write(args.output, report)
        deadline = time.monotonic()+180
        while time.monotonic() < deadline:
            if read(args.rules)["training_optimizer_steps"] != selected:
                raise RuntimeError("Final optimizer setting was changed manually.")
            applied = read(args.metrics).get("runtime_updates", {})
            if applied.get("applied_rules", {}).get("training_optimizer_steps") == selected and applied.get("status") == "applied":
                report["applied_final_steps"] = selected
                break
            time.sleep(args.poll)
        else:
            raise TimeoutError("The selected setting did not finish applying within 180s.")
        report.update(status="complete", finished_utc=now(), selected_steps=selected,
                      selection_metric="examples per second including both model saves and replay acknowledgement")
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_utc=now())
        try:
            if read(args.rules)["training_optimizer_steps"] == expected:
                change_steps(args.rules, original, expected)
                report["restored_steps"] = original
        except (OSError, ValueError):
            pass
    finally:
        write(args.output, report)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=root/"runtime/markets/korea/live/agent/metrics.json")
    parser.add_argument("--rules", type=Path, default=root/"configs/online_learning.json")
    parser.add_argument("--output", type=Path, default=root/"runtime/learning_tuning.json")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--phase-timeout", type=float, default=2400)
    parser.add_argument("--poll", type=float, default=2)
    parser.add_argument("--apply-best", action="store_true")
    args = parser.parse_args()
    if args.samples < 256 or any(step < 1 for step in args.steps) or args.poll < .5:
        parser.error("samples >= 256, steps >= 1, poll >= 0.5 required")
    def stop(signum, frame):
        raise KeyboardInterrupt("measurement terminated")
    signal.signal(signal.SIGTERM, stop)
    run(args)
