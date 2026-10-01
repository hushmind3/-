"""Run the regression checks once and write a compact machine-readable result."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root/"src")+os.pathsep+env.get("PYTHONPATH", "")
    start = time.perf_counter()
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
                            cwd=root, env=env, capture_output=True, text=True, timeout=180)
    text = result.stdout+result.stderr
    count = re.search(r"Ran (\d+) tests?", text)
    report = {"completed_utc": datetime.now(timezone.utc).isoformat(),
              "passed": result.returncode == 0, "tests": int(count[1]) if count else None,
              "seconds": time.perf_counter()-start,
              "details": text[-6000:] if result.returncode else text[-300:]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "tests", "seconds")}))
    sys.exit(result.returncode)
