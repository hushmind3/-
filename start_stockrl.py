"""Start the portable dashboard once, then open it in the user's browser."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


ROOT = Path(__file__).resolve().parent
PORT = 8766
URL = f"http://127.0.0.1:{PORT}/"


def default_runtime_dir() -> Path:
    configured = os.environ.get("STOCKRL_RUNTIME_DIR")
    if configured:
        return Path(configured).expanduser()
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / "AppData" / "Local"
    return base / "StockRL" / "runtime-global-korea-live"


def ready() -> bool:
    try:
        with urlopen(URL + "api/health", timeout=2) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def main() -> int:
    if ready():
        print(f"StockRL is already running: {URL}")
        webbrowser.open(URL, new=2)
        return 0

    log_dir = default_runtime_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["STOCKRL_RUNTIME_DIR"] = str(log_dir)
    env["STOCKRL_WEB_PORT"] = str(PORT)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", "stockrl.launch_web", "--server-only"]
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    with (log_dir / "web.stdout.log").open("a", encoding="utf-8") as output, \
            (log_dir / "web.stderr.log").open("a", encoding="utf-8") as errors:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=errors, creationflags=flags)
    for _ in range(40):
        if ready():
            print(f"StockRL started (PID {process.pid}): {URL}")
            webbrowser.open(URL, new=2)
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.5)
    print(f"StockRL did not start. See {log_dir / 'web.stderr.log'}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
