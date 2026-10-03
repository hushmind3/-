"""Start the portable dashboard once, then open it in the user's browser."""
from __future__ import annotations

import os
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from stockrl.paths import default_runtime_dir
PORT = 8766
URL = f"http://127.0.0.1:{PORT}/"


def ready() -> bool:
    try:
        with urlopen(URL + "api/health", timeout=2) as response:
            payload = json.load(response)
            if (response.status == 200 and payload.get("service") == "stockrl"
                    and payload.get("port") == PORT):
                return True
    except (OSError, URLError, ValueError):
        return False
    return False


def port_in_use() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=.25):
            return True
    except OSError:
        return False


def main() -> int:
    open_browser = "--no-browser" not in sys.argv[1:]
    if ready():
        print(f"StockRL is already running: {URL}")
        if open_browser:
            webbrowser.open(URL, new=2)
        return 0
    if port_in_use():
        print(f"Port {PORT} is occupied by an unrecognized or older server; it was not replaced.",
              file=sys.stderr)
        return 1

    log_dir = default_runtime_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["STOCKRL_RUNTIME_DIR"] = str(log_dir)
    env["STOCKRL_WEB_PORT"] = str(PORT)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    command = [sys.executable, "-m", "stockrl.launch_web", "--server-only", "--dashboard-only"]
    for log_name in ("web.stdout.log", "web.stderr.log"):
        log_file=log_dir/log_name
        if log_file.exists() and log_file.stat().st_size >= 2*1024*1024:
            log_file.write_text("",encoding="utf-8")
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
            if open_browser:
                webbrowser.open(URL, new=2)
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.5)
    print(f"StockRL did not start. See {log_dir / 'web.stderr.log'}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
