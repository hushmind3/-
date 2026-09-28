"""One-click local web application launcher."""
from __future__ import annotations

import socket
import os
import sys
from pathlib import Path
from .web_app import serve


def _free_port(start: int = 8765, end: int = 8799) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free dashboard port between {start} and {end}.")


def main() -> None:
    server_only = "--server-only" in sys.argv[1:]
    port = int(os.environ.get("STOCKRL_WEB_PORT", "8767")) if server_only else _free_port()
    runtime = Path(os.environ.get("STOCKRL_RUNTIME_DIR", "runtime-global-korea-live"))
    initial_champion = runtime / "live" / "agent" / "champion.pt"
    print(f"StockRL starting: http://127.0.0.1:{port}/", flush=True)
    serve(
        host="127.0.0.1", port=port,
        runtime=str(runtime), device="auto",
        # A 0.5B update is deliberately amortized over a larger experience
        # tranche so the observer can keep up with one-minute bars. The
        # candidate learner remains asynchronous and is still automatic.
        candidate_every=4096, fee=0.001, auto_start=True, open_browser=not server_only,
        horizon="1m", config="configs/live_symbols_korea.json",
        initial_champion=str(initial_champion),
    )


if __name__ == "__main__":
    main()
