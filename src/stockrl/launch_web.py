"""One-click local web application launcher."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from .web_app import serve
from .paths import default_runtime_dir


def main() -> None:
    server_only = "--server-only" in sys.argv[1:]
    dashboard_only = "--dashboard-only" in sys.argv[1:]
    port = 8766
    runtime = default_runtime_dir()
    model_dir = Path(os.environ.get("STOCKRL_MODEL_DIR", str(Path.home() / "Desktop" / "모델")))
    initial_champion = model_dir / "champion.pt"
    print(f"StockRL starting: http://127.0.0.1:{port}/", flush=True)
    serve(
        host="127.0.0.1", port=port,
        runtime=str(runtime), device="auto",
        # Candidate training runs asynchronously beside live observation.
        candidate_every=16, fee=0.001, auto_start=not dashboard_only,
        open_browser=not server_only,
        horizon="1m", config="configs/live_symbols_korea.json",
        initial_champion=str(initial_champion), model_dir=str(model_dir),
    )


if __name__ == "__main__":
    main()
