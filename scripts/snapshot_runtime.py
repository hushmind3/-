"""Disabled: StockRL keeps one live runtime and does not create runtime snapshots."""
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", default="korea")
    parser.add_argument("--allow-live", action="store_true")
    parser.parse_args()
    raise SystemExit(
        "Runtime snapshots are disabled. Runtime records are bounded and cleared in place."
    )


if __name__ == "__main__":
    main()
