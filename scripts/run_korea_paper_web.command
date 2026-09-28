#!/bin/sh
set -eu
PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
python3 -m stockrl web \
  --host 127.0.0.1 \
  --port 8765 \
  --runtime runtime-global-korea-live \
  --device auto \
  --config configs/live_symbols_korea.json \
  --initial-champion runtime-global-korea-live/live/agent/champion.pt \
  --candidate-every 256 \
  --fee 0.001 \
  --horizon 1m
