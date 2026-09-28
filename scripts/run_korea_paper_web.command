#!/bin/sh
set -eu
PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
RUNTIME_DIR="${STOCKRL_RUNTIME_DIR:-$HOME/Library/Application Support/StockRL/runtime-global-korea-live}"
MODEL_DIR="${STOCKRL_MODEL_DIR:-$HOME/Desktop/모델}"
python3 -m stockrl web \
  --host 127.0.0.1 \
  --port 8766 \
  --runtime "$RUNTIME_DIR" \
  --model-dir "$MODEL_DIR" \
  --device auto \
  --config configs/live_symbols_korea.json \
  --initial-champion "$MODEL_DIR/champion.pt" \
  --candidate-every 256 \
  --fee 0.001 \
  --horizon 1m
