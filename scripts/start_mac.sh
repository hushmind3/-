#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RUNTIME_DIR="${STOCKRL_RUNTIME_DIR:-$HOME/Library/Application Support/StockRL/runtime-global-korea-live}"
MODEL_DIR="${STOCKRL_MODEL_DIR:-$HOME/Desktop/모델}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi

echo "StockRL Mac dashboard"
echo "Project: $ROOT"
"$PYTHON_BIN" -c 'import torch; print("PyTorch:", torch.__version__); print("MPS:", bool(torch.backends.mps.is_available()))'
echo "Opening http://127.0.0.1:8766/"
exec "$PYTHON_BIN" -m stockrl web \
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
