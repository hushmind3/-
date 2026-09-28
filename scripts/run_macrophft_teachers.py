"""Run six downloaded MacroHFT sub-agent checkpoints on their real ETH test feed."""
from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch

from stockrl.research_ingest import MacroHFTSubagent, macrophft_features


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external_sources/macrophft"
FEED = SOURCE / "data/df_test.feather"
OUT = SOURCE / "teacher_outputs.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEE = 0.0002  # MacroHFT's own low_level_env.py transaction fee
MAX_HOLDING = 0.2  # ETHUSDT max inventory set in MacroHFT's agent


def main() -> None:
    if DEVICE.type != "cuda":
        raise RuntimeError("This verification is expected to run on the available CUDA GPU")
    frame = pd.read_feather(FEED).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    x1, x2 = macrophft_features(frame, SOURCE / "feature_list")
    # Keep the run representative and bounded; choose uniformly across the
    # entire held-out period, without using its labels to change actions.
    stride = 10
    indices = np.arange(0, len(frame), stride)
    if len(indices) < 100:
        raise RuntimeError("held-out ETH dataset unexpectedly small")
    q_by_previous = np.zeros((len(indices), 2, 2), dtype=np.float32)
    sample1, sample2 = x1[indices], x2[indices]
    started = time.perf_counter()
    for regime in ("slope", "vol"):
        for label in (1, 2, 3):
            name = f"{regime}_{label}_best_model.pkl"
            state = torch.load(SOURCE / name, map_location="cpu", weights_only=True)
            model = MacroHFTSubagent().to(DEVICE).eval()
            model.load_state_dict(state, strict=True)
            outputs = []
            with torch.inference_mode():
                for lo in range(0, len(indices), 2048):
                    hi = min(lo + 2048, len(indices))
                    a = torch.from_numpy(sample1[lo:hi]).to(DEVICE)
                    b = torch.from_numpy(sample2[lo:hi]).to(DEVICE)
                    both_a = a.repeat_interleave(2, dim=0)
                    both_b = b.repeat_interleave(2, dim=0)
                    prev = torch.tensor([0, 1], dtype=torch.long, device=DEVICE).repeat(hi-lo)
                    q = model(both_a, both_b, prev).float().reshape(hi-lo, 2, 2)
                    outputs.append(q.cpu().numpy())
            q = np.concatenate(outputs)
            prev_position = 0
            actions, signals, values, pnl, confidences = [], [], [], [], []
            closes = frame.close.to_numpy(float)[indices]
            equity = 0.0
            equity_curve = []
            for k in range(len(indices)-1):
                q0, q1 = q[k, prev_position]
                target = int(q1 > q0)
                signal = "HOLD" if target == prev_position else ("BUY" if target else "SELL")
                p0, p1 = closes[k], closes[k+1]
                position = MAX_HOLDING * target
                trade = MAX_HOLDING * abs(target-prev_position)
                reward = position*(p1-p0) - trade*p0*FEE
                equity += reward
                actions.append(target); signals.append(signal); values.append(max(float(q0),float(q1)))
                pnl.append(float(reward)); confidences.append(abs(float(q1)-float(q0))); equity_curve.append(equity)
                prev_position = target
            net_pct = equity / max(closes[0]*MAX_HOLDING, 1e-12) * 100
            peak = np.maximum.accumulate(np.asarray(equity_curve)) if equity_curve else np.array([0.])
            dd = np.asarray(equity_curve)-peak
            model_rows = pd.DataFrame({
                "source": "MacroHFT", "model": f"{regime}_{label}",
                "timestamp": frame.timestamp.iloc[indices[:-1]].astype(str).to_numpy(),
                "symbol": "ETHUSDT", "action": signals, "target_long": actions,
                "confidence": confidences,
                "value": values, "net_pnl_usd_per_0.2_eth": pnl,
            })
            out_mode = "w" if regime == "slope" and label == 1 else "a"
            model_rows.to_csv(OUT, mode=out_mode, index=False, header=(out_mode=="w"))
            del model, state, outputs
            print(json.dumps({"model": f"{regime}_{label}", "rows": len(actions),
                              "actions": {a: signals.count(a) for a in ("BUY","HOLD","SELL")},
                              "net_pnl_usd_per_0.2_eth": round(equity,6),
                              "net_return_pct": round(net_pct,6),
                              "max_drawdown_usd": round(float(dd.min()),6)}, ensure_ascii=False), flush=True)
    print(json.dumps({"device": torch.cuda.get_device_name(DEVICE), "samples": len(indices),
                      "stride": stride, "rows_per_teacher": len(indices)-1,
                      "elapsed_seconds": round(time.perf_counter()-started,3),
                      "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
