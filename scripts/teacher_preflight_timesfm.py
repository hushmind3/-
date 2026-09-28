from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime-global-market-training/teacher-preflight/timesfm-pkg"))
from timesfm import ForecastConfig
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch

def main():
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = TimesFM_2p5_200M_torch.from_pretrained(
        str(ROOT / "models/teachers/TimesFM-2.5-200M"), torch_compile=False,
        local_files_only=True,
    )
    model.compile(ForecastConfig(max_context=128, max_horizon=128, normalize_inputs=True,
        per_core_batch_size=52, use_continuous_quantile_head=True,
        force_flip_invariance=True, infer_is_positive=False, fix_quantile_crossing=True))
    nparams = sum(p.numel() for p in model.model.parameters())
    df = pd.read_csv(ROOT / "data/global_market_daily.csv", parse_dates=["date"])
    panel = df.pivot(index="date", columns="symbol", values="close").sort_index()
    close = panel.to_numpy(np.float64)
    ret = np.diff(np.log(close), axis=0)
    preds, actuals, p10s, p90s = [], [], [], []
    inf_start = time.perf_counter()
    for target_t in range(683, min(828, len(panel))):
        end = target_t - 2
        x = ret[max(0, end-127):end+1]
        if x.shape[0] < 32: continue
        series = [np.nan_to_num(x[:, j], nan=0., posinf=0., neginf=0.).astype(np.float32) for j in range(x.shape[1])]
        point, quantiles = model.forecast(horizon=1, inputs=series)
        # TimesFM reports median in quantile index 5, with 10th/90th at 1/9.
        actual = ret[target_t-1]
        mask = np.isfinite(actual) & np.isfinite(point[:, 0])
        preds.extend(point[mask, 0].tolist()); actuals.extend(actual[mask].tolist())
        p10s.extend(quantiles[mask, 0, 1].tolist()); p90s.extend(quantiles[mask, 0, 9].tolist())
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-inf_start
    p, y = np.asarray(preds), np.asarray(actuals)
    result = {
      "teacher":"TimesFM-2.5-200M", "params":int(nparams), "dtype":"FP32",
      "checkpoint_bytes":(ROOT/"models/teachers/TimesFM-2.5-200M/model.safetensors").stat().st_size,
      "input":"per-symbol 128 daily log returns, univariate; no contemporaneous future features",
      "output":"point and 10 quantiles; median evaluated as point forecast",
      "validation_range":[str(panel.index[683].date()),str(panel.index[min(827,len(panel)-1)].date())],
      "predictions":len(p), "mae_teacher":float(np.mean(np.abs(p-y))),
      "mae_zero_baseline":float(np.mean(np.abs(y))),
      "rmse_teacher":float(np.sqrt(np.mean((p-y)**2))),
      "rmse_zero_baseline":float(np.sqrt(np.mean(y**2))),
      "directional_accuracy_teacher":float(np.mean(np.sign(p)==np.sign(y))),
      "directional_accuracy_tie_adjusted_chance":0.5,
      "mean_pred_abs_bps":float(np.mean(np.abs(p))*10000), "mean_actual_abs_bps":float(np.mean(np.abs(y))*10000),
      "inference_seconds":elapsed, "seconds_per_date":elapsed/max(1,min(828,len(panel))-683),
      "peak_cuda_allocated_bytes":int(torch.cuda.max_memory_allocated()),
      "total_load_plus_eval_seconds":time.perf_counter()-t0,
    }
    dest=ROOT/"runtime-global-market-training/teacher-preflight/timesfm-validation.json"
    dest.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__": main()
