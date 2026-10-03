from __future__ import annotations
import json, sys, time, types
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
TEACHER_DIR = ROOT / "models/teachers/TimeMoE-50M"
# Work around an unrelated broken torchvision installation in this environment.
import transformers.utils.import_utils as iu
iu._torchvision_available = False
pkg = types.ModuleType("timemoe_teacher")
pkg.__path__ = [str(TEACHER_DIR)]
sys.modules["timemoe_teacher"] = pkg
from timemoe_teacher.modeling_time_moe import TimeMoeForPrediction

def main():
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = TimeMoeForPrediction.from_pretrained(
        str(ROOT / "models/teachers/TimeMoE-50M"), torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).eval().to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    df = pd.read_csv(ROOT / "data/global_market_daily.csv", parse_dates=["date"])
    panel = df.pivot(index="date", columns="symbol", values="close").sort_index()
    close = panel.to_numpy(np.float64)
    ret = np.diff(np.log(close), axis=0)
    # Train/validation split comes from the shared market cache (time indices 0:682 / 682:828).
    # Validation forecasts use only observations available strictly before each target date.
    forecasts, actuals, naive = [], [], []
    inf_start = time.perf_counter()
    with torch.inference_mode():
        for target_t in range(683, min(828, len(panel))):
            # `ret[k]` describes price change from panel[k] to panel[k+1].
            # Target return is ret[target_t-1]; inputs must end at ret[target_t-2].
            end = target_t - 2
            x = ret[max(0, end-127):end+1]
            if x.shape[0] < 32: continue
            mu = np.nanmean(x, axis=0)
            sd = np.nanstd(x, axis=0).clip(1e-6)
            z = np.nan_to_num((x-mu)/sd, nan=0., posinf=0., neginf=0.).T[..., None].astype(np.float32)
            inp = torch.from_numpy(z).to(device=dev, dtype=torch.bfloat16)
            out = model(input_ids=inp, use_cache=False, max_horizon_length=1, return_dict=True).logits
            # last context token's one-step forecast, then return to the source scale
            pred = out[:, -1, 0].float().cpu().numpy() * sd + mu
            actual = ret[target_t-1]
            mask = np.isfinite(actual) & np.isfinite(pred)
            forecasts.extend(pred[mask].tolist()); actuals.extend(actual[mask].tolist())
            naive.extend(np.zeros(int(mask.sum())).tolist())
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-inf_start
    p, y = np.array(forecasts), np.array(actuals)
    result = {
        "teacher":"Time-MoE-50M", "params":int(nparams), "dtype":"BF16",
        "validation_range": [str(panel.index[683].date()), str(panel.index[min(827,len(panel)-1)].date())],
        "predictions":int(len(p)), "mae_teacher":float(np.mean(np.abs(p-y))),
        "mae_zero_baseline":float(np.mean(np.abs(y))),
        "rmse_teacher":float(np.sqrt(np.mean((p-y)**2))),
        "rmse_zero_baseline":float(np.sqrt(np.mean(y**2))),
        "directional_accuracy_teacher":float(np.mean(np.sign(p)==np.sign(y))),
        "directional_accuracy_tie_adjusted_chance":0.5,
        "mean_pred_abs_bps":float(np.mean(np.abs(p))*10000), "mean_actual_abs_bps":float(np.mean(np.abs(y))*10000),
        "inference_seconds":elapsed, "seconds_per_date":elapsed/max(1,min(828,len(panel))-683),
        "peak_cuda_allocated_bytes":int(torch.cuda.max_memory_allocated()),
        "load_and_eval_total_seconds":time.perf_counter()-t0,
    }
    dest = ROOT / "runtime-global-market-training" / "teacher-preflight" / "timemoe50m-validation.json"
    dest.parent.mkdir(parents=True, exist_ok=True); dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__": main()
