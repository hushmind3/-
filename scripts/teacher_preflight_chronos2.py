from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"runtime-global-market-training/teacher-preflight/chronos-pkg"))
import transformers.utils.import_utils as iu
iu._torchvision_available=False  # workaround for this host's unrelated torchvision NMS error
from chronos import BaseChronosPipeline

def main():
    torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter()
    pipe=BaseChronosPipeline.from_pretrained(str(ROOT/"models/teachers/Chronos-2"),device_map="cuda")
    df=pd.read_csv(ROOT/"data/global_market_daily.csv",parse_dates=["date"])
    panel=df.pivot(index="date",columns="symbol",values="close").sort_index()
    close=panel.to_numpy(np.float64); ret=np.diff(np.log(close),axis=0)
    preds=[]; actual=[]; q10=[]; q90=[]; rank_ics=[]; spreads=[]; top_returns=[]; universe_returns=[]; dates=time.perf_counter()
    for target_t in range(683,min(828,len(panel))):
        end=target_t-2
        x=ret[max(0,end-127):end+1]
        if x.shape[0]<32: continue
        inp=x.T[None,:,:].astype(np.float32) # one related multivariate series: all symbols as variates
        out=pipe.predict(inp,prediction_length=1,batch_size=1,unrolled_quantiles=[.1,.5,.9])[0]
        pred=out[:,10,0].float().cpu().numpy(); lo=out[:,2,0].float().cpu().numpy(); hi=out[:,18,0].float().cpu().numpy()
        y=ret[target_t-1]; mask=np.isfinite(y)&np.isfinite(pred)
        preds.extend(pred[mask]); actual.extend(y[mask]); q10.extend(lo[mask]); q90.extend(hi[mask])
        if mask.sum()>=5:
            pp=pd.Series(pred[mask]); yy=pd.Series(y[mask]); rank_ics.append(float(pp.corr(yy,method="spearman")))
            order=np.argsort(pred[mask]); k=max(1,len(order)//4)
            spreads.append(float(y[mask][order[-k:]].mean()-y[mask][order[:k]].mean()))
            top_returns.append(float(y[mask][order[-k:]].mean())); universe_returns.append(float(y[mask].mean()))
    torch.cuda.synchronize(); elapsed=time.perf_counter()-dates
    p,y=np.asarray(preds),np.asarray(actual); lo,hi=np.asarray(q10),np.asarray(q90)
    res={"teacher":"Chronos-2","params":119477664,"dtype":"FP32","checkpoint_bytes":(ROOT/"models/teachers/Chronos-2/model.safetensors").stat().st_size,
      "input":"joint 52-symbol series, 128 prior daily log returns; NaNs retained as missing values","output":"21 quantiles, median and q10/q90; multivariate group forecast",
      "validation_range":[str(panel.index[683].date()),str(panel.index[min(827,len(panel)-1)].date())],"predictions":len(p),
      "mae_teacher":float(np.mean(np.abs(p-y))),"mae_zero_baseline":float(np.mean(np.abs(y))),
      "rmse_teacher":float(np.sqrt(np.mean((p-y)**2))),"rmse_zero_baseline":float(np.sqrt(np.mean(y**2))),
      "directional_accuracy":float(np.mean(np.sign(p)==np.sign(y))),"tie_adjusted_chance":0.5,
      "q10_q90_coverage":float(np.mean((y>=lo)&(y<=hi))),"mean_pred_abs_bps":float(np.mean(np.abs(p))*10000),
      "mean_daily_spearman_ic":float(np.nanmean(rank_ics)),"mean_top_quartile_minus_bottom_quartile_return":float(np.mean(spreads)),
      "mean_top_forecast_quartile_gross_daily_return":float(np.mean(top_returns)),"mean_equal_weight_universe_gross_daily_return":float(np.mean(universe_returns)),
      "top_quartile_daily_return_after_5bp_roundtrip_cost":float(np.mean(top_returns)-0.0005),
      "top_quartile_daily_return_after_10bp_roundtrip_cost":float(np.mean(top_returns)-0.001),
      "top_quartile_daily_return_after_20bp_roundtrip_cost":float(np.mean(top_returns)-0.002),
      "inference_seconds":elapsed,"seconds_per_date":elapsed/max(1,min(828,len(panel))-683),
      "peak_cuda_allocated_bytes":int(torch.cuda.max_memory_allocated()),"total_seconds":time.perf_counter()-t0}
    path=ROOT/"runtime-global-market-training/teacher-preflight/chronos2-validation.json"; path.write_text(json.dumps(res,indent=2),encoding="utf-8")
    print(json.dumps(res,indent=2)); del pipe; torch.cuda.empty_cache()

if __name__=="__main__": main()
