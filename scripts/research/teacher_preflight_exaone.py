from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"runtime-global-market-training/teacher-preflight/exaone-pkg"))
import transformers.utils.import_utils as iu
iu._torchvision_available=False  # workaround for this host's unrelated torchvision NMS error
from exaone_forecast.finance.forecaster import EXAONEFinanceForecaster

def main():
    ckpt=ROOT/"runtime-global-market-training/teacher-preflight/exaone-checkpoint"
    torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter()
    model=EXAONEFinanceForecaster(ckpt_dir=str(ckpt),device="cuda:0")
    nparams=sum(p.numel() for p in model.model.parameters())
    df=pd.read_csv(ROOT/"data/global_market_daily.csv",parse_dates=["date"])
    panel=df.pivot(index="date",columns="symbol",values="close").sort_index()
    close=panel.to_numpy(np.float64); ret=np.diff(np.log(close),axis=0)
    preds=[]; actual=[]; q10=[]; q90=[]; rank_ics=[]; top_bottom=[]; top_quartile_returns=[]; universe_returns=[]
    start=time.perf_counter()
    for target_t in range(683,min(828,len(panel))):
        end=target_t-2; x=ret[max(0,end-511):end+1]
        if x.shape[0]<32: continue
        series=[x[:,j].astype(np.float32) for j in range(x.shape[1])]
        q=model.predict(series,horizon=1,batch_size=52) # [symbols, 21 quantiles, horizon]
        mid=model.quantiles.index(.5); i10=model.quantiles.index(.1); i90=model.quantiles.index(.9)
        p=q[:,mid,0]; lo=q[:,i10,0]; hi=q[:,i90,0]; y=ret[target_t-1]
        mask=np.isfinite(y)&np.isfinite(p)
        preds.extend(p[mask]); actual.extend(y[mask]); q10.extend(lo[mask]); q90.extend(hi[mask])
        if mask.sum()>=5:
            pp=pd.Series(p[mask]); yy=pd.Series(y[mask]); rank_ics.append(float(pp.corr(yy,method="spearman")))
            order=np.argsort(p[mask]); k=max(1,len(order)//4); top_bottom.append(float(y[mask][order[-k:]].mean()-y[mask][order[:k]].mean()))
            top_quartile_returns.append(float(y[mask][order[-k:]].mean())); universe_returns.append(float(y[mask].mean()))
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    p,y,lo,hi=map(np.asarray,(preds,actual,q10,q90))
    res={"teacher":"EXAONE Finance 1.0","params":int(nparams),"dtype":"FP32","checkpoint_bytes":(ROOT/"models/teachers/EXAONE-Forecast-for-Finance-1.0/exaone-finance-1.0.safetensors").stat().st_size,
      "input":"52 univariate channels, each with up to 512 prior daily log returns; NaNs retained; model's published checkpoint predicts channels independently",
      "output":"21 quantiles, median and q10/q90","validation_range":[str(panel.index[683].date()),str(panel.index[min(827,len(panel)-1)].date())],"predictions":len(p),
      "mae_teacher":float(np.mean(np.abs(p-y))),"mae_zero_baseline":float(np.mean(np.abs(y))),
      "rmse_teacher":float(np.sqrt(np.mean((p-y)**2))),"rmse_zero_baseline":float(np.sqrt(np.mean(y**2))),
      "directional_accuracy":float(np.mean(np.sign(p)==np.sign(y))),"tie_adjusted_chance":0.5,
      "q10_q90_coverage":float(np.mean((y>=lo)&(y<=hi))),"mean_daily_spearman_ic":float(np.nanmean(rank_ics)),
      "mean_top_quartile_minus_bottom_quartile_return":float(np.mean(top_bottom)),
      "mean_top_forecast_quartile_gross_daily_return":float(np.mean(top_quartile_returns)),
      "mean_equal_weight_universe_gross_daily_return":float(np.mean(universe_returns)),
      "top_quartile_daily_return_after_5bp_roundtrip_cost":float(np.mean(top_quartile_returns)-0.0005),
      "top_quartile_daily_return_after_10bp_roundtrip_cost":float(np.mean(top_quartile_returns)-0.001),
      "top_quartile_daily_return_after_20bp_roundtrip_cost":float(np.mean(top_quartile_returns)-0.002),
      "mean_pred_abs_bps":float(np.mean(np.abs(p))*10000),"mean_actual_abs_bps":float(np.mean(np.abs(y))*10000),
      "inference_seconds":elapsed,"seconds_per_date":elapsed/max(1,min(828,len(panel))-683),
      "peak_cuda_allocated_bytes":int(torch.cuda.max_memory_allocated()),"total_seconds":time.perf_counter()-t0}
    path=ROOT/"runtime-global-market-training/teacher-preflight/exaone-validation.json"; path.write_text(json.dumps(res,indent=2),encoding="utf-8")
    print(json.dumps(res,indent=2)); del model; torch.cuda.empty_cache()

if __name__=="__main__": main()
