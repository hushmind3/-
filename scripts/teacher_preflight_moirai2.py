from __future__ import annotations
import json, sys, time, types
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/"runtime-global-market-training/teacher-preflight/uni2ts-pkg"
sys.path.insert(0,str(PKG))
# Load the official raw forecast module without optional Lightning/GluonTS wrappers;
# this mirrors Moirai2Forecast._convert and its quantile unpacking for one-step inference.
moirai_pkg=types.ModuleType("uni2ts.model.moirai2"); moirai_pkg.__path__=[str(PKG/"uni2ts/model/moirai2")]
sys.modules["uni2ts.model.moirai2"]=moirai_pkg
from uni2ts.model.moirai2.module import Moirai2Module

def main():
    torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter()
    model=Moirai2Module.from_pretrained(str(ROOT/"models/teachers/Moirai-2.0-R-small")).cuda().eval()
    nparams=sum(p.numel() for p in model.parameters()); qs=list(model.quantile_levels); mid=qs.index(.5); i10=qs.index(.1); i90=qs.index(.9)
    df=pd.read_csv(ROOT/"data/global_market_daily.csv",parse_dates=["date"])
    panel=df.pivot(index="date",columns="symbol",values="close").sort_index()
    close=panel.to_numpy(np.float64); ret=np.diff(np.log(close),axis=0)
    preds=[]; actual=[]; q10=[]; q90=[]; rank_ics=[]; spreads=[]
    start=time.perf_counter()
    with torch.inference_mode():
      for target_t in range(683,min(828,len(panel))):
        end=target_t-2; x=ret[max(0,end-511):end+1]
        if x.shape[0]<32: continue
        b=x.shape[1]; context=np.full((b,512),np.nan,dtype=np.float32)
        context[:,-x.shape[0]:]=x.T.astype(np.float32)
        observed=torch.isfinite(torch.from_numpy(context)).to("cuda")
        target=torch.nan_to_num(torch.from_numpy(context),nan=0.0).to("cuda")
        context_tok=32; total_tok=33; patch=16
        target=torch.cat([target.reshape(b,context_tok,patch),torch.zeros((b,1,patch),device="cuda")],dim=1)
        obs=torch.cat([observed.reshape(b,context_tok,patch),torch.zeros((b,1,patch),dtype=torch.bool,device="cuda")],dim=1)
        sample_id=torch.ones((b,total_tok),dtype=torch.long,device="cuda")
        time_id=torch.arange(total_tok,device="cuda").expand(b,-1)
        var_id=torch.zeros((b,total_tok),dtype=torch.long,device="cuda")
        prediction_mask=torch.zeros((b,total_tok),dtype=torch.bool,device="cuda"); prediction_mask[:,-1]=True
        out=model(target,obs,sample_id,time_id,var_id,prediction_mask,training_mode=False)
        # Official Moirai2Forecast takes the last context token's first predict token.
        out=out.reshape(b,total_tok,model.num_predict_token,model.num_quantiles,patch)
        q=out[:,context_tok-1,0,:,:]
        p=q[:,mid,0].float().cpu().numpy(); lo=q[:,i10,0].float().cpu().numpy(); hi=q[:,i90,0].float().cpu().numpy(); y=ret[target_t-1]
        mask=np.isfinite(y)&np.isfinite(p); preds.extend(p[mask]); actual.extend(y[mask]); q10.extend(lo[mask]); q90.extend(hi[mask])
        if mask.sum()>=5:
          pp=pd.Series(p[mask]); yy=pd.Series(y[mask]); rank_ics.append(float(pp.corr(yy,method="spearman")))
          order=np.argsort(p[mask]); k=max(1,len(order)//4); spreads.append(float(y[mask][order[-k:]].mean()-y[mask][order[:k]].mean()))
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    p,y,lo,hi=map(np.asarray,(preds,actual,q10,q90))
    res={"teacher":"Moirai 2.0 R-small","params":int(nparams),"dtype":"FP32","checkpoint_bytes":(ROOT/"models/teachers/Moirai-2.0-R-small/model.safetensors").stat().st_size,
      "input":"52 independent univariate channels, up to 512 prior daily log returns; patch=16; missing observations masked",
      "output":"9 quantiles; median and q10/q90","validation_range":[str(panel.index[683].date()),str(panel.index[min(827,len(panel)-1)].date())],"predictions":len(p),
      "mae_teacher":float(np.mean(np.abs(p-y))),"mae_zero_baseline":float(np.mean(np.abs(y))),
      "rmse_teacher":float(np.sqrt(np.mean((p-y)**2))),"rmse_zero_baseline":float(np.sqrt(np.mean(y**2))),
      "directional_accuracy":float(np.mean(np.sign(p)==np.sign(y))),"tie_adjusted_chance":0.5,
      "q10_q90_coverage":float(np.mean((y>=lo)&(y<=hi))),"mean_daily_spearman_ic":float(np.nanmean(rank_ics)),
      "mean_top_quartile_minus_bottom_quartile_return":float(np.mean(spreads)),"mean_pred_abs_bps":float(np.mean(np.abs(p))*10000),
      "inference_seconds":elapsed,"seconds_per_date":elapsed/max(1,min(828,len(panel))-683),
      "peak_cuda_allocated_bytes":int(torch.cuda.max_memory_allocated()),"total_seconds":time.perf_counter()-t0}
    path=ROOT/"runtime-global-market-training/teacher-preflight/moirai2-validation.json"; path.write_text(json.dumps(res,indent=2),encoding="utf-8")
    print(json.dumps(res,indent=2)); del model; torch.cuda.empty_cache()

if __name__=="__main__": main()
