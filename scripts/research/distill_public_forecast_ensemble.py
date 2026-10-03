"""Cache and distill the downloaded public forecasting teachers, one at a time.

The active student is never on CUDA during teacher inference. Each teacher is
unloaded before the next; the student is loaded only after all targets have
been written to disk. Only chronological train/validation splits are used.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import distill_fincast_offline as base
import distill_fincast_temporal as temporal
from bench_market_training_loader import batch_tensors, utility
from stockrl.global_transformer import TIME_SCALE_NAMES, parameter_count
from stockrl.market_training import load_market_candidate_checkpoint


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def returns_for(row: dict) -> tuple[list[int], list[np.ndarray]]:
    ids, series = [], []
    for j, close, _last in row["series"]:
        close = np.asarray(close, np.float32)
        if len(close) < 32 or np.any(close <= 0):
            continue
        r = np.diff(np.log(close)).astype(np.float32)
        if np.isfinite(r).sum() < 16:
            continue
        ids.append(int(j)); series.append(r)
    return ids, series


def distill_quantiles(qlo, qmed, qhi, teacher: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qlo = np.expm1(np.asarray(qlo, np.float64))
    qmed = np.expm1(np.asarray(qmed, np.float64))
    qhi = np.expm1(np.asarray(qhi, np.float64))
    probs, value = base.quantile_action_targets(qlo, qmed, qhi, 0.0022)
    return np.asarray(probs, np.float32), np.asarray(value, np.float32), np.stack((qlo, qmed, qhi), -1).astype(np.float32)


def run_chronos(rows, device):
    import transformers.utils.import_utils as iu
    iu._torchvision_available = False
    pkg = ROOT / "runtime-global-market-training/teacher-preflight/chronos-pkg"
    sys.path.insert(0, str(pkg))
    from chronos import BaseChronosPipeline
    path = ROOT / "models/teachers/Chronos-2"
    model = BaseChronosPipeline.from_pretrained(str(path), device_map="cuda")
    out = []
    with torch.inference_mode():
        for row in rows:
            ids, rs = returns_for(row)
            if not ids:
                out.append((ids, np.empty((0,3),np.float32), np.empty(0,np.float32), np.empty((0,3),np.float32))); continue
            x = np.full((len(rs), 128), np.nan, np.float32)
            for k, r in enumerate(rs): x[k, -min(128,len(r)):] = r[-128:]
            q = model.predict(x[None], prediction_length=5, batch_size=1,
                              unrolled_quantiles=[.1,.5,.9])[0]
            lo, med, hi = q[:,2,4].float().cpu().numpy(), q[:,10,4].float().cpu().numpy(), q[:,18,4].float().cpu().numpy()
            probs, values, quant = distill_quantiles(lo,med,hi,"Chronos-2")
            out.append((ids,probs,values,quant))
    del model; torch.cuda.empty_cache(); gc.collect(); sys.path.remove(str(pkg))
    return out, {"checkpoint":str(path/"model.safetensors"),"sha256":digest(path/"model.safetensors"),"params":119477664}


def run_timesfm(rows, device):
    pkg = ROOT / "runtime-global-market-training/teacher-preflight/timesfm-pkg"
    sys.path.insert(0, str(pkg))
    from timesfm import ForecastConfig
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
    path = ROOT / "models/teachers/TimesFM-2.5-200M"
    model = TimesFM_2p5_200M_torch.from_pretrained(str(path), torch_compile=False, local_files_only=True)
    model.compile(ForecastConfig(max_context=128,max_horizon=128,normalize_inputs=True,
        per_core_batch_size=52,use_continuous_quantile_head=True,force_flip_invariance=True,
        infer_is_positive=False,fix_quantile_crossing=True))
    out=[]
    for row in rows:
        ids, rs = returns_for(row)
        if not ids:
            out.append((ids,np.empty((0,3),np.float32),np.empty(0,np.float32),np.empty((0,3),np.float32))); continue
        series=[r[-128:].astype(np.float32) for r in rs]
        point, qs = model.forecast(horizon=5,inputs=series)
        probs,values,quant=distill_quantiles(qs[:,4,1],point[:,4],qs[:,4,9],"TimesFM")
        out.append((ids,probs,values,quant))
    del model; torch.cuda.empty_cache(); gc.collect(); sys.path.remove(str(pkg))
    f=path/"model.safetensors"
    return out,{"checkpoint":str(f),"sha256":digest(f),"params":231289280}


def run_exaone(rows, device):
    pkg=ROOT/"runtime-global-market-training/teacher-preflight/exaone-pkg"
    sys.path.insert(0,str(pkg))
    from exaone_forecast.finance.forecaster import EXAONEFinanceForecaster
    ckpt=ROOT/"runtime-global-market-training/teacher-preflight/exaone-checkpoint"
    model=EXAONEFinanceForecaster(ckpt_dir=str(ckpt),device="cuda:0")
    out=[]
    for row in rows:
        ids,rs=returns_for(row)
        if not ids:
            out.append((ids,np.empty((0,3),np.float32),np.empty(0,np.float32),np.empty((0,3),np.float32)));continue
        q=model.predict([r[-512:].astype(np.float32) for r in rs],horizon=5,batch_size=52)
        levels=list(model.quantiles)
        probs,values,quant=distill_quantiles(q[:,levels.index(.1),4],q[:,levels.index(.5),4],q[:,levels.index(.9),4],"EXAONE")
        out.append((ids,probs,values,quant))
    del model;torch.cuda.empty_cache();gc.collect();sys.path.remove(str(pkg))
    f=ROOT/"models/teachers/EXAONE-Forecast-for-Finance-1.0/exaone-finance-1.0.safetensors"
    return out,{"checkpoint":str(f),"sha256":digest(f),"params":202319520}


def run_moirai(rows, device):
    pkg=ROOT/"runtime-global-market-training/teacher-preflight/uni2ts-pkg"
    sys.path.insert(0,str(pkg))
    pmod=types.ModuleType("uni2ts.model.moirai2");pmod.__path__=[str(pkg/"uni2ts/model/moirai2")]
    sys.modules["uni2ts.model.moirai2"]=pmod
    from uni2ts.model.moirai2.module import Moirai2Module
    path=ROOT/"models/teachers/Moirai-2.0-R-small"
    model=Moirai2Module.from_pretrained(str(path)).to(device).eval()
    levels=list(model.quantile_levels);i10,imid,i90=levels.index(.1),levels.index(.5),levels.index(.9)
    out=[]
    with torch.inference_mode():
      for row in rows:
        ids,rs=returns_for(row)
        if not ids:
            out.append((ids,np.empty((0,3),np.float32),np.empty(0,np.float32),np.empty((0,3),np.float32)));continue
        b=len(rs); context=np.full((b,512),np.nan,np.float32)
        for k,r in enumerate(rs): context[k,-min(512,len(r)):]=r[-512:]
        obs=torch.isfinite(torch.from_numpy(context)).to(device); target=torch.nan_to_num(torch.from_numpy(context),nan=0.).to(device)
        ct,patch=32,16; total=ct+1
        target=torch.cat((target.reshape(b,ct,patch),torch.zeros((b,1,patch),device=device)),1)
        obs=torch.cat((obs.reshape(b,ct,patch),torch.zeros((b,1,patch),dtype=torch.bool,device=device)),1)
        sid=torch.ones((b,total),dtype=torch.long,device=device);tid=torch.arange(total,device=device).expand(b,-1);vid=torch.zeros((b,total),dtype=torch.long,device=device)
        pm=torch.zeros((b,total),dtype=torch.bool,device=device);pm[:,-1]=True
        q=model(target,obs,sid,tid,vid,pm,training_mode=False).reshape(b,total,model.num_predict_token,model.num_quantiles,patch)[:,ct-1,0,:,:]
        probs,values,quant=distill_quantiles(q[:,i10,4].float().cpu().numpy(),q[:,imid,4].float().cpu().numpy(),q[:,i90,4].float().cpu().numpy(),"Moirai2")
        out.append((ids,probs,values,quant))
    del model;torch.cuda.empty_cache();gc.collect();sys.path.remove(str(pkg))
    f=path/"model.safetensors"
    return out,{"checkpoint":str(f),"sha256":digest(f),"params":11387208}


def run_timemoe(rows, device):
    import transformers.utils.import_utils as iu
    iu._torchvision_available=False
    d=ROOT/"models/teachers/TimeMoE-50M"
    pkg=types.ModuleType("timemoe_teacher");pkg.__path__=[str(d)];sys.modules["timemoe_teacher"]=pkg
    from timemoe_teacher.modeling_time_moe import TimeMoeForPrediction
    model=TimeMoeForPrediction.from_pretrained(str(d),torch_dtype=torch.bfloat16,local_files_only=True).eval().to(device)
    out=[]
    with torch.inference_mode():
      for row in rows:
        ids,rs=returns_for(row)
        if not ids:
            out.append((ids,np.empty((0,3),np.float32),np.empty(0,np.float32),np.empty((0,3),np.float32)));continue
        lengths=min(128,min(map(len,rs))); x=np.stack([r[-lengths:] for r in rs]); mu=np.nanmean(x,1);sd=np.nanstd(x,1).clip(1e-6)
        z=np.nan_to_num((x-mu[:,None])/sd[:,None],nan=0,posinf=0,neginf=0).astype(np.float32)[...,None]
        inp=torch.from_numpy(z).to(device=device,dtype=torch.bfloat16)
        pred=model(input_ids=inp,use_cache=False,max_horizon_length=1,return_dict=True).logits[:,-1,0].float().cpu().numpy()*sd+mu
        # The published checkpoint exposes a point forecast. Build a transparent
        # five-session interval using its point and trailing realized volatility.
        med=pred*5; width=1.2816*sd*np.sqrt(5)
        probs,values,quant=distill_quantiles(med-width,med,med+width,"TimeMoE")
        out.append((ids,probs,values,quant))
    del model;torch.cuda.empty_cache();gc.collect()
    f=d/"model.safetensors"
    return out,{"checkpoint":str(f),"sha256":digest(f),"params":113352192,"adapter":"one-step point forecast scaled to five sessions; interval from trailing volatility"}


RUNNERS={"chronos2":run_chronos,"timesfm":run_timesfm,"exaone":run_exaone,"moirai2":run_moirai,"timemoe":run_timemoe}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--student",type=Path,required=True)
    ap.add_argument("--runtime",type=Path,required=True)
    ap.add_argument("--examples",type=int,default=32)
    ap.add_argument("--steps",type=int,default=30)
    ap.add_argument("--seed",type=int,default=20260930)
    args=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA is required")
    if args.runtime.exists():raise FileExistsError(args.runtime)
    args.runtime.mkdir(parents=True)
    torch.cuda.init()
    device=torch.device("cuda:0")
    loader,manifest=temporal.get_loader(temporal.DEFAULT_DAILY,args.runtime/"daily-cache",temporal.NormalizedBarAdapter(86400),args.seed)
    rows=temporal.assemble_examples(loader,"train",args.examples,128,1,5,True,args.seed+3)
    val_rows=temporal.assemble_examples(loader,"validation",min(8,args.examples),128,1,5,True,args.seed+100)
    reports={}; prob_sum=[np.zeros((len(r["local_ids"]),3),np.float64) for r in rows]; value_sum=[np.zeros(len(r["local_ids"]),np.float64) for r in rows]; weight_sum=[np.zeros(len(r["local_ids"]),np.float64) for r in rows]
    cache_dir=args.runtime/"teacher-cache";cache_dir.mkdir()
    for name,runner in RUNNERS.items():
        torch.cuda.reset_peak_memory_stats(device);t0=time.perf_counter()
        predictions,meta=runner(rows,device)
        for i,(ids,probs,values,quant) in enumerate(predictions):
            for k,j in enumerate(ids):
                if j>=len(prob_sum[i]) or not np.isfinite(probs[k]).all() or not np.isfinite(values[k]):continue
                prob_sum[i][j]+=probs[k];value_sum[i][j]+=values[k];weight_sum[i][j]+=1
        cache={"teacher":name,"checkpoint":meta["checkpoint"],"sha256":meta["sha256"],"params":meta["params"],"train_only":True,"examples":len(rows),"predictions":sum(len(x[0]) for x in predictions),"elapsed_seconds":time.perf_counter()-t0,"peak_vram_bytes":int(torch.cuda.max_memory_allocated(device),),**{k:v for k,v in meta.items() if k not in ("checkpoint","sha256","params")}}
        np.savez_compressed(cache_dir/f"{name}.npz",ids=np.asarray([x[0] for x in predictions],dtype=object),probs=np.asarray([x[1] for x in predictions],dtype=object),values=np.asarray([x[2] for x in predictions],dtype=object),quantiles=np.asarray([x[3] for x in predictions],dtype=object))
        (cache_dir/f"{name}.json").write_text(json.dumps(cache,indent=2,allow_nan=False),encoding="utf-8")
        reports[name]=cache
        print(json.dumps({"event":"teacher_cache_complete",**cache},allow_nan=False),flush=True)
    for i,row in enumerate(rows):
        valid=weight_sum[i]>0
        row["teacher_valid"] &= valid
        row["teacher_probs"][valid]=(prob_sum[i][valid]/weight_sum[i][valid,None]).astype(np.float32)
        row["teacher_value"][valid]=(value_sum[i][valid]/weight_sum[i][valid]).astype(np.float32)
    source_sha=digest(args.student)
    candidate_path=args.runtime/"candidate.pt"
    payload,_daily_id_map=temporal.make_candidate(args.student,candidate_path,loader.symbol_map)
    model,payload=load_market_candidate_checkpoint(candidate_path,"cpu")
    model.backbone.float();model.context_policy.float();model.context_value.float();model.activation_checkpointing=True
    model.to(device).train()
    before=temporal.evaluate(model,{3:val_rows},{3:loader},device)
    optimizer=torch.optim.SGD(model.parameters(),lr=5e-5)
    payload.update({"source_student":str(args.student),"source_student_sha256":source_sha,"public_teacher_ensemble":list(reports),"public_teacher_reports":reports,"teacher_target_aggregation":"equal-weight mean of available teacher action distributions/value; forecast quantiles converted net of 22bp round trip","validation_before":before,"student_parameters":parameter_count(model.backbone),"test_split_accessed":False,"promotion":"not performed"})
    temporal.save_candidate(candidate_path,model,payload,optimizer,0)
    log=args.runtime/"distillation.jsonl";t0=time.perf_counter();rng=np.random.default_rng(args.seed+700)
    for step in range(1,args.steps+1):
        row=rows[int(rng.integers(len(rows)))];batch=temporal.make_batch(loader,row,3,device)
        x,sid,mid,aid,mask,ctx,ret,valid=batch_tensors(batch,device); valid=valid[0]&mask[0,-1]
        tv=torch.as_tensor(row["teacher_valid"],device=device)&valid
        if not valid.any():continue
        with torch.autocast("cuda",dtype=torch.bfloat16):logits,values=model(x,sid,mid,aid,mask,ctx,torch.tensor([3],device=device))
        p=torch.softmax(logits[0].float(),-1);_,pnl=utility(logits[0],ret[0],valid);ev=(p*pnl).sum(-1)
        loss=-ev[valid].mean()+.5*F.smooth_l1_loss(values[0,valid].float(),ev[valid].detach())
        if tv.any():
            tp=torch.as_tensor(row["teacher_probs"],device=device);tt=torch.as_tensor(row["teacher_value"],device=device)
            loss=loss+.15*F.kl_div(F.log_softmax(logits[0,tv].float()/2,-1),tp[tv],reduction="batchmean")*4+.05*F.smooth_l1_loss(values[0,tv].float(),tt[tv])
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):continue
        loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1.0,foreach=False)
        if not torch.isfinite(gn):optimizer.zero_grad(set_to_none=True);continue
        optimizer.step()
        if step%5==0 or step==args.steps:
            temporal.save_candidate(candidate_path,model,payload,optimizer,step)
            event={"event":"public_teacher_distill_progress","step":step,"loss":float(loss.detach()),"updates":step,"elapsed_seconds":time.perf_counter()-t0,"peak_vram_bytes":int(torch.cuda.max_memory_allocated(device))}
            with log.open("a",encoding="utf-8") as f:f.write(json.dumps(event,allow_nan=False)+"\n")
            print(json.dumps(event),flush=True)
    after=temporal.evaluate(model,{3:val_rows},{3:loader},device)
    result={"event":"public_teacher_ensemble_distillation_complete","candidate":str(candidate_path),"source_student":str(args.student),"source_student_sha256":source_sha,"teachers":list(reports),"steps":args.steps,"elapsed_seconds":time.perf_counter()-t0,"peak_vram_bytes":int(torch.cuda.max_memory_allocated(device)),"validation_before":before,"validation_after":after,"test_split_accessed":False,"promotion":"not performed"}
    payload["validation_after_public_teacher_ensemble"]=after;temporal.save_candidate(candidate_path,model,payload,optimizer,args.steps)
    (args.runtime/"latest-result.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    with log.open("a",encoding="utf-8") as f:f.write(json.dumps(result,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,allow_nan=False),flush=True);loader.close()

if __name__=="__main__":main()
