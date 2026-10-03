"""Offline representation distillation from pretrained MOMENT into a student copy."""
from __future__ import annotations
import argparse,gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
import distill_fincast_temporal as temporal
from bench_market_training_loader import batch_tensors,utility
from stockrl.global_transformer import parameter_count
from stockrl.market_training import load_market_candidate_checkpoint

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--student',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--examples',type=int,default=16);ap.add_argument('--steps',type=int,default=20);args=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if args.runtime.exists():raise FileExistsError(args.runtime)
    args.runtime.mkdir(parents=True);torch.cuda.init();device=torch.device('cuda:0')
    pkg=ROOT/'runtime-global-market-training/teacher-preflight/moment-pkg';sys.path.insert(0,str(pkg))
    import transformers.utils.import_utils as iu;iu._torchvision_available=False
    from momentfm import MOMENTPipeline
    loader,_=temporal.get_loader(temporal.DEFAULT_DAILY,args.runtime/'daily-cache',temporal.NormalizedBarAdapter(86400),20260930)
    rows=temporal.assemble_examples(loader,'train',args.examples,128,1,5,True,20260933)
    val=temporal.assemble_examples(loader,'validation',min(8,args.examples),128,1,5,True,20261030)
    model=MOMENTPipeline.from_pretrained(str(ROOT/'models/teachers/MOMENT-1-Large'),model_kwargs={'task_name':'embedding'})
    model.init();model.to(device).eval();torch.cuda.reset_peak_memory_stats(device);t0=time.perf_counter()
    cached=[]
    with torch.inference_mode():
      for row in rows:
        b=len(row['local_ids']);x=np.zeros((b,1,128),np.float32);mask=np.zeros((b,128),np.float32)
        for j,series,_ in row['series']:
          close=np.asarray(series,np.float32);r=np.diff(np.log(close)).astype(np.float32);n=min(128,len(r));x[j,0,-n:]=r[-n:];mask[j,-n:]=1
        out=model(x_enc=torch.from_numpy(x).to(device),input_mask=torch.from_numpy(mask).to(device))
        z=F.normalize(out.embeddings.float(),dim=-1).cpu().numpy()
        cached.append(z)
    cache=args.runtime/'moment_embeddings_train.npz';np.savez_compressed(cache,embeddings=np.stack(cached))
    meta={'teacher':'AutonLab/MOMENT-1-large','task':'pretrained encoder representation (forecast head is randomly initialized and is not used)','teacher_parameters':341240320,'checkpoint_bytes':(ROOT/'models/teachers/MOMENT-1-Large/model.safetensors').stat().st_size,'train_examples':len(rows),'representation_shape':list(cached[0].shape),'teacher_peak_vram_bytes':int(torch.cuda.max_memory_allocated(device)),'teacher_seconds':time.perf_counter()-t0,'split':'train only'}
    (args.runtime/'moment-cache.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');del model,out;torch.cuda.empty_cache();gc.collect();sys.path.remove(str(pkg))
    candidate=args.runtime/'candidate.pt';payload,_=temporal.make_candidate(args.student,candidate,loader.symbol_map)
    student,payload=load_market_candidate_checkpoint(candidate,'cpu');student.backbone.float();student.context_policy.float();student.context_value.float();student.activation_checkpointing=True
    student.to(device).train();projection=nn.Linear(student.backbone.cfg.d_model,1024,bias=False).to(device)
    optimizer=torch.optim.SGD(list(student.parameters())+list(projection.parameters()),lr=5e-5)
    before=temporal.evaluate(student,{3:val},{3:loader},device)
    captured={};handle=student.backbone.final_norm.register_forward_hook(lambda _m,_i,o:captured.update(hidden=o))
    rng=np.random.default_rng(20261031);t1=time.perf_counter();losses=[]
    for step in range(1,args.steps+1):
      ix=int(rng.integers(len(rows)));row=rows[ix];batch=temporal.make_batch(loader,row,3,device);x,sid,mid,aid,mask,ctx,ret,tv=batch_tensors(batch,device)
      # batch_tensors returns valid symbol mask; ensure only labels with actual future returns.
      valid=torch.as_tensor(row['valid'],device=device)&mask[0,-1]
      with torch.autocast('cuda',dtype=torch.bfloat16):logits,values=student(x,sid,mid,aid,mask,ctx,torch.tensor([3],device=device))
      target=torch.as_tensor(cached[ix],device=device,dtype=torch.float32);target=F.normalize(target,dim=-1)
      h=captured['hidden'][0].float();z=F.normalize(projection(h),dim=-1)
      repr_loss=F.mse_loss(z[valid],target[valid]) if valid.any() else h.sum()*0
      probs=torch.softmax(logits[0].float(),-1);_,pnl=utility(logits[0],ret[0],valid);ev=(probs*pnl).sum(-1)
      pnl_loss=-ev[valid].mean()+.5*F.smooth_l1_loss(values[0,valid].float(),ev[valid].detach()) if valid.any() else h.sum()*0
      loss=pnl_loss+.05*repr_loss
      optimizer.zero_grad(set_to_none=True)
      if not torch.isfinite(loss):continue
      loss.backward();gn=nn.utils.clip_grad_norm_(list(student.parameters())+list(projection.parameters()),1.0,foreach=False)
      if not torch.isfinite(gn):optimizer.zero_grad(set_to_none=True);continue
      optimizer.step();losses.append(float(loss.detach()))
      if step%5==0 or step==args.steps:
        temporal.save_candidate(candidate,student,payload,optimizer,step)
        print(json.dumps({'event':'moment_repr_distill_progress','step':step,'loss':losses[-1],'repr_loss':float(repr_loss.detach()),'peak_vram_bytes':int(torch.cuda.max_memory_allocated(device))}),flush=True)
    handle.remove();after=temporal.evaluate(student,{3:val},{3:loader},device)
    result={**meta,'event':'moment_representation_distillation_complete','candidate':str(candidate),'source_student':str(args.student),'steps':args.steps,'updates':len(losses),'training_seconds':time.perf_counter()-t1,'peak_student_vram_bytes':int(torch.cuda.max_memory_allocated(device)),'validation_before':before,'validation_after':after,'test_split_accessed':False,'promotion':'not performed'}
    payload.update({'moment_representation_distillation':result,'validation_after_moment':after});temporal.save_candidate(candidate,student,payload,optimizer,args.steps)
    (args.runtime/'latest-result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(result,indent=2,allow_nan=False),flush=True);loader.close()
if __name__=='__main__':main()
