"""Distill the six MacroHFT ETHUSDT Q experts on MacroHFT's train split only."""
from __future__ import annotations
import argparse,gc,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import distill_fincast_temporal as temporal
from bench_market_training_loader import batch_tensors,utility
from stockrl.global_transformer import parameter_count
from stockrl.market_training import load_market_candidate_checkpoint,CommonMarketTrainingLoader,NormalizedBarAdapter
from stockrl.research_ingest import MacroHFTSubagent,macrophft_features

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--student',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--examples',type=int,default=64);ap.add_argument('--steps',type=int,default=20);args=ap.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 if args.runtime.exists():raise FileExistsError(args.runtime)
 args.runtime.mkdir(parents=True);torch.cuda.init();device=torch.device('cuda:0')
 root=ROOT/'data/external_sources/macrophft';feed=root/'data/df_train.feather'
 frame=pd.read_feather(feed).sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
 frame['day']=pd.to_datetime(frame.timestamp,utc=True).dt.strftime('%Y-%m-%d')
 x1,x2=macrophft_features(frame,root/'feature_list')
 ix=frame.groupby('day',sort=True).tail(1).index.to_numpy();dates=frame.loc[ix,'day'].to_numpy();device_reports=[]
 qsum=np.zeros((len(ix),2,2),np.float64);t0=time.perf_counter()
 teacher_dir=root
 with torch.inference_mode():
  for regime in ('slope','vol'):
   for label in (1,2,3):
    name=f'{regime}_{label}_best_model.pkl';state=torch.load(teacher_dir/name,map_location='cpu',weights_only=True)
    model=MacroHFTSubagent().to(device).eval();model.load_state_dict(state,strict=True)
    for lo in range(0,len(ix),256):
     ids=ix[lo:lo+256];a=torch.from_numpy(x1[ids]).to(device);b=torch.from_numpy(x2[ids]).to(device)
     a2=a.repeat_interleave(2,0);b2=b.repeat_interleave(2,0);prev=torch.tensor([0,1],device=device).repeat(len(ids))
     q=model(a2,b2,prev).float().reshape(len(ids),2,2).cpu().numpy();qsum[lo:lo+len(ids)]+=q
    device_reports.append({'expert':name,'checkpoint_bytes':(teacher_dir/name).stat().st_size,'params':sum(p.numel() for p in model.parameters())})
    del model,state;torch.cuda.empty_cache();gc.collect()
 qbar=qsum/len(device_reports)
 # Each expert's two Q outputs mean flat/long. Map the two inventory states
 # into common BUY/HOLD/SELL probabilities and average all six experts.
 probs0=torch.softmax(torch.as_tensor(qbar[:,0],dtype=torch.float64),-1).numpy()
 probs1=torch.softmax(torch.as_tensor(qbar[:,1],dtype=torch.float64),-1).numpy()
 target_probs=np.stack((.5*probs1[:,0],.5*(probs0[:,0]+probs1[:,1]),.5*probs0[:,1]),-1).astype(np.float32)
 daily=root.parent.parent/'global_market_daily.csv';eth=frame.groupby('day',sort=True).agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'),bid=('bid1_price','last'),ask=('ask1_price','last'),bid_size=('bid1_size','last'),ask_size=('ask1_size','last')).reset_index().rename(columns={'day':'date'})
 eth['symbol']='ETHUSDT';eth['market']='CRYPTO';eth['asset_class']='CRYPTO';eth['buy_volume']=0.;eth['sell_volume']=0.;eth['trade_count']=0
 source=args.runtime/'global_market_daily_plus_eth.csv';pd.concat([pd.read_csv(daily),eth],ignore_index=True).sort_values(['date','symbol']).to_csv(source,index=False)
 cache=args.runtime/'shared-cache';loader=CommonMarketTrainingLoader(cache,[source],[NormalizedBarAdapter(86400)],chunksize=250000,seed=20261001,train_fraction=.70,validation_fraction=.15,context_stale_seconds=86400);manifest=loader.build(force=False)
 rows=temporal.assemble_examples(loader,'train',args.examples,128,1,5,True,20261004);val=temporal.assemble_examples(loader,'validation',min(8,args.examples),128,1,5,True,20261101)
 eth_ids=[i for i,s in enumerate(loader.symbol_names) if s.rsplit('|',1)[-1]=='ETHUSDT']
 if len(eth_ids)!=1:raise RuntimeError(f'Expected one unique ETHUSDT symbol, got {eth_ids}')
 eth_id=eth_ids[0];qmap={str(d):target_probs[i] for i,d in enumerate(dates)}
 labeled=0
 for row in rows:
  row['teacher_valid'][:]=False
  stamp=pd.Timestamp(int(loader.time_keys[int(row['end'])]),unit='s',tz='UTC').strftime('%Y-%m-%d')
  matches=np.flatnonzero(row['local_ids']==eth_id)
  if stamp in qmap and len(matches):
   j=int(matches[0]);row['teacher_probs'][j]=qmap[stamp];row['teacher_value'][j]=float(np.mean(np.max(qbar[dates==stamp],axis=(0,2)))) if np.any(dates==stamp) else 0.;row['teacher_valid'][j]=bool(row['valid'][j]);labeled+=1
 cache_dir=args.runtime/'teacher-cache';cache_dir.mkdir();np.savez_compressed(cache_dir/'macrophft_eth_train.npz',dates=dates,q_values=qbar,teacher_probs=target_probs)
 meta={'teacher':'MacroHFT','source':'ETHUSDT train feed only','source_sha256':__import__('hashlib').sha256(feed.read_bytes()).hexdigest(),'experts':device_reports,'daily_teacher_dates':len(dates),'student_examples':len(rows),'labeled_examples':labeled,'symbols':len(loader.symbol_names),'mapping':'six flat/long Q experts -> BUY/HOLD/SELL by averaging their two inventory-state distributions','test_accessed':False,'teacher_generation_seconds':time.perf_counter()-t0,'teacher_peak_vram_bytes':int(torch.cuda.max_memory_allocated(device))}
 (args.runtime/'teacher-cache.json').write_text(json.dumps(meta,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps({'event':'macrophft_teacher_cache_complete',**meta},allow_nan=False),flush=True)
 candidate=args.runtime/'candidate.pt';payload,_=temporal.make_candidate(args.student,candidate,loader.symbol_map);student,payload=load_market_candidate_checkpoint(candidate,'cpu');student.backbone.float();student.context_policy.float();student.context_value.float();student.activation_checkpointing=True;student.to(device).train();before=temporal.evaluate(student,{3:val},{3:loader},device);optimizer=torch.optim.SGD(student.parameters(),lr=5e-5);payload.update({'source_student':str(args.student),'macrophft_teacher':meta,'validation_before':before,'student_parameters':parameter_count(student.backbone),'test_split_accessed':False,'promotion':'not performed'})
 temporal.save_candidate(candidate,student,payload,optimizer,0);rng=np.random.default_rng(20261102);t1=time.perf_counter();updates=0
 for step in range(1,args.steps+1):
  row=rows[int(rng.integers(len(rows)))];batch=temporal.make_batch(loader,row,3,device);x,sid,mid,aid,mask,ctx,ret,valid=batch_tensors(batch,device);valid=valid[0]&mask[0,-1];tv=torch.as_tensor(row['teacher_valid'],device=device)&valid
  if not valid.any():continue
  with torch.autocast('cuda',dtype=torch.bfloat16):logits,values=student(x,sid,mid,aid,mask,ctx,torch.tensor([3],device=device))
  p=torch.softmax(logits[0].float(),-1);_,pnl=utility(logits[0],ret[0],valid);ev=(p*pnl).sum(-1);loss=-ev[valid].mean()+.5*F.smooth_l1_loss(values[0,valid].float(),ev[valid].detach())
  if tv.any():
   tp=torch.as_tensor(row['teacher_probs'],device=device);tvv=torch.as_tensor(row['teacher_value'],device=device);loss=loss+.15*F.kl_div(F.log_softmax(logits[0,tv].float()/2,-1),tp[tv],reduction='batchmean')*4+.05*F.smooth_l1_loss(values[0,tv].float(),tvv[tv])
  optimizer.zero_grad(set_to_none=True)
  if not torch.isfinite(loss):continue
  loss.backward();norm=nn.utils.clip_grad_norm_(student.parameters(),1.,foreach=False)
  if not torch.isfinite(norm):optimizer.zero_grad(set_to_none=True);continue
  optimizer.step();updates+=1
  if step%5==0 or step==args.steps:
   temporal.save_candidate(candidate,student,payload,optimizer,step);print(json.dumps({'event':'macrophft_distill_progress','step':step,'updates':updates,'loss':float(loss.detach()),'peak_vram_bytes':int(torch.cuda.max_memory_allocated(device))}),flush=True)
 after=temporal.evaluate(student,{3:val},{3:loader},device);result={'event':'macrophft_distillation_complete','candidate':str(candidate),'teachers':6,'train_rows_with_teacher_targets':labeled,'steps':args.steps,'updates':updates,'training_seconds':time.perf_counter()-t1,'peak_vram_bytes':int(torch.cuda.max_memory_allocated(device)),'validation_before':before,'validation_after':after,'test_split_accessed':False,'promotion':'not performed'}
 payload['validation_after_macrophft']=after;temporal.save_candidate(candidate,student,payload,optimizer,args.steps);(args.runtime/'latest-result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(result,indent=2,allow_nan=False),flush=True);loader.close()
if __name__=='__main__':main()
