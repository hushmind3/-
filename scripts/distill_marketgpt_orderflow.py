"""Distill MarketGPT's AAPL ITCH event-side distribution into an isolated student copy."""
from __future__ import annotations
import argparse,gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import distill_fincast_temporal as temporal
from bench_market_training_loader import utility
from stockrl.global_transformer import parameter_count,stable_id
from stockrl.market_training import load_market_candidate_checkpoint

DATA=ROOT/'data/external_sources/marketgpt/12302019_AAPL/12302019.NASDAQ_ITCH50_AAPL_message_proc.npy'
WEIGHT=ROOT/'models/teachers/MarketGPT-100M/unpacked/ckpt_finetune_AAPL_v3.pt'
SRC=ROOT/'models/teachers/MarketGPT-100M/source'

def make_bars(raw:np.ndarray, stop:int):
    # Columns are the official MarketGPT preprocessor's 18-field ITCH record.
    # Time is exchange-local seconds in columns 10/11; prices are cents in 4.
    sec=np.asarray(raw[:,10],np.int64);minute=sec//60-34200//60
    in_session=(minute>=0)&(minute<stop)&(sec>=34200)&(sec<57600)
    ix=np.flatnonzero(in_session);m=minute[ix].astype(np.int64)
    n=stop;open_=np.full(n,np.nan);high=open_.copy();low=open_.copy();close=open_.copy();volume=np.zeros(n);buy=np.zeros(n);sell=np.zeros(n);trades=np.zeros(n);last_message=np.full(n,-1,np.int64)
    price=np.asarray(raw[ix,4],np.float64)/100.;size=np.maximum(np.asarray(raw[ix,6],np.float64),0);side=np.asarray(raw[ix,3],np.int64);etype=np.asarray(raw[ix,2],np.int64)
    for t in np.unique(m):
        z=np.flatnonzero(m==t);good=np.isfinite(price[z])&(price[z]>0)
        if not good.any():continue
        z=z[good];pp=price[z];open_[t]=pp[0];high[t]=pp.max();low[t]=pp.min();close[t]=pp[-1];last_message[t]=ix[z[-1]]
        ss=size[z];sd=side[z];volume[t]=ss.sum();buy[t]=ss[sd==0].sum();sell[t]=ss[sd==1].sum();trades[t]=np.sum(etype[z]==2)
    # Carry the last known reference price over minute gaps, then trim to the
    # observable train/validation prefix. Test bars are never read downstream.
    for a in (open_,high,low,close):
        good=np.isfinite(a)
        if good.any():a[:]=np.interp(np.arange(n),np.flatnonzero(good),a[good])
    for t in range(n):
        if last_message[t]<0:last_message[t]=last_message[t-1] if t else 0
    high=np.where(np.isfinite(high),high,close);low=np.where(np.isfinite(low),low,close);open_=np.where(np.isfinite(open_),open_,close)
    return {'open':open_,'high':high,'low':low,'close':close,'volume':volume,'buy':buy,'sell':sell,'trades':trades,'last_message':last_message}


def build_features(b):
    c=b['close'];ret=np.zeros_like(c);ret[1:]=c[1:]/np.maximum(c[:-1],1e-12)-1
    f=np.zeros((len(c),17),np.float32);f[:,0]=ret
    for span,col in ((5,1),(20,2)):
        f[span:,col]=c[span:]/np.maximum(c[:-span],1e-12)-1
    f[:,3]=(b['high']-b['low'])/np.maximum(c,1e-12)
    lv=np.log1p(b['volume']);mu=np.array([lv[max(0,i-19):i+1].mean() for i in range(len(lv))]);sd=np.array([lv[max(0,i-19):i+1].std() for i in range(len(lv))]).clip(1e-5);f[:,4]=np.clip((lv-mu)/sd,-8,8)
    for i in range(len(c)):
        r=ret[max(0,i-13):i+1];g=np.maximum(r,0).mean();l=np.maximum(-r,0).mean();f[i,5]=g/max(g+l,1e-12);f[i,6]=ret[max(0,i-19):i+1].std()
    den=np.maximum(b['buy']+b['sell'],1e-12);imb=(b['buy']-b['sell'])/den;f[:,8]=imb;f[:,9]=imb;f[:,10]=np.log1p(b['trades'])
    ctx=np.zeros((len(c),16),np.float32)
    for i in range(len(c)):
        r=ret[max(0,i-19):i+1];v=b['volume'][max(0,i-19):i+1];ctx[i,0]=1.;ctx[i,1]=np.mean(b['trades'][max(0,i-19):i+1]>0);ctx[i,2]=float(ret[i]>0);ctx[i,3]=float(ret[i]<0);ctx[i,4]=ret[i]*100;ctx[i,5]=abs(ret[i])*100;ctx[i,6]=r.mean()*100;ctx[i,7]=np.median(r)*100;ctx[i,8]=r.std()*100;ctx[i,9]=np.log1p(b['volume'][i]);ctx[i,10]=imb[i];ctx[i,14]=1.;ctx[i,15]=f[i,6]*100
    return f,ctx,ret


def batch_at(features,context,ret,t,symbol_id,market_id,asset_id,device):
    ends=np.arange(t-127,t+1);valid=ends>=0
    x=torch.as_tensor(features[ends][None,:,None,:],dtype=torch.float16,device=device)
    ctx=torch.as_tensor(context[ends][None],dtype=torch.float32,device=device)
    mask=torch.as_tensor(valid[None,:,None],dtype=torch.bool,device=device)
    sid=torch.tensor([[symbol_id]],dtype=torch.long,device=device);mid=torch.tensor([[market_id]],dtype=torch.long,device=device);aid=torch.tensor([[asset_id]],dtype=torch.long,device=device)
    target=torch.tensor([[ret[t]]],dtype=torch.float32,device=device)
    return x,sid,mid,aid,mask,ctx,target


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--student',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--steps',type=int,default=30);args=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if args.runtime.exists():raise FileExistsError(args.runtime)
    args.runtime.mkdir(parents=True);torch.cuda.init();device=torch.device('cuda:0')
    # 390 one-minute bars in the regular session. Keep only the train and
    # validation prefix in memory; no test labels or test bars are used.
    n=390;train_end=int(n*.70);val_end=int(n*.85);observable_end=val_end
    raw=np.load(DATA,mmap_mode='r');bars=make_bars(raw,observable_end);features,context,returns=build_features(bars)
    price_valid=np.isfinite(bars['close'])&(bars['close']>0);future=np.zeros(n,np.float32);valid_target=np.zeros(n,bool)
    for t in range(127,observable_end-5):
        if price_valid[t] and price_valid[t+5]:future[t]=bars['close'][t+5]/bars['close'][t]-1;valid_target[t]=True
    train_ix=np.flatnonzero(valid_target & (np.arange(n)<train_end-5));val_ix=np.flatnonzero(valid_target&(np.arange(n)>=train_end)&(np.arange(n)<val_end-5))

    # MarketGPT teacher stage: load the official ~94M trainable-parameter
    # model on CUDA alone. Its checkpoint repeats giant causal masks; instantiate
    # a short 256-token adapter and reuse only learned weights.
    sys.path.insert(0,str(SRC));from equities.fast_model import Transformer,ModelArgs
    from equities.data_processing.itch_encoding import Vocab,encode_msgs
    vocab=Vocab();checkpoint=torch.load(WEIGHT,map_location='cpu',weights_only=False);model_args=dict(checkpoint['model_args']);model_args.update({'max_seq_len':256,'dropout':0.0})
    teacher=Transformer(ModelArgs(**model_args));state={k:v for k,v in checkpoint['model'].items() if not k.endswith('.attention.mask')};teacher.load_state_dict(state,strict=False);teacher=teacher.to(device,dtype=torch.bfloat16).eval();del checkpoint,state;gc.collect();torch.cuda.empty_cache()
    sid_tokens=np.asarray(vocab.ENCODING['side'][1][3:],np.int64);targets={};t0=time.perf_counter();last=np.asarray(bars['last_message'],np.int64)
    with torch.inference_mode():
      for t in train_ix:
        end=int(last[t]);begin=max(0,end-8);events=np.asarray(raw[begin:end+1],dtype=np.int64)
        if len(events)<2:continue
        toks=encode_msgs(events,vocab.ENCODING).astype(np.int64).reshape(-1)[-8*24:]
        seq=torch.cat((torch.tensor([[vocab.SINK_TOK]],device=device),torch.as_tensor(toks[None],device=device)),dim=1)
        # Autoregressively predict next event ticker and type, then obtain the
        # side distribution. Raw side 0 is bid/buy, raw side 1 is ask/sell.
        for _ in range(2):seq=torch.cat((seq,teacher(seq)[:,-1].argmax(-1,keepdim=True)),dim=1)
        logits=teacher(seq)[:,-1].float();ps=torch.softmax(logits[:,torch.as_tensor(sid_tokens,device=device)],-1)[0].cpu().numpy()
        edge=float(ps[0]-ps[1]);pb=max(edge,0.);psell=max(-edge,0.);phold=max(0.,1.-abs(edge));dist=np.asarray([psell,phold,pb],np.float32);dist/=max(float(dist.sum()),1e-12)
        vol=float(np.std(returns[max(0,t-19):t+1])*np.sqrt(5));targets[int(t)]=(dist,float(edge*vol))
    teacher_meta={'teacher':'MarketGPT AAPL fine-tune','repo':'aaronwheeler/MarketGPT-100m','params':sum(p.numel() for p in teacher.parameters()),'checkpoint_zip_bytes':(ROOT/'models/teachers/MarketGPT-100M/ckpt_finetune_AAPL_v3.zip').stat().st_size,'checkpoint_pt_bytes':WEIGHT.stat().st_size,'dtype':'FP32 checkpoint; BF16 CUDA inference','context_events':8,'teacher_samples':len(targets),'teacher_peak_vram_bytes':int(torch.cuda.max_memory_allocated(device)),'teacher_seconds':time.perf_counter()-t0,'mapping':'P(next ITCH side) -> SELL/HOLD/BUY order-flow edge'}
    np.savez_compressed(args.runtime/'marketgpt_teacher_train.npz',times=np.asarray(sorted(targets)),probabilities=np.stack([targets[k][0] for k in sorted(targets)]),values=np.asarray([targets[k][1] for k in sorted(targets)],np.float32));(args.runtime/'teacher-cache.json').write_text(json.dumps(teacher_meta,indent=2),encoding='utf-8')
    del teacher;torch.cuda.empty_cache();gc.collect();sys.path.remove(str(SRC));print(json.dumps({'event':'marketgpt_teacher_cache_complete',**teacher_meta}),flush=True)

    # Student stage starts from the requested existing candidate and remains in
    # an isolated runtime. Symbol/market/asset ids match its canonical mapping.
    payload=torch.load(args.student,map_location='cpu',weights_only=False);smap=payload['symbol_map'];s_matches=[(k,v) for k,v in smap.items() if k.rsplit('|',1)[-1]=='AAPL']
    if len(s_matches)!=1:raise RuntimeError(f'Expected one AAPL id, got {s_matches}')
    symbol_name,symbol_id=s_matches[0];market_id=stable_id(symbol_name.split('|',1)[0],64);asset_id=stable_id(symbol_name.split('|',2)[1],32)
    del payload;model,payload=load_market_candidate_checkpoint(args.student,'cpu');model.backbone.float();model.context_policy.float();model.context_value.float();model.activation_checkpointing=True;model.to(device).train()
    optimizer=torch.optim.SGD(model.parameters(),lr=5e-5);candidate=args.runtime/'candidate.pt';temporal.save_candidate(candidate,model,payload,optimizer,0)
    def evaluate(ix):
      model.eval();total=0.;count=0;acts=np.zeros(3,np.int64)
      with torch.inference_mode():
       for t in ix:
        if not valid_target[t]:continue
        x,sid,mid,aid,mask,ctx,target=batch_at(features,context,future,t,symbol_id,market_id,asset_id,device)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=model(x,sid,mid,aid,mask,ctx,torch.tensor([1],device=device))
        _,pnl=utility(logits[0],target[0],mask[0,-1]);action=logits[0].float().argmax(-1);total+=float(pnl[0,action[0]].cpu());count+=1;acts[int(action[0])]+=1
      model.train();return {'net_return_sum':total,'valid_decisions':count,'mean_net_return':total/max(count,1),'actions':{'SELL':int(acts[0]),'HOLD':int(acts[1]),'BUY':int(acts[2])}}
    before=evaluate(val_ix);rng=np.random.default_rng(20260927);t1=time.perf_counter();updates=0
    for step in range(1,args.steps+1):
      t=int(rng.choice(train_ix));x,sid,mid,aid,mask,ctx,target=batch_at(features,context,future,t,symbol_id,market_id,asset_id,device)
      with torch.autocast('cuda',dtype=torch.bfloat16):logits,values=model(x,sid,mid,aid,mask,ctx,torch.tensor([1],device=device))
      valid=mask[0,-1];p=torch.softmax(logits[0].float(),-1);_,pnl=utility(logits[0],target[0],valid);ev=(p*pnl).sum(-1);loss=-ev[valid].mean()+.5*F.smooth_l1_loss(values[0,valid].float(),ev[valid].detach())
      if t in targets:
       tp=torch.as_tensor(targets[t][0],device=device);tv=torch.tensor([targets[t][1]],device=device);loss=loss+.15*F.kl_div(F.log_softmax(logits[0].float()/2,-1),tp[None],reduction='batchmean')*4+.05*F.smooth_l1_loss(values[0].float(),tv)
      optimizer.zero_grad(set_to_none=True)
      if not torch.isfinite(loss):continue
      loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False)
      if not torch.isfinite(gn):optimizer.zero_grad(set_to_none=True);continue
      optimizer.step();updates+=1
      if step%5==0 or step==args.steps:
       temporal.save_candidate(candidate,model,payload,optimizer,step);print(json.dumps({'event':'marketgpt_distill_progress','step':step,'updates':updates,'loss':float(loss.detach()),'peak_vram_bytes':int(torch.cuda.max_memory_allocated(device))}),flush=True)
    after=evaluate(val_ix);result={'event':'marketgpt_distillation_complete','candidate':str(candidate),'teacher':teacher_meta,'student_parameters':parameter_count(model.backbone),'train_examples':len(train_ix),'teacher_labeled_examples':len(set(train_ix)&set(targets)),'validation_before':before,'validation_after':after,'steps':args.steps,'updates':updates,'student_peak_vram_bytes':int(torch.cuda.max_memory_allocated(device)),'test_split_accessed':False,'promotion':'not performed'}
    payload.update({'marketgpt_distillation':result,'validation_after_marketgpt':after});temporal.save_candidate(candidate,model,payload,optimizer,args.steps);(args.runtime/'latest-result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(result,indent=2,allow_nan=False),flush=True)
if __name__=='__main__':main()
