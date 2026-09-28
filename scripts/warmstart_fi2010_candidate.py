"""One audited CUDA warm-start update from real FI-2010 train examples.

This creates an isolated candidate and never writes to a verified runtime or
champion. FI direction labels are a temporary teacher signal; they are not
used for PnL promotion, which requires an independent net-PnL validation.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from stockrl.global_online import load_model, save_model
from stockrl.global_transformer import stable_id, parameter_count
from stockrl.research_ingest import fi2010_book_to_features, fi2010_file_to_arrays


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external_sources/fi2010/selected"
BASE = ROOT / "runtime-global-cuda-final/champion.pt"
OUT = ROOT / "runtime-global-research-pretrain/candidate.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 8


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""): h.update(block)
    return h.hexdigest()


def load_features(path: Path):
    raw, labels = fi2010_file_to_arrays(path)
    x = np.stack([fi2010_book_to_features(r) for r in raw])
    # Derive only past-and-current microstructure deltas; labels are never
    # copied into observations. Dataset quotes are z-scored, not dollar prices.
    mid = (raw[:,0] + raw[:,20]) * 0.5
    x[:,0] = np.clip(np.r_[0.0, np.diff(mid)],-10,10)
    x[:,1] = np.clip(mid - np.r_[np.zeros(5),mid[:-5]],-10,10)
    x[:,2] = np.clip(mid - np.r_[np.zeros(20),mid[:-20]],-10,10)
    x[:,4] = np.clip(np.mean(np.abs(raw[:,10:20]),axis=1)+np.mean(np.abs(raw[:,30:40]),axis=1),0,10)
    return np.nan_to_num(x), labels


def pack(model, x: np.ndarray, end: int):
    seq=np.zeros((T,1,x.shape[1]),np.float32)
    lo=max(0,end-T+1); part=x[lo:end+1]
    seq[-len(part):,0]=part
    sid=torch.tensor([[stable_id("FI2010",model.cfg.max_symbols)]],device=DEVICE)
    mid=torch.zeros((1,1),dtype=torch.long,device=DEVICE)
    aid=torch.zeros((1,1),dtype=torch.long,device=DEVICE)
    mask=torch.ones((1,T,1),dtype=torch.bool,device=DEVICE)
    feat=torch.from_numpy(seq[None]).to(DEVICE,dtype=torch.float16 if DEVICE.type=="cuda" else torch.float32)
    return feat,sid,mid,aid,mask


@torch.inference_mode()
def score(model,x,labels,indices):
    model.eval(); correct=0; loss_sum=0.0; n=0
    for ix in indices:
        f,s,m,a,k=pack(model,x,int(ix))
        logits,_=model(f,s,m,a,k)
        pred=logits[0,0].float()
        target=int(labels[ix,0])-1
        correct+=int(pred.argmax().item()==target)
        loss_sum+=float(F.cross_entropy(pred[None],torch.tensor([target],device=DEVICE)))
        n+=1
    return {"n":n,"accuracy":correct/max(n,1),"cross_entropy":loss_sum/max(n,1)}


def main():
    if not BASE.exists(): raise FileNotFoundError(f"baseline champion absent: {BASE}")
    if DEVICE.type!="cuda": raise RuntimeError("RTX 3070 CUDA is present but CUDA was not selected")
    train_path=next((DATA/"NoAuction_Zscore_Training").glob("Train_Dst_NoAuction_ZScore_CF_1.txt"))
    test_path=next((DATA/"NoAuction_Zscore_Testing").glob("Test_Dst_NoAuction_ZScore_CF_1.txt"))
    train_x,train_y=load_features(train_path); test_x,test_y=load_features(test_path)
    model,cfg=load_model(BASE,DEVICE)
    nparams=parameter_count(model)
    test_ix=np.linspace(T-1,len(test_x)-1,min(24,max(1,len(test_x)-T)),dtype=int)
    before=score(model,test_x,test_y,test_ix)
    train_ix=min(257,len(train_x)-1)
    train_f, sid, mid, aid, mask=pack(model,train_x,train_ix)
    action=torch.tensor([int(train_y[train_ix,0])-1],device=DEVICE,dtype=torch.long)
    value_target=torch.tensor([float(action.item()-1)],device=DEVICE,dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6,weight_decay=0.0,eps=1e-4)
    torch.cuda.reset_peak_memory_stats(DEVICE); torch.cuda.synchronize(DEVICE); started=time.perf_counter()
    model.train(); logits,values=model(train_f,sid,mid,aid,mask)
    policy=F.cross_entropy(logits[0,0].float().unsqueeze(0),action)
    critic=F.smooth_l1_loss(values[0,0].float().reshape(1),value_target)
    loss=policy+0.25*critic
    optimizer.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),0.1); optimizer.step()
    torch.cuda.synchronize(DEVICE); elapsed=time.perf_counter()-started
    after=score(model,test_x,test_y,test_ix)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    save_model(OUT,model,cfg,step=1)
    metrics={"source":"FI-2010 NoAuction z-score raw 40 features; label horizon 10 events",
             "train_file":train_path.name,"heldout_file":test_path.name,
             "teacher_label_map":{"1":"SELL","2":"HOLD","3":"BUY"},
             "device":torch.cuda.get_device_name(DEVICE),"parameter_count":nparams,
             "train_events":len(train_x),"heldout_events":len(test_x),
             "one_gradient_update_seconds":round(elapsed,4),"loss":float(loss.detach().cpu()),
             "heldout_before":before,"heldout_after":after,
             "candidate_checkpoint":str(OUT),"candidate_sha256":sha256(OUT),
             "baseline_champion_sha256":sha256(BASE),
             "peak_cuda_allocated_bytes":int(torch.cuda.max_memory_allocated(DEVICE)),
             "promotion":"NOT evaluated/promoted; directional accuracy is not net PnL"}
    (OUT.parent/"fi2010_warmstart_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf8")
    print(json.dumps(metrics,indent=2))


if __name__=="__main__": main()
