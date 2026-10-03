"""Apply one isolated teacher-distillation update from the saved MacroHFT replay."""
from pathlib import Path
import hashlib,json,time
import torch
import torch.nn.functional as F

from stockrl.global_online import GlobalReplayBuffer,load_model,save_model
from stockrl.global_transformer import parameter_count

ROOT=Path(__file__).resolve().parents[2]
CKPT=ROOT/"runtime-global-research-pretrain/candidate.pt"
REPLAY=ROOT/"runtime-global-research-pretrain/teacher_replay.pt"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    if DEVICE.type!="cuda":raise RuntimeError("Expected RTX 3070 CUDA")
    model,cfg=load_model(CKPT,DEVICE)
    replay=GlobalReplayBuffer(capacity=20_000);replay.load(REPLAY)
    teacher=replay.sample_source("teacher:MacroHFT")
    if teacher is None:raise RuntimeError("No MacroHFT replay examples found")
    x=torch.as_tensor(teacher.features[None],device=DEVICE,dtype=torch.float16)
    sid=torch.as_tensor(teacher.symbol_ids[None],device=DEVICE,dtype=torch.long)
    mid=torch.as_tensor(teacher.market_ids[None],device=DEVICE,dtype=torch.long)
    aid=torch.as_tensor(teacher.asset_ids[None],device=DEVICE,dtype=torch.long)
    mask=torch.as_tensor(teacher.valid_mask[None],device=DEVICE,dtype=torch.bool)
    target=torch.tensor([teacher.action],device=DEVICE,dtype=torch.long)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-6,weight_decay=0.0,eps=1e-4)
    model.eval()
    with torch.inference_mode():
        before=F.cross_entropy(model(x,sid,mid,aid,mask)[0][0,teacher.symbol_index].float()[None],target)
    torch.cuda.reset_peak_memory_stats(DEVICE);torch.cuda.synchronize();start=time.perf_counter()
    model.train();logits,_=model(x,sid,mid,aid,mask)
    loss=F.cross_entropy(logits[0,teacher.symbol_index].float()[None],target)
    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.1);opt.step()
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    model.eval()
    with torch.inference_mode():
        after=F.cross_entropy(model(x,sid,mid,aid,mask)[0][0,teacher.symbol_index].float()[None],target)
    save_model(CKPT,model,cfg,step=2)
    metrics={"source":"MacroHFT pretrained ETH long/flat policy outputs",
      "replay_source":teacher.source,"teacher_action":teacher.action,
      "input_shape":list(teacher.features.shape),"device":torch.cuda.get_device_name(DEVICE),
      "parameter_count":parameter_count(model),"update_seconds":round(elapsed,4),
      "sample_loss_before":float(before.cpu()),"sample_loss_after":float(after.cpu()),
      "candidate_path":str(CKPT),"candidate_bytes":CKPT.stat().st_size,
      "candidate_sha256":hashlib.sha256(CKPT.read_bytes()).hexdigest(),
      "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(DEVICE),
      "promotion":"not evaluated/promoted; training-sample imitation loss is not PnL evidence"}
    out=ROOT/"runtime-global-research-pretrain/macrophft_warmstart_metrics.json"
    out.write_text(json.dumps(metrics,indent=2),encoding="utf8")
    print(json.dumps(metrics,indent=2))
if __name__=="__main__":main()
