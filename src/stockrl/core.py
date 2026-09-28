from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

BASE_FEATURES = ["ret1", "ret5", "range", "volume_z", "rsi", "volatility"]
OPTIONAL_FEATURES = ["spread_bps", "book_imbalance", "trade_imbalance", "trade_intensity"]
VIDEO_FEATURES = ["video_chart_signal", "video_volume_signal"]
FEATURES = BASE_FEATURES + OPTIONAL_FEATURES + VIDEO_FEATURES
ACTION_NAMES = ("SELL", "HOLD", "BUY")


def device_for(name: str = "auto") -> torch.device:
    if name != "auto":
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable; install a MPS-enabled PyTorch build on Apple Silicon.")
        return torch.device(name)
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_market(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in lower]
    if missing: raise ValueError(f"CSV missing required columns: {missing}")
    df = df.rename(columns={lower[c]: c for c in required})
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for c in required[1:]: df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[required[1:]].isna().any().any() or (df.close <= 0).any():
        raise ValueError("OHLCV values must be numeric; close must be positive.")
    close = df.close
    ret1 = close.pct_change()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df["ret1"] = ret1
    df["ret5"] = close.pct_change(5)
    df["range"] = (df.high - df.low) / close
    logvol = np.log1p(df.volume)
    df["volume_z"] = (logvol-logvol.rolling(20,min_periods=2).mean()) / logvol.rolling(20,min_periods=2).std().replace(0,np.nan)
    df["rsi"] = (100-100/(1+rs)).fillna(50)/100
    df["volatility"] = ret1.rolling(20,min_periods=2).std()
    # Optional microstructure and video-derived columns. If bid/ask or size/trade
    # components exist, derive stable normalized signals without future inputs.
    if "bid" in lower and "ask" in lower:
        bid, ask = df[lower["bid"]].astype(float), df[lower["ask"]].astype(float)
        df["spread_bps"] = 10000 * (ask-bid) / ((ask+bid)/2).replace(0,np.nan)
        if "bid_size" in lower and "ask_size" in lower:
            bs, ass = df[lower["bid_size"]].astype(float), df[lower["ask_size"]].astype(float)
            df["book_imbalance"] = (bs-ass)/(bs+ass).replace(0,np.nan)
    if "buy_volume" in lower and "sell_volume" in lower:
        bv, sv = df[lower["buy_volume"]].astype(float), df[lower["sell_volume"]].astype(float)
        df["trade_imbalance"] = (bv-sv)/(bv+sv).replace(0,np.nan)
    if "trade_count" in lower:
        tc = df[lower["trade_count"]].astype(float)
        df["trade_intensity"] = tc/(tc.rolling(20,min_periods=2).mean()+1e-8)-1
    for col in OPTIONAL_FEATURES + VIDEO_FEATURES:
        if col in lower and col not in df:
            df[col] = pd.to_numeric(df[lower[col]],errors="coerce")
        if col not in df:
            df[col] = 0.0
    df[FEATURES] = df[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0).clip(-5,5)
    return df


@dataclass
class Config:
    window: int = 32
    fee: float = 0.001
    train_fraction: float = 0.70
    valid_fraction: float = 0.15
    seed: int = 7
    device: str = "auto"
    d_model: int = 64
    epochs: int = 15
    batch_size: int = 128
    lr: float = 0.001
    ppo_updates: int = 4


class Teacher:
    """Normalized adapter contract: three action logits and one scalar value."""
    def __init__(self, name: str, weight: float = 1.0): self.name, self.weight = name, float(weight)
    def predict(self, obs: np.ndarray) -> tuple[np.ndarray, float]: raise NotImplementedError


class HeuristicTeacher(Teacher):
    """Fallback policies for a runnable demo; these are not pretrained market models."""
    def __init__(self, name: str, kind: str, weight: float = 1.0): super().__init__(name,weight); self.kind=kind
    def predict(self, obs: np.ndarray) -> tuple[np.ndarray,float]:
        x=obs[-1]; momentum=float(x[0]+0.45*x[1])
        if self.kind=="mean_reversion": score=-momentum-0.15*(x[4]-0.5)
        elif self.kind=="trend": score=momentum+0.1*(x[4]-0.5)
        else: score=momentum-0.15*float(x[5])
        return np.array([-score*30,0.15-abs(score)*8,score*30],np.float32),float(np.clip(score*100,-1,1))


class SB3Teacher(Teacher):
    """Optional adapter for local Stable-Baselines3 PPO/A2C/DQN zip checkpoints."""
    def __init__(self,name:str,path:str,algorithm:str,weight:float=1.0):
        super().__init__(name,weight)
        try: from stable_baselines3 import A2C,DQN,PPO
        except ImportError as exc: raise RuntimeError("Install optional dependencies: pip install -e '.[teachers]'") from exc
        classes={"ppo":PPO,"a2c":A2C,"dqn":DQN}
        if algorithm.lower() not in classes: raise ValueError(f"Unsupported SB3 algorithm {algorithm}; choose ppo, a2c, or dqn")
        self.model=classes[algorithm.lower()].load(path,device="cpu")
    def predict(self,obs:np.ndarray)->tuple[np.ndarray,float]:
        action,_=self.model.predict(obs.reshape(-1).astype(np.float32),deterministic=True)
        logits=np.full(3,-3,dtype=np.float32); logits[int(np.clip(int(np.asarray(action).reshape(-1)[0]),0,2))]=3
        return logits,0.0


def make_teachers(spec:list[dict[str,Any]])->list[Teacher]:
    teachers=[]
    for t in spec:
        if t.get("type")=="public_maksim_ppo":
            from .public_teachers import SB3PPOTeacher
            teachers.append(SB3PPOTeacher(t["name"],t["path"],t.get("weight",1)))
        elif t.get("type")=="public_hf_ppo":
            from .public_teachers import HFPPOTeacher
            teachers.append(HFPPOTeacher(t["name"],t["path"],t["scaler_path"],t.get("weight",1)))
        elif t.get("type")=="public_deepbio_dqn":
            from .public_teachers import DeepBioDQNTeacher
            teachers.append(DeepBioDQNTeacher(t["name"],t["path"],t.get("weight",1)))
        elif t.get("type")=="public_recurrent_ppo":
            from .public_teachers import RecurrentPFETeacher
            teachers.append(RecurrentPFETeacher(t["name"],t["path"],t.get("weight",1)))
        elif t.get("type","heuristic")=="sb3": teachers.append(SB3Teacher(t["name"],t["path"],t["algorithm"],t.get("weight",1)))
        else: teachers.append(HeuristicTeacher(t["name"],t.get("kind","trend"),t.get("weight",1)))
    if not teachers or sum(max(t.weight,0) for t in teachers)<=0: raise ValueError("Provide teachers with positive weights")
    return teachers


def teacher_targets(df:pd.DataFrame,teachers:list[Teacher],window:int)->tuple[np.ndarray,np.ndarray]:
    weights=np.array([max(t.weight,0) for t in teachers],np.float64); weights/=weights.sum()
    feats=df[FEATURES].to_numpy(np.float32); predictions=[]
    for t in teachers:
        if hasattr(t,"predict_series"):
            tw=int(getattr(t,"required_window",window))
            logits,values=t.predict_series(df,tw)
            predictions.append((np.asarray(logits,np.float32),np.asarray(values,np.float32)))
        else:
            logits_rows=[]; value_rows=[]
            for i in range(len(df)):
                obs=feats[max(0,i-window+1):i+1]
                if len(obs)<window: obs=np.pad(obs,((window-len(obs),0),(0,0)))
                logits,val=t.predict(obs); logits_rows.append(logits); value_rows.append(val)
            predictions.append((np.asarray(logits_rows,np.float32),np.asarray(value_rows,np.float32)))
    probabilities=[]; value_matrix=[]
    for logits,values in predictions:
        logits=logits-logits.max(axis=1,keepdims=True); probs=np.exp(logits); probs/=probs.sum(axis=1,keepdims=True)
        probabilities.append(probs); value_matrix.append(values)
    ensemble=np.average(np.stack(probabilities),axis=0,weights=weights)
    values=np.average(np.stack(value_matrix),axis=0,weights=weights)
    return np.log(np.maximum(ensemble,1e-8)).astype(np.float32),values.astype(np.float32)


class TemporalActorCritic(nn.Module):
    def __init__(self,n_features:int=len(FEATURES),d_model:int=64,n_actions:int=3):
        super().__init__(); self.input=nn.Linear(n_features,d_model); self.gru=nn.GRU(d_model,d_model,batch_first=True)
        self.feature_names = FEATURES[:n_features]
        self.norm=nn.LayerNorm(d_model); self.policy=nn.Sequential(nn.Linear(d_model,d_model),nn.Tanh(),nn.Linear(d_model,n_actions))
        self.value=nn.Sequential(nn.Linear(d_model,d_model),nn.Tanh(),nn.Linear(d_model,1))
    def forward(self,x:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        z=torch.tanh(self.input(x)); z,_=self.gru(z); h=self.norm(z[:,-1])
        return self.policy(h),self.value(h).squeeze(-1)


def observations(df:pd.DataFrame,window:int,feature_names:list[str]|None=None)->np.ndarray:
    feature_names=feature_names or FEATURES
    feats=df[feature_names].to_numpy(np.float32); out=np.zeros((len(feats),window,len(feature_names)),np.float32)
    for i in range(len(feats)):
        part=feats[max(0,i-window+1):i+1]; out[i,-len(part):]=part
    return out


def split_indices(n:int,train_fraction:float,valid_fraction:float)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    a,b=int(n*train_fraction),int(n*(train_fraction+valid_fraction))
    if a<2 or b<=a or b>=n: raise ValueError("Need enough chronological rows for train, validation, and test")
    return np.arange(a),np.arange(a,b),np.arange(b)


def train_distill(model:TemporalActorCritic,x:np.ndarray,target_logits:np.ndarray,target_values:np.ndarray,
                  indices:np.ndarray,cfg:Config,device:torch.device,checkpoint_dir:Path)->None:
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr); model.train()
    for epoch in range(cfg.epochs):
        order=np.random.permutation(indices); losses=[]
        for start in range(0,len(order),cfg.batch_size):
            ix=order[start:start+cfg.batch_size]; logits,values=model(torch.as_tensor(x[ix],device=device))
            probs=torch.softmax(torch.as_tensor(target_logits[ix],device=device),dim=-1)
            tv=torch.as_tensor(target_values[ix],device=device)
            loss=-(probs*torch.log_softmax(logits,dim=-1)).sum(-1).mean()+0.2*nn.functional.mse_loss(values,tv)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach().cpu()))
        save_checkpoint(checkpoint_dir/"latest.pt",model,cfg); print(f"distill epoch {epoch+1}/{cfg.epochs} loss={np.mean(losses):.5f}")


def save_checkpoint(path:Path,model:TemporalActorCritic,cfg:Config)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"config":asdict(cfg),"features":model.feature_names,"actions":ACTION_NAMES},path)


def load_checkpoint(path:str|Path,device:torch.device)->tuple[TemporalActorCritic,Config]:
    ckpt=torch.load(path,map_location=device,weights_only=False)
    feature_names=ckpt.get("features",FEATURES)
    if any(name not in FEATURES for name in feature_names): raise ValueError("Checkpoint contains unknown feature names")
    cfg=Config(**ckpt["config"]); model=TemporalActorCritic(n_features=len(feature_names),d_model=cfg.d_model).to(device)
    model.feature_names=list(feature_names)
    model.load_state_dict(ckpt["state_dict"]); model.eval(); return model,cfg


def ppo_finetune(model:TemporalActorCritic,df:pd.DataFrame,x:np.ndarray,indices:np.ndarray,cfg:Config,
                 device:torch.device,checkpoint_dir:Path)->None:
    """Compact PPO on training rows, with fee-aware position-change reward."""
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr*0.2); close=df.close.to_numpy(np.float32); model.train()
    for epoch in range(cfg.ppo_updates):
        old_values=[]; actions=[]; rewards=[]; logps=[]; states=[]; position=0
        for i in indices[:-1]:
            state=torch.as_tensor(x[i:i+1],device=device)
            with torch.no_grad():
                logits,val=model(state); dist=torch.distributions.Categorical(logits=logits); a=dist.sample()
            new_position=int(a.item())-1; turnover=abs(new_position-position)
            reward=new_position*(close[i+1]/close[i]-1)-cfg.fee*turnover
            states.append(x[i]); old_values.append(float(val.item())); actions.append(int(a.item()))
            rewards.append(float(reward)); logps.append(float(dist.log_prob(a).item())); position=new_position
        rets=np.zeros(len(rewards),np.float32); running=0.0
        for j in range(len(rewards)-1,-1,-1): running=rewards[j]+0.99*running; rets[j]=running
        adv=rets-np.asarray(old_values,np.float32)
        if len(adv)>1: adv=(adv-adv.mean())/(adv.std()+1e-8)
        ts=torch.as_tensor(np.asarray(states),device=device); ta=torch.as_tensor(actions,device=device)
        tl=torch.as_tensor(logps,device=device); tad=torch.as_tensor(adv,device=device); tr=torch.as_tensor(rets,device=device)
        for _ in range(3):
            logits,values=model(ts); dist=torch.distributions.Categorical(logits=logits); ratio=torch.exp(dist.log_prob(ta)-tl)
            actor=-torch.minimum(ratio*tad,torch.clamp(ratio,.8,1.2)*tad).mean()
            loss=actor+.5*nn.functional.mse_loss(values,tr)-.01*dist.entropy().mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        save_checkpoint(checkpoint_dir/"latest.pt",model,cfg); print(f"ppo update {epoch+1}/{cfg.ppo_updates} mean_reward={np.mean(rewards):.6f}")


def evaluate(model:TemporalActorCritic,df:pd.DataFrame,x:np.ndarray,indices:np.ndarray,cfg:Config,device:torch.device)->dict[str,float]:
    x=x[:,:,:]
    close=df.close.to_numpy(np.float64); position=0; equity=peak=1.0; max_dd=0.0; trades=0; model.eval()
    with torch.no_grad():
        for i in indices[:-1]:
            logits,_=model(torch.as_tensor(x[i:i+1],device=device)); action=int(logits.argmax(-1).item())-1
            turn=abs(action-position); trades+=int(bool(turn)); equity*=max(1e-8,1+action*(close[i+1]/close[i]-1)-cfg.fee*turn)
            position=action; peak=max(peak,equity); max_dd=max(max_dd,(peak-equity)/peak)
    return {"return":float(equity-1),"max_drawdown":float(max_dd),"trades":float(trades)}


def infer(model:TemporalActorCritic,df:pd.DataFrame,window:int,device:torch.device)->dict[str,Any]:
    x=observations(df,window,model.feature_names)
    with torch.no_grad(): logits,value=model(torch.as_tensor(x[-1:],device=device)); probs=torch.softmax(logits,-1)[0].cpu().numpy()
    i=int(probs.argmax())
    return {"action":ACTION_NAMES[i],"probabilities":{n:float(probs[k]) for k,n in enumerate(ACTION_NAMES)},
            "value":float(value.item()),"date":str(df.date.iloc[-1].date())}
