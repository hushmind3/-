"""Adapters for downloaded public trading checkpoints with source-specific observations."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

from .core import ACTION_NAMES, Teacher


def _logits_for(action: int, value: float = 0.0) -> tuple[np.ndarray, float]:
    # Public models expose a discrete decision, not calibrated 3-way probabilities.
    logits = np.full(3, -5.0, dtype=np.float32)
    logits[int(action)] = 5.0
    return logits, float(value)


def maksim_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df.close.astype(float)
    f = pd.DataFrame(index=df.index)
    f["log_return"] = np.log(c / c.shift(1))
    f["sma20"] = c.rolling(20).mean()
    f["sma50"] = c.rolling(50).mean()
    ema12 = c.ewm(span=12, adjust=False).mean(); ema26 = c.ewm(span=26, adjust=False).mean()
    f["macd"] = ema12 - ema26
    delta = c.diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = -delta.clip(upper=0).rolling(14).mean()
    f["rsi14"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    f["volume_change"] = df.volume.astype(float).pct_change()
    return f.replace([np.inf, -np.inf], np.nan).fillna(0)


class SB3PPOTeacher(Teacher):
    """maksimprivalov PPO: 30x6 flat daily indicators, Discrete(3) SELL/HOLD/BUY."""
    required_window = 15
    def __init__(self, name: str, path: str, weight: float = 1.0):
        super().__init__(name, weight)
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")
        if tuple(self.model.observation_space.shape) != (90,):
            raise ValueError(f"Unexpected PPO observation space: {self.model.observation_space}")
        if not hasattr(self.model.action_space, "n") or self.model.action_space.n != 3:
            raise ValueError(f"Unexpected PPO action space: {self.model.action_space}")

    def predict_series(self, df: pd.DataFrame, window: int) -> tuple[np.ndarray, np.ndarray]:
        features = maksim_features(df).to_numpy(np.float32)
        window = 15
        if window * features.shape[1] != int(np.prod(self.model.observation_space.shape)):
            raise ValueError("Maksim PPO requires --teacher-window 15 to match its 90-float observation")
        outs=[]; vals=[]
        for i in range(len(df)):
            obs=np.zeros((window,6),np.float32); part=features[max(0,i-window+1):i+1]
            obs[-len(part):]=part
            action,_=self.model.predict(obs.reshape(-1),deterministic=True)
            idx=int(np.asarray(action).reshape(-1)[0])  # source: 0 SELL, 1 HOLD, 2 BUY
            with torch.no_grad():
                value=self.model.policy.predict_values(torch.as_tensor(obs.reshape(1,-1),dtype=torch.float32)).item()
            logits,val=_logits_for(idx,value); outs.append(logits); vals.append(val)
        return np.asarray(outs,np.float32),np.asarray(vals,np.float32)


class HFPPOTeacher(Teacher):
    """Adilbai HF PPO: 60x50 standardized features + 8 portfolio values; Box action."""
    required_window = 60
    def __init__(self, name: str, path: str, scaler_path: str, weight: float = 1.0):
        super().__init__(name,weight)
        from stable_baselines3 import PPO
        self.model=PPO.load(path,device="cpu")
        self.scaler=pickle.load(open(scaler_path,"rb"))
        if tuple(self.model.observation_space.shape)!=(3008,) or self.scaler.n_features_in_!=50:
            raise ValueError("Hugging Face checkpoint/scaler schema mismatch")
        self.window=60

    @staticmethod
    def _features(df:pd.DataFrame)->pd.DataFrame:
        d=pd.DataFrame({"Date":df.date,"Open":df.open,"High":df.high,"Low":df.low,"Close":df.close,"Volume":df.volume})
        c=d.Close; d["SMA_5"]=c.rolling(5).mean(); d["SMA_10"]=c.rolling(10).mean()
        d["SMA_20"]=c.rolling(20).mean(); d["SMA_50"]=c.rolling(50).mean()
        d["EMA_12"]=c.ewm(span=12).mean(); d["EMA_26"]=c.ewm(span=26).mean()
        d["MACD"]=d.EMA_12-d.EMA_26; d["MACD_Signal"]=d.MACD.ewm(span=9).mean()
        d["MACD_Histogram"]=d.MACD-d.MACD_Signal
        delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=-delta.clip(upper=0).rolling(14).mean()
        d["RSI"]=100-100/(1+gain/loss.replace(0,np.nan))
        d["BB_Middle"]=c.rolling(20).mean(); std=c.rolling(20).std()
        d["BB_Upper"]=d.BB_Middle+2*std; d["BB_Lower"]=d.BB_Middle-2*std
        d["BB_Width"]=d.BB_Upper-d.BB_Lower; d["BB_Position"]=(c-d.BB_Lower)/d.BB_Width.replace(0,np.nan)
        d["Volatility"]=c.rolling(20).std(); d["Price_Change"]=c.pct_change(); d["Price_Change_5d"]=c.pct_change(5)
        d["High_Low_Ratio"]=d.High/d.Low; d["Open_Close_Ratio"]=d.Open/d.Close
        d["Volume_SMA"]=d.Volume.rolling(20).mean(); d["Volume_Ratio"]=d.Volume/d.Volume_SMA
        for col in ["Close","Volume","Price_Change","RSI","MACD","Volatility"]:
            for lag in [1,2,3,5,10]: d[f"{col}_lag_{lag}"]=d[col].shift(lag)
        return d

    def predict_series(self,df:pd.DataFrame,window:int)->tuple[np.ndarray,np.ndarray]:
        if window != self.window: raise ValueError("HF PPO requires a 60-row lookback")
        d=self._features(df)
        names=list(self.scaler.feature_names_in_)
        absent=[n for n in names if n not in d.columns]
        if absent: raise ValueError(f"HF PPO missing feature columns: {absent}")
        raw=d[names].replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(0).to_numpy(np.float64)
        scaled=self.scaler.transform(raw).astype(np.float32)
        # Model's 8 portfolio fields are defined in enviromentcreator.py.
        balance=10000.0; shares=0.0; net=balance; trades=0; costs=0.0; peak=balance; daily_returns=[]
        outs=[]; vals=[]
        for i in range(len(df)):
            start=max(0,i-self.window+1); seq=scaled[start:i+1]
            if len(seq)<self.window: seq=np.concatenate([np.repeat(seq[:1],self.window-len(seq),axis=0),seq],axis=0)
            price=float(df.close.iloc[i]); net=balance+shares*price; peak=max(peak,net)
            portfolio=np.array([balance/10000,shares*price/10000,net/10000,(net-10000)/10000,
                                trades/100,costs/10000,(peak-net)/peak,
                                float(np.std(daily_returns[-20:])) if len(daily_returns)>1 else 0],np.float32)
            obs=np.concatenate([seq.reshape(-1),portfolio]).astype(np.float32)
            action,_=self.model.predict(obs,deterministic=True)
            a=np.asarray(action,dtype=float).reshape(-1); action_type=int(np.clip(np.rint(a[0]),0,2)); size=float(np.clip(a[1],0,1))
            # HF model source: 0 hold, 1 buy, 2 sell; map to common sell/hold/buy.
            common={0:1,1:2,2:0}[action_type]
            with torch.no_grad(): value=float(self.model.policy.predict_values(torch.as_tensor(obs[None,:])).item())
            logits,val=_logits_for(common,value); outs.append(logits); vals.append(val)
            if action_type==1 and balance>0:
                spend=balance*size; fee=spend*0.001; shares+=(spend-fee)/price; balance-=spend; costs+=fee; trades+=1
            elif action_type==2 and shares>0:
                sell=shares*size; gross=sell*price; fee=gross*0.001; balance+=gross-fee; shares-=sell; costs+=fee; trades+=1
            new_net=balance+shares*price
            if i: daily_returns.append(new_net/net-1 if net else 0)
        return np.asarray(outs,np.float32),np.asarray(vals,np.float32)


class RecurrentPFETeacher(Teacher):
    """jk2500/Pfizer-trader RecurrentPPO: LSTM, (100,9) observations, 2 directional actions."""
    required_window=100
    def __init__(self,name:str,path:str,weight:float=1.0):
        super().__init__(name,weight)
        from sb3_contrib import RecurrentPPO
        self.model=RecurrentPPO.load(path,device="cpu")
        if tuple(self.model.observation_space.shape)!=(100,9):
            raise ValueError(f"Unexpected recurrent PPO observation space: {self.model.observation_space}")
        if not hasattr(self.model.action_space,"n") or self.model.action_space.n!=2:
            raise ValueError(f"Unexpected recurrent PPO action space: {self.model.action_space}")

    @staticmethod
    def _features(df:pd.DataFrame)->pd.DataFrame:
        c=df.close.astype(float); f=pd.DataFrame(index=df.index)
        f["Open"]=df.open; f["High"]=df.high; f["Low"]=df.low; f["Close"]=df.close; f["Volume"]=df.volume
        f["Adj Close"]=df.adj_close if "adj_close" in df else df.close
        f["MA10"]=c.rolling(10).mean(); f["MA50"]=c.rolling(50).mean()
        delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=-delta.clip(upper=0).rolling(14).mean()+1e-8
        f["RSI"]=100-100/(1+gain/loss)
        return f.replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(0)

    def predict_series(self,df:pd.DataFrame,window:int)->tuple[np.ndarray,np.ndarray]:
        if window!=100: raise ValueError("Pfizer RecurrentPPO requires 100x9 feature windows")
        values=self._features(df).to_numpy(np.float64)
        n=max(50,int(len(values)*0.7)); scaler=StandardScaler().fit(values[:n])
        scaled=scaler.transform(values).astype(np.float32)
        state=None; episode_start=np.ones(1,dtype=bool); outs=[]; vals=[]
        for i in range(len(scaled)):
            part=scaled[max(0,i-window+1):i+1]
            if len(part)<window: part=np.concatenate([np.repeat(part[:1],window-len(part),axis=0),part],axis=0)
            action,state=self.model.predict(part[None,:,:],state=state,episode_start=episode_start,deterministic=True)
            action=int(np.asarray(action).reshape(-1)[0])
            # Source policy semantics: action 1 is long, action 0 is short.
            common=2 if action==1 else 0
            logits,val=_logits_for(common,0.0); outs.append(logits); vals.append(val)
            episode_start=np.zeros(1,dtype=bool)
        return np.asarray(outs,np.float32),np.asarray(vals,np.float32)


class DeepBioDQNTeacher(Teacher):
    """DeepBioLab MIT DQN, 1x3 normalized Close/BB-upper/BB-lower differences."""
    required_window = 1
    def __init__(self,name:str,path:str,weight:float=1.0):
        super().__init__(name,weight)
        class QNetwork(nn.Module):
            def __init__(self):
                super().__init__(); self.Q=nn.Sequential(nn.Linear(3,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),
                                                         nn.Linear(32,8),nn.ReLU(),nn.Linear(8,3))
            def forward(self,x): return self.Q(x)
        self.net=QNetwork()
        sd=torch.load(path,map_location="cpu",weights_only=True); self.net.load_state_dict(sd); self.net.eval()

    def predict_series(self,df:pd.DataFrame,window:int)->tuple[np.ndarray,np.ndarray]:
        if window!=1: raise ValueError("DeepBio DQN checkpoint was trained with window_size=1")
        c=df.close.astype(float); mid=c.rolling(20).mean(); std=c.rolling(20).std()
        raw=pd.DataFrame({"Close":c,"BB_upper":mid+2*std,"BB_lower":mid-2*std}).bfill().ffill()
        # Upstream normalizer uses a static standard scaler on its full training data.
        fit_n=max(20,int(len(raw)*0.70)); mu=raw.iloc[:fit_n].mean().to_numpy(); sigma=raw.iloc[:fit_n].std(ddof=0).replace(0,1).to_numpy()
        normalized=((raw.to_numpy()-mu)/(sigma+1e-8)).astype(np.float32)
        # Upstream Environment._get_state: consecutive differences, window_size=1.
        x=np.zeros((len(raw),3),np.float32); x[1:]=normalized[1:]-normalized[:-1]
        with torch.no_grad(): q=self.net(torch.as_tensor(x)); values,actions=q.max(-1)
        # DeepBio action ids: 0 HOLD, 1 BUY, 2 SELL.
        remap=np.asarray([1,2,0]); result=[]
        q_values=values.numpy().astype(np.float32)
        for a,v in zip(actions.tolist(),q_values.tolist()): result.append(_logits_for(int(remap[a]),v)[0])
        return np.asarray(result,np.float32),q_values
