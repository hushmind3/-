"""Small learnable memory that reads every supplied completed daily OHLCV bar."""
import hashlib

import torch
from torch import nn


class DailyHistoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(6,32),nn.GELU())
        self.temporal=nn.Conv1d(32,32,5,padding=2)
        self.attention=nn.Linear(32,1)
        self.cache=None
        self.cache_hits=0
        self.cache_builds=0

    def forward(self,history):
        b,n,t,width=history.shape
        if width!=6:
            raise ValueError("daily history must contain OHLC returns, volume and validity")
        key=None
        if not self.training and not torch.is_grad_enabled():
            key=(tuple(history.shape),tuple(p._version for p in self.parameters()),
                hashlib.sha256(history.detach().cpu().numpy().tobytes()).digest())
            if self.cache is not None and self.cache[0]==key:
                self.cache_hits+=1
                return self.cache[1].to(history.device)
        rows=history.float().reshape(b*n,t,6)
        valid=rows[...,5]>0
        count=valid.sum(dim=-1,keepdim=True)
        # Position is relative to the completed history, not wall-clock time.
        position=(valid.cumsum(dim=-1)/count.clamp_min(1)).to(rows.dtype)
        inputs=torch.cat((rows[...,:5],position[...,None]),dim=-1)
        encoded=self.project(inputs)*valid[...,None]
        encoded=torch.nn.functional.gelu(self.temporal(encoded.transpose(1,2)).transpose(1,2))
        scores=self.attention(encoded).squeeze(-1).masked_fill(~valid,-1e4)
        weights=torch.softmax(scores,dim=-1)*valid
        memory=(encoded*weights[...,None]).sum(dim=1)*(count>0)
        memory=memory.reshape(b,n,32)
        if key is not None:
            self.cache_builds+=1
            self.cache=(key,memory.detach().cpu())
        return memory
