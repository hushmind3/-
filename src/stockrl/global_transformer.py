"""Global cross-asset Transformer and leakage-safe panel construction.

The default configuration is intentionally a real ~0.5B parameter model.
Tests construct an explicit small TransformerConfig; production uses the default.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .market_panel import (GLOBAL_FEATURES, ACTION_NAMES, TIME_SCALE_NAMES,
    MARKET_CONTEXT_FEATURES, GlobalMarketPanel, stable_id, _feature_panel)



@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 1408
    n_heads: int = 16
    n_layers: int = 21
    ff_mult: int = 4
    max_symbols: int = 8192
    n_markets: int = 64
    n_asset_types: int = 32
    max_seq_len: int = 128
    dropout: float = 0.0
    feature_count: int = len(GLOBAL_FEATURES)




class TransformerBlock(nn.Module):
    _stockrl_sdpa_enabled = True
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        d, h, f = cfg.d_model, cfg.n_heads, cfg.d_model * cfg.ff_mult
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=cfg.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, f), nn.GELU(), nn.Linear(f, d))
        self.residual_scale = 1.0 / math.sqrt(2.0 * cfg.n_layers)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        y = self.norm1(x)
        if self._stockrl_sdpa_enabled:
            # Same MHA weights and mask, directly through fused scaled attention.
            # Avoid MHA's different training/evaluation dispatch paths.
            batch, length, width = y.shape
            heads = self.attn.num_heads
            qkv = torch.nn.functional.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
            q, k, v = qkv.reshape(batch, length, 3, heads, width//heads).unbind(2)
            mask = None if pad_mask is None else (~pad_mask.bool())[:, None, None, :]
            attended = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask,
                dropout_p=self.attn.dropout if self.training else 0.0)
            attended = attended.transpose(1, 2).reshape(batch, length, width)
            attended = torch.nn.functional.linear(attended, self.attn.out_proj.weight, self.attn.out_proj.bias)
        else:
            attended = self.attn(y, y, y, key_padding_mask=pad_mask, need_weights=False)[0]
        x = x + self.residual_scale * attended
        return x + self.residual_scale * self.ff(self.norm2(x))


class GlobalMarketTransformer(nn.Module):
    """Alternating temporal and cross-market self-attention actor-critic.

    Input layout: features/ids/mask ``[batch,time,symbol,...]``. Logits and
    values describe each symbol at the final observation timestamp.
    """
    def __init__(self, cfg: TransformerConfig | None = None):
        super().__init__()
        self.cfg = cfg or TransformerConfig()
        c = self.cfg
        self.input_proj = nn.Linear(c.feature_count, c.d_model)
        self.symbol_embedding = nn.Embedding(c.max_symbols, c.d_model)
        self.market_embedding = nn.Embedding(c.n_markets, c.d_model)
        self.asset_embedding = nn.Embedding(c.n_asset_types, c.d_model)
        self.time_embedding = nn.Embedding(c.max_seq_len, c.d_model)
        # Small conditioning vector: the same market pattern can mean different
        # things at different sampling cadences. Zero initialization preserves
        # exact behavior for legacy checkpoints until this is trained.
        self.time_scale_embedding = nn.Embedding(len(TIME_SCALE_NAMES), c.d_model)
        nn.init.zeros_(self.time_scale_embedding.weight)
        self.blocks = nn.ModuleList([TransformerBlock(c) for _ in range(c.n_layers)])
        self.final_norm = nn.LayerNorm(c.d_model)
        self.policy_head = nn.Linear(c.d_model, 3)
        self.value_head = nn.Linear(c.d_model, 1)

    def forward(self, features: torch.Tensor, symbol_ids: torch.Tensor,
                market_ids: torch.Tensor, asset_ids: torch.Tensor,
                valid_mask: torch.Tensor,
                time_scale_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, n, _ = features.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len={self.cfg.max_seq_len}")
        x = self.input_proj(features)
        x = x + self.symbol_embedding(symbol_ids)[:, None] + self.market_embedding(market_ids)[:, None]
        x = x + self.asset_embedding(asset_ids)[:, None]
        x = x + self.time_embedding(torch.arange(t, device=x.device))[None, :, None]
        if time_scale_ids is not None:
            scale_ids = torch.as_tensor(time_scale_ids, device=x.device, dtype=torch.long)
            if scale_ids.ndim == 0:
                scale_ids = scale_ids.expand(b)
            if scale_ids.shape != (b,):
                raise ValueError(f"time_scale_ids must be scalar or shape [{b}]")
            if torch.any((scale_ids < 0) | (scale_ids >= len(TIME_SCALE_NAMES))):
                raise ValueError(f"time_scale_ids must be in [0,{len(TIME_SCALE_NAMES)-1}]")
            x = x + self.time_scale_embedding(scale_ids)[:, None, None, :]
        valid = valid_mask.bool()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        prefix_cache=getattr(self,"_online_prefix_cache",None)
        prefix_key=getattr(self,"_online_prefix_key",None)
        prefix_layers=getattr(self,"_online_prefix_layers",0) or 0
        cached=prefix_cache.get(prefix_key) if prefix_cache is not None and prefix_key is not None else None
        if cached is not None: x=cached
        for i, block in enumerate(self.blocks):
            if cached is not None and i<prefix_layers: continue
            if i % 2 == 0:  # within-asset temporal attention
                z = x.permute(0, 2, 1, 3).reshape(b*n, t, -1)
                mask = (~valid.permute(0, 2, 1)).reshape(b*n, t)
                # MHA returns NaNs for fully masked rows. Give them one safe key.
                mask[:, 0] &= ~mask.all(-1)
                if self.training and torch.is_grad_enabled() and (z.requires_grad or any(p.requires_grad for p in block.parameters())):
                    z = checkpoint(lambda value, layer=block, pad_mask=mask: layer(value, pad_mask),
                                   z, use_reentrant=False)
                else:
                    z = block(z, mask)
                x = z.reshape(b, n, t, -1).permute(0, 2, 1, 3)
            else:  # cross-asset/cross-market attention at each timestamp
                z = x.reshape(b*t, n, -1)
                mask = (~valid).reshape(b*t, n)
                mask[:, 0] &= ~mask.all(-1)
                if self.training and torch.is_grad_enabled() and (z.requires_grad or any(p.requires_grad for p in block.parameters())):
                    z = checkpoint(lambda value, layer=block, pad_mask=mask: layer(value, pad_mask),
                                   z, use_reentrant=False)
                else:
                    z = block(z, mask)
                x = z.reshape(b, t, n, -1)
            x = x * valid[..., None]
            if cached is None and prefix_cache is not None and i+1==prefix_layers:
                prefix_cache.put(prefix_key,x)
        x = self.final_norm(self.last_observed_state(x, valid))
        return self.policy_head(x), self.value_head(x).squeeze(-1)

    @staticmethod
    def last_observed_state(x, valid):
        # Asynchronous markets do not all print at the final panel timestamp.
        # Read each symbol's last actual observation, always within this window.
        b,t,n,_=x.shape
        positions=torch.arange(t,device=x.device)[None,:,None]
        latest=torch.where(valid,positions,-1).amax(dim=1).clamp_min(0)
        rows=x[torch.arange(b,device=x.device)[:,None],latest,
               torch.arange(n,device=x.device)[None,:]]
        return rows*valid.any(dim=1)[...,None]


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_compatible_state_dict(model: nn.Module, state: dict, strict: bool = True):
    """Load legacy checkpoints, zero-initializing added optional adapters."""
    target = model.state_dict()
    migrated = dict(state)
    for key in target:
        if key in migrated:
            if key in ("multiscale_policy.0.weight","multiscale_value.0.weight") and migrated[key].shape!=target[key].shape:
                prior=migrated[key]
                if prior.ndim!=2 or prior.shape[0]!=target[key].shape[0] or prior.shape[1]>target[key].shape[1]:
                    raise ValueError("incompatible multiscale adapter shape")
                expanded=torch.zeros_like(target[key])
                expanded[:,:prior.shape[1]]=prior.to(expanded)
                migrated[key]=expanded
            continue
        if key.endswith("time_scale_embedding.weight"):
            migrated[key] = torch.zeros_like(target[key])
        elif key.startswith("goal_"):
            migrated[key] = torch.zeros_like(target[key])
        elif (key.startswith("portfolio_action.") or
              key.startswith("portfolio_allocation.") or
              key.startswith("portfolio_cash.") or
              key.startswith("multiscale_policy.") or
              key.startswith("multiscale_value.") or
              key.startswith("daily_history_")):
            # Keep adapter hidden-layer initialization; their final layers are
            # zero-initialized by ContextConditionedTransformer.__init__.
            migrated[key] = target[key]
    return model.load_state_dict(migrated, strict=strict)
