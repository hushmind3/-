"""Append identities for newly configured instruments without remapping old IDs."""
from dataclasses import replace

import torch
from torch import nn


def extend_live_symbols(model, cfg, instruments):
    symbol_map=getattr(model,"_stockrl_symbol_map",None)
    if not symbol_map:
        return cfg
    aliases={"KRX":"KR","KOSDAQ":"KR"}
    additions=[]
    for item in instruments:
        if item.get("asset_class") not in ("equity","etf"):
            continue
        symbol=str(item["symbol"]).upper()
        market=aliases.get(item.get("market"),item.get("market",""))
        key=f"{market}|{item['asset_class']}|{symbol}"
        if any(existing.upper()==key.upper() for existing in symbol_map):
            continue
        # US aliases already resolve to a single historical identity.
        if market in ("US","NASDAQ","NYSE","NYSEARCA","AMEX"):
            matches=[key for key in symbol_map if key.rsplit("|",1)[-1].upper()==symbol]
            if len(matches)==1:
                continue
        if key not in additions:
            additions.append(key)
    if not additions:
        return cfg
    old=model.backbone.symbol_embedding
    next_id=max(map(int,symbol_map.values()))+1
    if next_id!=old.num_embeddings or len(symbol_map)!=next_id:
        raise ValueError("existing symbol identities must be contiguous before extension")
    expanded=nn.Embedding(next_id+len(additions),old.embedding_dim,
        device=old.weight.device,dtype=old.weight.dtype)
    with torch.no_grad():
        expanded.weight[:next_id].copy_(old.weight)
        for offset,key in enumerate(additions):
            region=key.split("|",1)[0]
            peers=[sid for identity,sid in symbol_map.items() if identity.split("|",1)[0]==region]
            seed=(old.weight[peers].float().mean(0) if peers else old.weight.float().mean(0))
            expanded.weight[next_id+offset].copy_(seed.to(old.weight.dtype))
    expanded.weight.requires_grad_(old.weight.requires_grad)
    model.backbone.symbol_embedding=expanded
    model._stockrl_symbol_map={**symbol_map,**{key:next_id+i for i,key in enumerate(additions)}}
    cfg=replace(cfg,max_symbols=expanded.num_embeddings)
    model.backbone.cfg=cfg
    return cfg
