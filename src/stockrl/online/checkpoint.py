"""Model checkpoint loading and atomic saving."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json, os
import numpy as np
import torch
from ..global_transformer import GlobalMarketTransformer, TransformerConfig, load_compatible_state_dict
from ..state_io import atomic_json
from ..multiscale import MULTISCALE_FEATURE_ORDER

def _atomic_save(obj, path: Path, temp_dir: Path | None = None):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp_dir=Path(temp_dir) if temp_dir is not None else path.parent
    temp_dir.mkdir(parents=True,exist_ok=True)
    tmp=temp_dir/(path.name+".tmp")
    try:
        torch.save(obj,tmp)
        # Replay may be acknowledged only after its checkpoint reaches disk.
        with tmp.open("rb+") as stream:
            stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp,path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_json(obj,path:Path):
    atomic_json(obj,path,default=lambda value: value.item() if isinstance(value,np.generic) else str(value))


def save_model(path:Path, model:GlobalMarketTransformer, cfg:TransformerConfig, optimizer=None, step=0,
               temp_dir:Path|None=None, replay_commit:dict|None=None):
    payload={"state_dict":model.state_dict(),"config":asdict(cfg),"step":step,
             "optimizer":optimizer.state_dict() if optimizer is not None else None}
    if replay_commit is not None:
        payload["replay_commit"]=replay_commit
    if getattr(model,"_stockrl_uses_market_context",False):
        from ..market_training import CONTEXT_FEATURES
        payload["context_features"]=list(CONTEXT_FEATURES)
        payload["symbol_map"]=model._stockrl_symbol_map
        payload["market_context_model"]=True
        payload["multiscale_feature_order"]=list(MULTISCALE_FEATURE_ORDER)
    _atomic_save(payload,path,temp_dir=temp_dir)


def load_model(path:Path,device:torch.device,instrument_config:Path|None=None):
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    cfg=TransformerConfig(**ckpt["config"])
    state=ckpt["state_dict"]
    contextual=(bool(ckpt.get("market_context_model")) or
                ("context_policy.weight" in state and
                 any(k.startswith("backbone.") for k in state)))
    if contextual:
        from ..market_training import CONTEXT_FEATURES, ContextConditionedTransformer
        if ckpt.get("context_features",list(CONTEXT_FEATURES))!=list(CONTEXT_FEATURES):
            raise ValueError("checkpoint market_context feature order does not match the inference schema")
        multiscale_order=list(MULTISCALE_FEATURE_ORDER)
        saved_order=ckpt.get("multiscale_feature_order",multiscale_order)
        if saved_order!=multiscale_order[:len(saved_order)]:
            raise ValueError("checkpoint multiscale feature order does not match the inference schema")
        symbol_map=ckpt.get("symbol_map")
        if not isinstance(symbol_map,dict) or len(symbol_map)!=cfg.max_symbols:
            raise ValueError("context checkpoint must contain its complete deterministic symbol_map")
        backbone=GlobalMarketTransformer(cfg)
        if device.type=="cuda": backbone=backbone.half()
        model=ContextConditionedTransformer(backbone)
        model.context_policy.float(); model.context_value.float()
        load_compatible_state_dict(model,state,strict=True)
        model._stockrl_uses_market_context=True
        model._stockrl_symbol_map=symbol_map
    else:
        model=GlobalMarketTransformer(cfg)
        load_compatible_state_dict(model,state,strict=True)
        if device.type=="cuda": model=model.half()
    model._stockrl_replay_commit=ckpt.get("replay_commit",{})
    del ckpt
    if instrument_config is not None:
        from ..instrument_ids import extend_live_symbols
        instruments=json.loads(Path(instrument_config).read_text(encoding="utf-8"))["instruments"]
        cfg=extend_live_symbols(model,cfg,instruments)
    model._stockrl_cfg=cfg
    model.to(device).eval()
    return model,cfg
