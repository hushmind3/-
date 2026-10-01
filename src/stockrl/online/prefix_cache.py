"""Bounded reuse of identical, frozen market encodings across account inputs."""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import numpy as np
import torch
from .data import ONLINE_TRAINABLE_BLOCKS


class FrozenPrefixCache:
    def __init__(self, max_bytes=512*1024*1024, max_entries=8):
        self.max_bytes=max_bytes;self.max_entries=max_entries
        self.entries=OrderedDict();self.bytes=0;self.hits=0;self.misses=0

    def get(self,key):
        value=self.entries.get(key)
        if value is None:
            self.misses+=1
        else:
            self.hits+=1;self.entries.move_to_end(key)
        return value

    def put(self,key,value):
        size=value.numel()*value.element_size()
        if size>self.max_bytes: return
        if value.is_cuda:
            free,_=torch.cuda.mem_get_info(value.device)
            available=free+torch.cuda.memory_reserved(value.device)-torch.cuda.memory_allocated(value.device)
            if available<size+192*1024*1024: return
        while self.entries and (self.bytes+size>self.max_bytes or len(self.entries)>=self.max_entries):
            _,old=self.entries.popitem(last=False);self.bytes-=old.numel()*old.element_size()
        self.entries[key]=value.detach();self.bytes+=size

    def snapshot(self):
        return {"hits":self.hits,"misses":self.misses,"entries":len(self.entries),"bytes":self.bytes,
            "limit_bytes":self.max_bytes,"scope":"identical inputs and frozen weights only"}


def prefix_fingerprint(base, layers):
    parameters=[(name,p) for name,p in base.named_parameters() if
        not name.startswith(("final_norm.","policy_head.","value_head.")) and
        (not name.startswith("blocks.") or int(name.split(".")[1])<layers)]
    if not layers or base.cfg.dropout or any(p.requires_grad for _,p in parameters): return None
    versions=tuple((name,p._version,str(p.dtype),str(p.device)) for name,p in parameters)
    if getattr(base,"_prefix_parameter_versions",None)!=versions:
        digest=hashlib.sha256(json.dumps(asdict(base.cfg),sort_keys=True).encode())
        digest.update(str(layers).encode())
        for name,p in parameters:
            digest.update(name.encode());digest.update(memoryview(p.detach().cpu().contiguous().numpy()).cast("B"))
        base._prefix_fingerprint=digest.hexdigest();base._prefix_parameter_versions=versions
    return base._prefix_fingerprint


@contextmanager
def prefix_input(model, experience, cache):
    if cache is None:
        yield
        return
    base=getattr(model,"backbone",model)
    layers=max(0,len(getattr(base,"blocks",()))-ONLINE_TRAINABLE_BLOCKS)
    fingerprint=prefix_fingerprint(base,layers)
    if fingerprint is None:
        yield
        return
    digest=hashlib.sha256()
    feature_dtype=np.float16 if base.input_proj.weight.dtype==torch.float16 else np.float32
    for name,dtype in (("features",feature_dtype),("symbol_ids",np.int64),
                       ("market_ids",np.int64),("asset_ids",np.int64),("valid_mask",np.bool_)):
        value=np.ascontiguousarray(getattr(experience,name),dtype=dtype)
        digest.update(str(value.shape).encode());digest.update(memoryview(value).cast("B"))
    # Zero-dropout frozen SDPA uses the identical calculation in both modes.
    # Policy/value/portfolio heads and trainable blocks still run independently.
    mode = None if getattr(type(base.blocks[0]),"_stockrl_sdpa_enabled",False) else bool(base.training)
    key=(fingerprint,digest.hexdigest(),mode,str(base.input_proj.weight.device))
    previous=(getattr(base,"_online_prefix_key",None),getattr(base,"_online_prefix_cache",None),getattr(base,"_online_prefix_layers",None))
    base._online_prefix_key=key;base._online_prefix_cache=cache;base._online_prefix_layers=layers
    try:
        yield
    finally:
        base._online_prefix_key,base._online_prefix_cache,base._online_prefix_layers=previous


def install_prefix_forward():
    """Replace only the existing backbone forward method at a round boundary."""
    import ast
    import inspect
    from pathlib import Path
    from .. import global_transformer
    path=Path(global_transformer.__file__);stamp=path.stat().st_mtime_ns
    prefix_path=Path(__file__)
    stamp=(stamp,prefix_path.stat().st_mtime_ns)
    if getattr(global_transformer,"_online_forward_stamp",None)==stamp:
        return globals().get("_live_prefix_input",prefix_input)
    tree=ast.parse(path.read_text(encoding="utf-8"))
    namespace=dict(vars(global_transformer))
    for name in ("TransformerBlock","GlobalMarketTransformer"):
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==name)
        method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=="forward")
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),"exec"),namespace)
        updated=namespace["forward"];old=getattr(global_transformer,name).forward
        if list(inspect.signature(updated).parameters)!=list(inspect.signature(old).parameters):
            raise ValueError("backbone input schema changed; explicit migration required")
        getattr(global_transformer,name).forward=updated
    global_transformer.TransformerBlock._stockrl_sdpa_enabled=True
    prefix_tree=ast.parse(prefix_path.read_text(encoding="utf-8"))
    method=next(node for node in prefix_tree.body if isinstance(node,ast.FunctionDef) and node.name=="prefix_input")
    prefix_namespace=dict(globals())
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(prefix_path),"exec"),prefix_namespace)
    globals()["_live_prefix_input"]=prefix_namespace["prefix_input"]
    global_transformer._online_forward_stamp=stamp
    return globals()["_live_prefix_input"]
