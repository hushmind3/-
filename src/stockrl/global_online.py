"""Asynchronous online actor-critic over global asset panels.

Market observation and candidate training use separate threads and separate
model instances. Only a held-out score improvement swaps the champion.
"""
from __future__ import annotations

from collections import deque, OrderedDict
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
import json, os, random, shutil, threading, time, pickle, sqlite3, zlib, queue, io, gc
import re
import hashlib

import numpy as np
import psutil
import torch
from torch import nn
from torch.distributions import Categorical

from .global_transformer import (ACTION_NAMES, GlobalMarketPanel, GlobalMarketTransformer,
                                 TransformerConfig, parameter_count, load_compatible_state_dict)
from .paper_account import PaperAccount

REWARD_VERSION="symbol_and_portfolio_v4"
PAPER_EXPLORATION_EPSILON=0.05
REWARD_DEFINITION=("symbol_and_portfolio_v4: sampled-policy action probabilities with clipped importance weighting; "
                   "per-symbol net PnL trains action/allocation contribution; "
                   "normalized whole-account net equity change is a separate portfolio-value target")


class IncrementalMarketCSV:
    """Read an append-only feed once, then parse only complete appended rows."""
    def __init__(self, path: str | Path, retain_timestamps: int = 4096):
        self.path=Path(path); self.retain_timestamps=retain_timestamps
        self.frame=None; self.offset=0; self.partial=b""; self.header=b""; self.fingerprint=None

    def _fingerprint(self):
        stat=self.path.stat()
        with self.path.open("rb") as stream:
            prefix=stream.read(2048)
        return stat.st_size,stat.st_mtime_ns,getattr(stat,"st_ino",0),hashlib.sha256(prefix).digest()

    def _trim(self, frame):
        import pandas as pd
        if frame.empty or "date" not in frame or "symbol" not in frame:
            return frame
        stamps=pd.to_datetime(frame["date"],utc=True,errors="coerce")
        unique=stamps.dropna().drop_duplicates().sort_values()
        if len(unique)<=self.retain_timestamps:
            return frame
        cutoff=unique.iloc[-self.retain_timestamps]
        recent=frame.loc[stamps>=cutoff]
        reference=~frame.get("asset_class",pd.Series("equity",index=frame.index)).astype(str).str.casefold().eq("equity")
        carry=frame.loc[(stamps<cutoff)&reference].groupby("symbol",sort=False).tail(20)
        return pd.concat((carry,recent),ignore_index=True).sort_values(
            ["date","symbol"],kind="stable").reset_index(drop=True)

    def refresh(self):
        import pandas as pd
        signature=self._fingerprint()
        reset=(self.frame is None or signature[0]<self.offset or
               (self.fingerprint is not None and signature[2:]!=self.fingerprint[2:]))
        if reset:
            data=self.path.read_bytes()
            end=data.rfind(b"\n")+1
            complete=data[:end]
            self.partial=data[end:]
            self.header=complete.splitlines(keepends=True)[0] if complete else b""
            self.frame=pd.read_csv(io.BytesIO(complete)) if complete else pd.DataFrame()
            self.offset=len(data)
            self.frame=self._trim(self.frame)
        elif signature[0]>self.offset:
            with self.path.open("rb") as stream:
                stream.seek(self.offset); appended=stream.read()
            self.offset+=len(appended)
            combined=self.partial+appended
            end=combined.rfind(b"\n")+1
            complete=combined[:end]; self.partial=combined[end:]
            if complete:
                fresh=pd.read_csv(io.BytesIO(self.header+complete))
                self.frame=pd.concat((self.frame,fresh),ignore_index=True)
                self.frame=self.frame.drop_duplicates(["date","symbol"],keep="last")
                self.frame=self._trim(self.frame)
        self.fingerprint=signature
        return self.frame,signature


@dataclass
class Experience:
    features: np.ndarray
    symbol_ids: np.ndarray
    market_ids: np.ndarray
    asset_ids: np.ndarray
    valid_mask: np.ndarray
    symbol_index: int
    action: int
    reward: float
    timestamp: str
    source: str = "paper"
    regime: float = 0.0
    reward_version: str = "symbol_and_portfolio_v4"
    market_context: np.ndarray | None = None
    portfolio_state: np.ndarray | None = None
    account_state: np.ndarray | None = None
    portfolio_reward: float | None = None
    portfolio_transition: bool = False
    portfolio_value_transition: bool = False
    forward_return: float | None = None
    behavior_log_prob: float | None = None


class GlobalReplayBuffer:
    """Bounded replay with explicit recent/old/extreme regime mixture."""
    max_journal_bytes=64*1024*1024
    def __init__(self, capacity: int = 4_096, seed: int = 7,
                 journal_path: str|Path|None = None):
        self.capacity=capacity; self.items: deque[Experience]=deque(maxlen=capacity)
        self.rng=random.Random(seed); self.lock=threading.Lock(); self.paper_outcomes_seen=0
        self.journal_path=Path(journal_path) if journal_path is not None else None
        self.row_ids={}
        self.window_cache=OrderedDict()
        if self.journal_path is not None:
            self._load_journal()

    def _connect(self):
        assert self.journal_path is not None
        self.journal_path.parent.mkdir(parents=True,exist_ok=True)
        db=sqlite3.connect(self.journal_path,timeout=30)
        db.execute("PRAGMA journal_size_limit=8388608")
        return db

    @staticmethod
    def _window_key(exp:Experience):
        names=("features","symbol_ids","market_ids","asset_ids","valid_mask",
               "market_context","portfolio_state","account_state")
        key=hashlib.sha256()
        for name in names:
            value=getattr(exp,name)
            if value is None:
                key.update(name.encode()+b"=none")
            else:
                array=np.ascontiguousarray(value)
                key.update(name.encode()); key.update(str(array.dtype).encode())
                key.update(repr(array.shape).encode()); key.update(array.tobytes())
        return key.hexdigest()

    @staticmethod
    def _window_payload(exp:Experience):
        names=("features","symbol_ids","market_ids","asset_ids","valid_mask",
               "market_context","portfolio_state","account_state")
        arrays={name:(None if getattr(exp,name) is None else
                      np.ascontiguousarray(getattr(exp,name))) for name in names}
        return zlib.compress(pickle.dumps(arrays,protocol=5),level=1)

    @staticmethod
    def _metadata(exp:Experience):
        metadata={name:getattr(exp,name) for name in (
            "symbol_index","action","reward","timestamp","source","regime",
            "reward_version","portfolio_reward","portfolio_transition",
            "portfolio_value_transition","forward_return",
            "behavior_log_prob")}
        return pickle.dumps(metadata,protocol=5)

    def _load_journal(self):
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA journal_size_limit=8388608")
            db.execute("CREATE TABLE IF NOT EXISTS windows (key TEXT PRIMARY KEY, payload BLOB NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS experiences (id INTEGER PRIMARY KEY AUTOINCREMENT, window_key TEXT NOT NULL, metadata BLOB NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS pending_records (kind TEXT NOT NULL, record_key TEXT NOT NULL, metadata BLOB NOT NULL, PRIMARY KEY(kind,record_key))")
            with db:
                db.execute("DELETE FROM experiences WHERE id NOT IN (SELECT id FROM experiences ORDER BY id DESC LIMIT ?)",(self.capacity,))
                db.execute("DELETE FROM windows WHERE key NOT IN (SELECT DISTINCT window_key FROM experiences)")
            rows=db.execute("SELECT e.id,e.window_key,e.metadata,w.payload FROM experiences e JOIN windows w ON w.key=e.window_key ORDER BY e.id DESC LIMIT ?",(self.capacity,)).fetchall()
        decoded_windows={}
        portfolio_value_timestamps=set()
        for row_id,window_key,metadata_blob,payload_blob in reversed(rows):
            metadata=pickle.loads(metadata_blob)
            if metadata.get("portfolio_transition") and "portfolio_value_transition" not in metadata:
                stamp=metadata.get("timestamp")
                metadata["portfolio_value_transition"]=stamp not in portfolio_value_timestamps
                portfolio_value_timestamps.add(stamp)
            elif metadata.get("portfolio_value_transition"):
                portfolio_value_timestamps.add(metadata.get("timestamp"))
            arrays=decoded_windows.get(window_key)
            if arrays is None:
                arrays=pickle.loads(zlib.decompress(payload_blob))
                decoded_windows[window_key]=arrays
            exp=Experience(**arrays,**metadata)
            self.items.append(exp); self.row_ids[id(exp)]=int(row_id)

    def load_pending(self, kind: str) -> list[dict]:
        if self.journal_path is None:
            return []
        with closing(self._connect()) as db:
            db.execute("CREATE TABLE IF NOT EXISTS pending_records (kind TEXT NOT NULL, record_key TEXT NOT NULL, metadata BLOB NOT NULL, PRIMARY KEY(kind,record_key))")
            rows=db.execute("SELECT metadata FROM pending_records WHERE kind=? ORDER BY record_key",(kind,)).fetchall()
        return [pickle.loads(row[0]) for row in rows]

    def save_pending(self, regular: list[dict], portfolio: list[dict]) -> None:
        if self.journal_path is None:
            return
        transient_arrays={"features","symbol_ids","market_ids","asset_ids","valid_mask","market_context"}
        records=[]
        for kind,items in (("regular",regular),("portfolio",portfolio)):
            for item in items:
                metadata={key:value for key,value in item.items() if key not in transient_arrays}
                record_key=f"{metadata.get('timestamp','')}|{metadata.get('symbol',metadata.get('symbol_index',''))}"
                records.append((kind,record_key,pickle.dumps(metadata,protocol=5)))
        with closing(self._connect()) as db:
            db.execute("CREATE TABLE IF NOT EXISTS pending_records (kind TEXT NOT NULL, record_key TEXT NOT NULL, metadata BLOB NOT NULL, PRIMARY KEY(kind,record_key))")
            with db:
                db.executemany("INSERT OR REPLACE INTO pending_records(kind,record_key,metadata) VALUES(?,?,?)",
                               ((kind,key,sqlite3.Binary(metadata)) for kind,key,metadata in records))
                for kind,items in (("regular",regular),("portfolio",portfolio)):
                    keys=[f"{item.get('timestamp','')}|{item.get('symbol',item.get('symbol_index',''))}" for item in items]
                    keep={key for key in keys}
                    existing=db.execute("SELECT record_key FROM pending_records WHERE kind=?",(kind,)).fetchall()
                    stale=[(kind,row[0]) for row in existing if row[0] not in keep]
                    db.executemany("DELETE FROM pending_records WHERE kind=? AND record_key=?",stale)

    def _journal_add(self,exp:Experience,pending_ack:tuple[str,str]|None=None)->int|None:
        if self.journal_path is None:return None
        key=self._window_key(exp)
        payload=self.window_cache.get(key)
        if payload is None:
            payload=self._window_payload(exp)
            self.window_cache[key]=payload
            while len(self.window_cache)>4:self.window_cache.popitem(last=False)
        else:
            self.window_cache.move_to_end(key)
        metadata=self._metadata(exp)
        with closing(self._connect()) as db:
            with db:
                db.execute("INSERT OR IGNORE INTO windows(key,payload) VALUES(?,?)",(key,sqlite3.Binary(payload)))
                cursor=db.execute("INSERT INTO experiences(window_key,metadata) VALUES(?,?)",
                                  (key,sqlite3.Binary(metadata)))
                row_id=int(cursor.lastrowid)
                if pending_ack is not None:
                    db.execute("DELETE FROM pending_records WHERE kind=? AND record_key=?",pending_ack)
                db.execute("DELETE FROM experiences WHERE id NOT IN (SELECT id FROM experiences ORDER BY id DESC LIMIT ?)",(self.capacity,))
                db.execute("DELETE FROM windows WHERE key NOT IN (SELECT DISTINCT window_key FROM experiences)")
        return row_id
    def note_paper_outcome(self) -> None:
        with self.lock: self.paper_outcomes_seen+=1
    def add(self, exp: Experience, pending_ack: tuple[str,str]|None=None) -> None:
        with self.lock:
            if exp.portfolio_transition and any(
                    row.portfolio_transition and row.timestamp==exp.timestamp and
                    row.symbol_index==exp.symbol_index for row in self.items):
                if pending_ack is not None:
                    with closing(self._connect()) as db:
                        with db: db.execute("DELETE FROM pending_records WHERE kind=? AND record_key=?",pending_ack)
                return
            row_id=self._journal_add(exp,pending_ack)
            self.items.append(exp)
            if row_id is not None:self.row_ids[id(exp)]=row_id
            self._prune_row_ids()
            self._enforce_disk_limit_locked()

    def acknowledge_pending(self, kind: str, decision: dict) -> None:
        if self.journal_path is None:
            return
        key=f"{decision.get('timestamp','')}|{decision.get('symbol',decision.get('symbol_index',''))}"
        with closing(self._connect()) as db:
            with db:
                db.execute("DELETE FROM pending_records WHERE kind=? AND record_key=?",(kind,key))
    def _prune_row_ids(self):
        live={id(item) for item in self.items}
        self.row_ids={key:value for key,value in self.row_ids.items() if key in live}
    def disk_bytes(self)->int:
        if self.journal_path is None:return 0
        return sum(path.stat().st_size for path in (
            self.journal_path,Path(str(self.journal_path)+"-wal"),Path(str(self.journal_path)+"-shm"))
            if path.exists())
    def _enforce_disk_limit_locked(self):
        """Prune oldest records and compact only after the journal cap is crossed."""
        if self.journal_path is None or self.disk_bytes()<=self.max_journal_bytes:return
        keep_ids=set(); seen_windows=set(); logical_bytes=0
        budget=self.max_journal_bytes-4*1024*1024
        with closing(self._connect()) as db:
            rows=db.execute("SELECT e.id,e.window_key,length(e.metadata),length(w.payload) "
                            "FROM experiences e JOIN windows w ON w.key=e.window_key "
                            "ORDER BY e.id DESC").fetchall()
            for row_id,key,metadata_bytes,payload_bytes in rows:
                extra=int(metadata_bytes)+(0 if key in seen_windows else int(payload_bytes))
                if not keep_ids and extra>budget:break
                if keep_ids and logical_bytes+extra>budget:break
                keep_ids.add(int(row_id)); logical_bytes+=extra; seen_windows.add(key)
            oldest_kept=min(keep_ids) if keep_ids else 2**63-1
            with db:
                db.execute("DELETE FROM experiences WHERE id<?",(oldest_kept,))
                db.execute("DELETE FROM windows WHERE key NOT IN (SELECT DISTINCT window_key FROM experiences)")
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            db.execute("VACUUM")
        survivors=[]
        for item in self.items:
            row_id=self.row_ids.get(id(item))
            if row_id is None or row_id in keep_ids:survivors.append(item)
            else:self.row_ids.pop(id(item),None)
        self.items=deque(survivors,maxlen=self.capacity)
        self._prune_row_ids()
    def __len__(self):
        with self.lock: return len(self.items)
    def sample(self, batch_size: int, exclude_ids: set[int]|None=None) -> list[Experience]:
        with self.lock: items=list(self.items)
        if exclude_ids:
            items=[item for item in items if id(item) not in exclude_ids]
        if not items: return []
        teachers=[x for x in items if x.source.startswith("teacher")]
        # Candidate learning must optimize the same cash-only account used by
        # paper validation. The isolated per-action proxy can assign a short
        # reward to SELL, while PaperAccount forbids naked shorts; keep those
        # rows in the journal for audit, but do not sample them for training.
        paper=[x for x in items if x.source.startswith("paper")
               and getattr(x,"reward_version","legacy_transition_v1")==REWARD_VERSION]
        # Keep old samples on disk for auditability, but don't mix their former
        # position-transition rewards with the new isolated-horizon trade target.
        items=paper+teachers
        if not items: return []
        n=min(batch_size,len(items))
        teachers=[x for x in items if x.source.startswith("teacher")]
        paper=[x for x in items if not x.source.startswith("teacher")] or items
        valid_paper=[x for x in items if not x.source.startswith("teacher")
                     and getattr(x,"reward_version","legacy_transition_v1")==REWARD_VERSION]
        # Warm-start from demonstrations, then linearly remove their influence
        # as genuine net-PnL paper experiences accumulate. At 10k self outcomes
        # the replay becomes fully self-experience driven.
        teacher_fraction=(max(0.0,1.0-self.paper_outcomes_seen/10_000.0) if teachers else 0.0)
        recent=paper[-max(int(len(paper)*.2),1):]
        older=paper[:-len(recent)] or paper
        extreme=[x for x in paper if abs(x.regime)>.025 or abs(x.reward)>.02] or paper
        chosen=[]; chosen_ids=set()
        for _ in range(n):
            use_teacher=bool(teachers and (not valid_paper or self.rng.random()<teacher_fraction))
            pools=[teachers] if use_teacher else [recent,older,extreme]
            available=[[x for x in pool if id(x) not in chosen_ids] for pool in pools]
            available=[pool for pool in available if pool]
            if not available:
                available=[[x for x in items if id(x) not in chosen_ids]]
            if not available[0]:
                break
            if use_teacher:
                selected=self.rng.choice(available[0])
            else:
                bucket=self.rng.choices(available,weights=(.5,.3,.2)[:len(available)],k=1)[0]
                selected=self.rng.choice(bucket)
            chosen.append(selected); chosen_ids.add(id(selected))
        return chosen
    def discard(self, experiences: list[Experience]) -> int:
        """Drop examples that a successful candidate update actually used."""
        if not experiences:
            return 0
        consumed={id(item) for item in experiences}
        with self.lock:
            before=len(self.items)
            row_ids=[self.row_ids.pop(key) for key in consumed if key in self.row_ids]
            if row_ids and self.journal_path is not None:
                with closing(self._connect()) as db:
                    with db:
                        db.executemany("DELETE FROM experiences WHERE id=?",((row_id,) for row_id in row_ids))
                        db.execute("DELETE FROM windows WHERE key NOT IN (SELECT DISTINCT window_key FROM experiences)")
                    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.items=deque((item for item in self.items if id(item) not in consumed),
                             maxlen=self.capacity)
            return before-len(self.items)
    def row_ids_for(self, experiences: list[Experience]) -> list[int]:
        """Return stable journal IDs for the exact experiences selected to train."""
        with self.lock:
            return sorted({self.row_ids[id(item)] for item in experiences
                           if id(item) in self.row_ids})
    def discard_row_ids(self, row_ids: list[int]) -> int:
        """Consume journal rows after their candidate has been promoted."""
        ids={int(row_id) for row_id in row_ids}
        if not ids:
            return 0
        with self.lock:
            row_to_object={row_id:object_id for object_id,row_id in self.row_ids.items()}
            removed=0
            if self.journal_path is not None:
                with closing(self._connect()) as db:
                    with db:
                        cursor=db.executemany("DELETE FROM experiences WHERE id=?",
                                              ((row_id,) for row_id in ids))
                        removed=max(0,int(cursor.rowcount))
                        db.execute("DELETE FROM windows WHERE key NOT IN (SELECT DISTINCT window_key FROM experiences)")
                    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            removed_objects={row_to_object[row_id] for row_id in ids if row_id in row_to_object}
            self.items=deque((item for item in self.items if id(item) not in removed_objects),
                             maxlen=self.capacity)
            for object_id in removed_objects:
                self.row_ids.pop(object_id,None)
            return removed
    def teacher_fraction(self)->float:
        with self.lock:
            teachers=any(x.source.startswith("teacher") for x in self.items)
            paper=self.paper_outcomes_seen
        return max(0.0,1.0-paper/10_000.0) if teachers else 0.0
    def sample_source(self, source:str) -> Experience|None:
        with self.lock:
            rows=[x for x in self.items if x.source.startswith(source)]
            return self.rng.choice(rows) if rows else None
    def trainable_count(self)->int:
        with self.lock:
            return sum(x.source.startswith("teacher") or
                       (x.source.startswith("paper") and
                        getattr(x,"reward_version","legacy_transition_v1")==REWARD_VERSION)
                       for x in self.items)
    def load(self,path:Path,manifest_path:Path|None=None):
        if path.exists():
            manifest_path=manifest_path or path.with_suffix(".manifest.json")
            if manifest_path.exists():
                manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("format")!="stockrl-replay-chunks-v1":
                    raise ValueError(f"unsupported replay manifest: {manifest_path}")
                root=path.parent/(path.stem+"_chunks")
                generation=str(manifest.get("generation",""))
                for name in manifest.get("chunks",[]):
                    chunk_path=root/generation/name
                    data=torch.load(chunk_path,map_location="cpu",weights_only=False)
                    with self.lock:
                        for x in data: self.items.append(x)
                        self._prune_row_ids()
                    del data
            else:
                data=torch.load(path,map_location="cpu",weights_only=False)
                with self.lock:
                    for x in data[-self.capacity:]: self.items.append(x)
                    self._prune_row_ids()


def _atomic_save(obj, path: Path, temp_dir: Path | None = None):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp_dir=Path(temp_dir) if temp_dir is not None else path.parent
    temp_dir.mkdir(parents=True,exist_ok=True)
    tmp=temp_dir/(path.name+".tmp")
    try:
        torch.save(obj,tmp)
        os.replace(tmp,path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_json(obj,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,
                              default=lambda value: value.item() if isinstance(value,np.generic) else str(value)),
                   encoding="utf-8"); os.replace(tmp,path)


def should_promote(candidate_score:float, champion_score:float, minimum_delta:float=0.0)->bool:
    """Finite, strict promotion gate shared by online control and its checks."""
    return bool(np.isfinite(candidate_score) and np.isfinite(champion_score)
                and candidate_score>champion_score+minimum_delta)


def parse_horizon(value: int | str) -> tuple[str, int]:
    """Return (bars|seconds, amount); durations resolve to the first observed bar at/after target."""
    if isinstance(value, int):
        if value < 1: raise ValueError("horizon bars must be >= 1")
        return "bars", value
    match=re.fullmatch(r"\s*(\d+)\s*(bars?|s|sec|seconds?|m|min|minutes?|h|hours?)\s*",str(value).lower())
    if not match: raise ValueError("horizon must be e.g. 30s, 1m, 5m, 1bar, or an integer bar count")
    amount=int(match.group(1)); unit=match.group(2)
    if amount<1: raise ValueError("horizon duration must be >= 1")
    if unit.startswith("bar"): return "bars",amount
    scale=1 if unit in ("s","sec","second","seconds") else 60 if unit in ("m","min","minute","minutes") else 3600
    return "seconds",amount*scale


def net_action_reward(action:int, forward_return:float, previous_position:int,
                      fee:float, slippage:float) -> tuple[float,float,float]:
    """One equal-notional paper trade held to horizon; charge entry and exit costs."""
    position=int(action)-1
    # Each decision is evaluated as an isolated paper trade over its configured
    # horizon. BUY/SELL pay for opening and closing; HOLD stays in cash.
    turnover=2 if position else 0
    gross=float(position)*float(forward_return)
    fee_cost=float(fee)*turnover; slippage_cost=float(slippage)*turnover
    return gross-fee_cost-slippage_cost,fee_cost,slippage_cost


def save_model(path:Path, model:GlobalMarketTransformer, cfg:TransformerConfig, optimizer=None, step=0,
               temp_dir:Path|None=None):
    payload={"state_dict":model.state_dict(),"config":asdict(cfg),"step":step,
             "optimizer":optimizer.state_dict() if optimizer is not None else None}
    if getattr(model,"_stockrl_uses_market_context",False):
        from .market_training import CONTEXT_FEATURES
        payload["context_features"]=list(CONTEXT_FEATURES)
        payload["symbol_map"]=model._stockrl_symbol_map
        payload["market_context_model"]=True
    _atomic_save(payload,path,temp_dir=temp_dir)


def load_model(path:Path,device:torch.device):
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    cfg=TransformerConfig(**ckpt["config"])
    state=ckpt["state_dict"]
    contextual=(bool(ckpt.get("market_context_model")) or
                ("context_policy.weight" in state and
                 any(k.startswith("backbone.") for k in state)))
    if contextual:
        from .market_training import CONTEXT_FEATURES, ContextConditionedTransformer
        if ckpt.get("context_features",list(CONTEXT_FEATURES))!=list(CONTEXT_FEATURES):
            raise ValueError("checkpoint market_context feature order does not match the inference schema")
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
    del ckpt
    model.to(device).eval()
    return model,cfg


class OnlineGlobalAgent:
    def __init__(self, state_dir: str|Path, device="auto", config:TransformerConfig|None=None,
                 capacity=4_096, window=128, horizon=1, fee=.001, slippage_bps=1.0,
                 min_replay=8, batch_size=4, updates_per_candidate=4, lr=2e-6, seed=7,
                 candidate_interval=16, initial_champion: str|Path|None=None,
                 teacher_replay_path: str|Path|None=None,
                 model_dir: str|Path|None=None):
        from .core import device_for
        from .paths import ensure_project_path, validate_model_dir
        self.state_dir=ensure_project_path(state_dir, "runtime")
        self.model_dir=validate_model_dir(model_dir)
        if initial_champion is not None:
            source = Path(initial_champion).expanduser().resolve()
            if source.parent != self.model_dir or source.name not in {"champion.pt", "candidate.pt"}:
                raise ValueError(f"initial champion must be an existing checkpoint in {self.model_dir}")
            initial_champion = source
        self.state_dir.mkdir(parents=True,exist_ok=True)
        self.model_dir.mkdir(parents=True,exist_ok=True)
        self.device=device_for(device); self.cfg=config or TransformerConfig(); self.window=window
        self.horizon=horizon; self.horizon_kind,self.horizon_amount=parse_horizon(horizon)
        self.max_pending_age_seconds=30*86400
        self.fee=fee; self.slippage=slippage_bps/10_000
        self.min_replay=min_replay; self.batch_size=batch_size; self.updates_per_candidate=updates_per_candidate
        # Keep the live learner responsive while training from a real replay
        # minibatch; UI settings may request a shorter interval.
        self.candidate_interval=max(1,min(int(candidate_interval),512))
        self.min_validation_dates=0
        self.lr=lr; self.seed=seed; self.lock=threading.RLock(); self.stop=threading.Event()
        self.replay=GlobalReplayBuffer(capacity,seed,self.state_dir/"replay.sqlite3")
        # Replay is a bounded SQLite journal in the local runtime; checkpoints
        # remain the only model files. The optional teacher source is read-only.
        if teacher_replay_path: self.replay.load(Path(teacher_replay_path))
        self.champion_path=self.model_dir/"champion.pt"
        if initial_champion is not None and not Path(initial_champion).is_file():
            raise FileNotFoundError(f"configured champion checkpoint does not exist: {initial_champion}")
        if not self.champion_path.exists() and initial_champion:
            # Seed this isolated runtime from a prior verified champion without
            # writing to or replacing the source checkpoint.
            shutil.copy2(initial_champion,self.champion_path)
        if self.champion_path.exists():
            self.champion,self.cfg=load_model(self.champion_path,self.device)
        else:
            if model_dir is not None or initial_champion is not None:
                raise FileNotFoundError(f"configured champion checkpoint does not exist: {self.champion_path}")
            self.champion=GlobalMarketTransformer(self.cfg)
            if self.device.type=="cuda": self.champion=self.champion.half()
            self.champion=self.champion.to(self.device).eval()
            save_model(self.champion_path,self.champion,self.cfg,temp_dir=self.state_dir)
        self.candidate=None; self.optimizer=None; self.steps=0; self.updates=0
        self.candidate_retry_attempts=0; self.candidate_retry_after=0.0
        self.policy_rng=random.Random(seed+1)
        self.validation=[]
        self.portfolio_validation=[]
        self.validation_lock=threading.Lock()
        self.validation_window_dates=64
        self.validation_dates=deque()
        self.portfolio_validation_dates=deque()
        self.positions={}
        self.paper_account=PaperAccount(self.state_dir/"paper_account.json",self.fee,self.slippage)
        if not self.paper_account.path.exists(): self.paper_account.save()
        self.validation_window_bars=64
        self.validation_queue=queue.Queue(maxsize=2)
        self.validation_generation=0
        self.validation_queue_invalid_reason=None
        self.validation_queue_delay_limit_seconds=60.0
        self.validation_queue_thread=threading.Thread(
            target=self._validation_worker,name="candidate-validation",daemon=True)
        self.validation_state_path=self.state_dir/"candidate_validation.json"
        self.validation_champion_account=PaperAccount(
            self.state_dir/"candidate_validation_champion.json",self.fee,self.slippage)
        self.validation_candidate_account=PaperAccount(
            self.state_dir/"candidate_validation_candidate.json",self.fee,self.slippage)
        self.validation_start_after=None
        self.validation_source_sha256=None
        self.validation_bars=0
        self.validation_active=False
        self.current_market_timestamp=None
        positions_path=self.state_dir/"live_positions.json"
        if positions_path.exists():
            saved_positions=json.loads(positions_path.read_text(encoding="utf-8"))
            for key,value in saved_positions.items():
                try:
                    parsed_key=int(key)
                    parsed_key=parsed_key if str(parsed_key)==str(key) else str(key)
                except (TypeError,ValueError):
                    parsed_key=str(key)
                self.positions[parsed_key]=int(value)
        self.metrics={"observations":0,"decisions":0,"matured":0,"updates":0,"promotions":0,
                      "rejections":0,"nonfinite_updates":0,"teacher_examples_trained":0,"paper_examples_trained":0,
                      "paper_experiences_seen":0,"paper_experiences_since_candidate":0,
                      "paper_net_reward":0.0,"last_market_timestamp":None,"candidate_training":False,
                      "agent_health":"starting","agent_input_errors":0,
                      "agent_last_input_error":None,"agent_last_input_error_utc":None,
                      "observation_caught_up":False,"unmatched_live_symbols":[],
                      "unmatched_live_symbol_count":0,
                      "last_update_utc":None,"last_promotion_utc":None,"last_rejection_utc":None,
                      "inference_during_candidate":0,"reward_definition":REWARD_DEFINITION,
                      "champion_live_inference_count":0,
                      "champion_live_inference_seconds_total":0.0,
                      "champion_validation_inference_count":0,
                      "champion_validation_inference_seconds_total":0.0,
                      "candidate_validation_inference_count":0,
                      "candidate_validation_inference_seconds_total":0.0,
                      "candidate_replay_rows_held_for_validation":0,
                      "candidate_replay_rows_consumed":0,
                      "candidate_replay_cleanup_pending":False,
                      "update_losses":[],"inference_seconds":[],
                      "update_seconds":[],"weight_delta_l1":[]}
        metrics_path=self.state_dir/"metrics.json"
        if metrics_path.exists():
            try:
                self.metrics.update(json.loads(metrics_path.read_text(encoding="utf-8")))
            except (OSError,json.JSONDecodeError):
                pass
        stale_replay=[x for x in self.replay.items
                      if not x.source.startswith("teacher") and
                      getattr(x,"reward_version","legacy_transition_v1")!=REWARD_VERSION]
        if stale_replay:
            removed=self.replay.discard(stale_replay)
            # The old rows no longer contribute learning. Compact the same
            # replay journal in place; preserve its pending records and window
            # continuity, and never create a second replay file.
            with closing(self.replay._connect()) as db:
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                db.execute("VACUUM")
            self.metrics["legacy_replay_rows_discarded"] = int(
                self.metrics.get("legacy_replay_rows_discarded",0))+removed
            self.metrics["paper_experiences_seen"]=0
            self.metrics["paper_experiences_since_candidate"]=0
        prior_reward_definition=self.metrics.get("reward_definition")
        if prior_reward_definition!=REWARD_DEFINITION:
            # Counts from the previous account-wide reward are not eligible to
            # open a training cycle for the new symbol-level reward schema.
            self.metrics["paper_experiences_since_candidate"]=0
            self.metrics["reward_schema_reset_from"]=prior_reward_definition
        self.metrics["reward_definition"]=REWARD_DEFINITION
        for key in ("update_losses","update_seconds","weight_delta_l1"):
            self.metrics[key]=list(self.metrics.get(key,[]))[-2000:]
        self.promotion_baseline_path=self.state_dir/"promotion_baseline.json"
        current_champion_sha=self._sha256_file(self.champion_path).upper()
        # SHA identifies the champion captured for a validation window; it is
        # not a fixed allowlist. Clear obsolete fixed-SHA hold messages.
        self.promotion_baseline_sha256=current_champion_sha
        if str(self.metrics.get("promotion_blocked_reason", "")).startswith(
                ("champion lineage unresolved", "promotion held: champion lineage")):
            self.metrics["promotion_blocked_reason"]=None
            self.metrics["candidate_validation_error"]=None
        self.metrics.setdefault("paper_experiences_seen",
                                int(self.metrics.get("paper_examples_trained",0)))
        last_update=self.metrics.get("last_update_utc")
        if last_update:
            try:
                last_update_ts=np.datetime64(last_update)
                unconsumed_account_rows=sum(
                    exp.source=="paper_account_symbol" and exp.reward_version==REWARD_VERSION and
                    np.datetime64(exp.timestamp)>last_update_ts for exp in self.replay.items)
                self.metrics["paper_experiences_since_candidate"]=max(
                    int(self.metrics.get("paper_experiences_since_candidate",0)),
                    unconsumed_account_rows)
            except (TypeError,ValueError):
                pass
        self.replay.paper_outcomes_seen=int(self.metrics["paper_experiences_seen"])
        self.metrics["legacy_reward_examples_ignored"]=sum(
            not x.source.startswith("teacher") and getattr(x,"reward_version","legacy_transition_v1")!=REWARD_VERSION
            for x in self.replay.items)
        self.metrics["replay_persistence"]="bounded_sqlite_runtime"
        self.metrics["portfolio_experiences"] = int(self.metrics.get("portfolio_experiences",0))
        self.updates=int(self.metrics.get("updates",0)); self.steps=self.updates
        # Replay is restored from the bounded runtime journal after restart.
        self.metrics["candidate_training"]=False
        validation_state={}
        try:
            validation_state=json.loads(self.validation_state_path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError):
            pass
        # Complete the replay-consumption half of a successful promotion if
        # the process stopped after champion.pt was replaced but before the
        # SQLite delete was committed.
        if validation_state.get("status")=="promoted" and not validation_state.get("replay_rows_finalized"):
            consumed=self.replay.discard_row_ids(validation_state.get("trained_replay_row_ids",[]))
            validation_state["replay_rows_consumed"]=consumed
            validation_state["replay_rows_finalized"]=True
            self.metrics["candidate_replay_rows_consumed"]=consumed
            self.metrics["candidate_replay_cleanup_pending"]=False
            _atomic_json(validation_state,self.validation_state_path)
        self.validation_meta_path=self.state_dir/"validation_config.json"
        saved_validation_config={}
        try:
            saved_validation_config=json.loads(self.validation_meta_path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError):
            pass
        if saved_validation_config.get("reward_version")!=REWARD_VERSION:
            if validation_state.get("status")=="collecting":
                validation_state={"status":"discarded","reason":"paper reward schema changed"}
                _atomic_json(validation_state,self.validation_state_path)
            # A candidate trained with the old, currency-bugged reward cannot
            # be carried into the new policy cycle.
            shutil.copy2(self.champion_path,self.model_dir/"candidate.pt")
        if validation_state.get("status")=="collecting":
            if validation_state.get("source_champion_sha256")==self._sha256_file(self.champion_path):
                candidate_path=self.model_dir/"candidate.pt"
                if candidate_path.is_file():
                    self.candidate,self.cfg=load_model(candidate_path,self.device)
                    self.validation_start_after=validation_state.get("start_after")
                    self.validation_source_sha256=validation_state.get("source_champion_sha256")
                    self.validation_bars=int(validation_state.get("bars",0))
                    self.validation_generation=int(validation_state.get("generation",0))
                    self.validation_active=True
                    self.metrics["candidate_stage"]="sequential_paper_validation"
                    self.metrics["candidate_validation_bars"]=self.validation_bars
                    champion_ts=self.validation_champion_account.state.get("last_timestamp")
                    candidate_ts=self.validation_candidate_account.state.get("last_timestamp")
                    cursor_ts=None
                    try:
                        cursor_ts=json.loads((self.state_dir/"live_cursor.json").read_text(
                            encoding="utf-8")).get("last_timestamp")
                    except (OSError,json.JSONDecodeError,AttributeError):
                        pass
                    account_ts=self.paper_account.state.get("last_timestamp")
                    latest_cursor=max((str(x) for x in (cursor_ts,account_ts) if x),default=None)
                    saved_ts=validation_state.get("last_timestamp")
                    consistent=(champion_ts==candidate_ts==saved_ts
                                and int(validation_state.get("bars",-1))==self.validation_bars
                                and ((self.validation_bars==0 and saved_ts is None)
                                     or (self.validation_bars>0 and saved_ts is not None)))
                    missed_bars=(latest_cursor is not None and
                                 (saved_ts is None or latest_cursor>str(saved_ts)))
                    if not consistent or missed_bars:
                        self.validation_champion_account.reset()
                        self.validation_candidate_account.reset()
                        self.validation_start_after=latest_cursor or saved_ts
                        self.validation_bars=0
                        self.validation_generation+=1
                        validation_state.update({"status":"collecting",
                            "start_after":self.validation_start_after,"bars":0,
                            "source_champion_sha256":self.validation_source_sha256,
                            "generation":self.validation_generation,
                            "last_timestamp":None,
                            "reset_reason":"validation ledger timestamps were inconsistent or feed advanced"})
                        self._atomic_json(validation_state,self.validation_state_path)
                        self.metrics["candidate_validation_bars"]=0
            else:
                validation_state={"status":"discarded","reason":"champion changed during validation"}
                _atomic_json(validation_state,self.validation_state_path)
        self.metrics["candidate_skip_reason"]="새 paper 경험과 검증 시각을 기다리는 중"
        # Older runtime metrics may contain an obsolete held stage; resume the
        # normal candidate cycle without discarding replay or validation state.
        if self.metrics.get("candidate_stage") == "promotion_held":
            self.metrics["candidate_stage"] = "waiting"
        # New observations are journaled in bounded SQLite replay and trigger candidate learning.
        self.last_train_replay_size=int(self.metrics.get("last_train_replay_size",0))
        _atomic_json({"reward_version":REWARD_VERSION,"horizon":str(self.horizon)},self.validation_meta_path)
        self.thread=threading.Thread(target=self._learner,name="global-learner",daemon=True)

    @staticmethod
    def _sha256_file(path:Path)->str:
        digest=hashlib.sha256()
        with Path(path).open("rb") as stream:
            while chunk:=stream.read(1024*1024): digest.update(chunk)
        return digest.hexdigest()

    def _append_validation(self, target:list, date_order:deque, timestamp:str, item) -> None:
        """Keep a fixed, recent validation window shared by both models."""
        with self.validation_lock:
            if not date_order or date_order[-1] != timestamp:
                date_order.append(timestamp)
                while len(date_order)>self.validation_window_dates:
                    expired=date_order.popleft()
                    if target is self.validation:
                        target[:]=[row for row in target if row[7]!=expired]
                    else:
                        target[:]=[row for row in target if row.timestamp!=expired]
            target.append(item)

    def _reset_candidate_file(self) -> None:
        """Keep candidate.pt present but make it match the current champion."""
        candidate=self.model_dir/"candidate.pt"
        shutil.copy2(self.champion_path,candidate)

    def _target_index(self,panel:GlobalMarketPanel,start:int,symbol:int)->int|None:
        future=np.flatnonzero(panel.observed[start+1:,symbol])+start+1
        if not len(future): return None
        if self.horizon_kind=="bars":
            return int(future[self.horizon_amount-1]) if len(future)>=self.horizon_amount else None
        target=panel.dates[start]+np.timedelta64(self.horizon_amount,"s")
        eligible=future[panel.dates[future]>=target]
        return int(eligible[0]) if len(eligible) else None

    def start(self):
        if self.validation_queue_thread.ident is None: self.validation_queue_thread.start()
        if self.thread.ident is None: self.thread.start()
    def close(self):
        self.stop.set()
        if self.thread.is_alive(): self.thread.join(timeout=300)
        if self.validation_queue_thread.is_alive(): self.validation_queue_thread.join(timeout=300)

    def _infer(self,panel,index,portfolio_state=None,account_state=None):
        args=[x.to(self.device) for x in self._window(panel,index)]
        with self.lock:
            model=self.champion
        args[0]=args[0].to(dtype=next(model.parameters()).dtype)
        with torch.inference_mode():
            if self.device.type=="cuda": torch.cuda.synchronize(self.device)
            t=time.perf_counter()
            if getattr(model,"_stockrl_uses_market_context",False) and portfolio_state is not None:
                pstate=torch.as_tensor(np.asarray(portfolio_state,dtype=np.float32)[None],device=self.device)
                astate=torch.as_tensor(np.asarray(account_state,dtype=np.float32)[None],device=self.device)
                logits,values,allocation=model(*args,portfolio_state=pstate,
                                                account_state=astate,return_allocation=True)
            else:
                logits,values=model(*args); allocation=None
            elapsed=time.perf_counter()-t
            if self.device.type=="cuda":
                torch.cuda.synchronize(self.device); elapsed=time.perf_counter()-t
        self.metrics["inference_seconds"].append(elapsed)
        self.metrics["champion_live_inference_count"]=(
            int(self.metrics.get("champion_live_inference_count",0))+1)
        self.metrics["champion_live_inference_seconds_total"]=(
            float(self.metrics.get("champion_live_inference_seconds_total",0.0))+elapsed)
        if self.metrics.get("candidate_training"):
            self.metrics["inference_during_candidate"]=int(self.metrics.get("inference_during_candidate",0))+1
        if len(self.metrics["inference_seconds"])>2000: self.metrics["inference_seconds"]=self.metrics["inference_seconds"][-2000:]
        return (logits[0].cpu().numpy(),values[0].cpu().numpy(),
                allocation[0].cpu().float().numpy() if allocation is not None else None)

    def _window(self,panel,index):
        contextual=getattr(self.champion,"_stockrl_uses_market_context",False)
        return panel.window(index,self.window,include_context=contextual)

    def _mature(self,pending,panel,end_index):
        keep=[]
        for dec in pending:
            if "entry_price" in dec:
                # Keep the entry price with the decision so rolling the panel
                # cannot make the original row disappear before its outcome.
                stamp=np.datetime64(dec["timestamp"])
                now=panel.dates[end_index]
                if float((now-stamp)/np.timedelta64(1,"s"))>self.max_pending_age_seconds:
                    self.metrics["pending_expired_after_window"] = int(
                        self.metrics.get("pending_expired_after_window",0))+1
                    continue
                symbol=dec.get("symbol")
                if symbol not in panel.symbols:
                    keep.append(dec); continue
                symbol_ix=panel.symbols.index(symbol)
                if now<=stamp or not panel.observed[end_index,symbol_ix]:
                    keep.append(dec); continue
                if self.horizon_kind=="bars":
                    dec["bars_elapsed"]=int(dec.get("bars_elapsed",0))+1
                    matured=dec["bars_elapsed"]>=self.horizon_amount
                else:
                    target=stamp+np.timedelta64(self.horizon_amount,"s")
                    matured=now>=target
                if not matured:
                    keep.append(dec); continue
                entry=float(dec["entry_price"]); exit_price=float(panel.closes[end_index,symbol_ix])
                if not np.isfinite(entry*exit_price) or entry<=0:
                    self.metrics["pending_expired_after_window"] = int(
                        self.metrics.get("pending_expired_after_window",0))+1
                    continue
                forward=exit_price/entry-1.0
                dec["index"]=end_index; dec["symbol_index"]=symbol_ix
            else:
                # Finite historical runs still refer to their fixed panel rows.
                if dec.get("timestamp"):
                    stamp=np.datetime64(dec["timestamp"])
                    start_ix=int(np.searchsorted(panel.dates,stamp,side="left"))
                    if start_ix>=len(panel.dates) or panel.dates[start_ix]!=stamp:
                        keep.append(dec); continue
                    symbol_ix=(panel.symbols.index(dec["symbol"])
                               if dec.get("symbol") in panel.symbols else dec["symbol_index"])
                    dec["index"]=start_ix; dec["symbol_index"]=symbol_ix
                symbol_ix=dec["symbol_index"]
                target=self._target_index(panel,dec["index"],symbol_ix)
                if target is None or end_index<target:
                    keep.append(dec); continue
                forward=panel.return_to(dec["index"],target,symbol_ix)
            i=symbol_ix
            self.metrics["matured"]+=1
            if dec.get("promotion_holdout",False):
                self.metrics["promotion_validation_outcomes"] = int(
                    self.metrics.get("promotion_validation_outcomes",0))+1
                self.replay.acknowledge_pending("regular",dec)
                continue
            if dec.get("is_validation",False):
                self._append_validation(self.validation,self.validation_dates,dec["timestamp"],
                    (dec["features"],dec["symbol_ids"],dec["market_ids"],dec["asset_ids"],
                    dec["valid_mask"],i,forward,dec["timestamp"],dec.get("previous_position",0),
                    dec.get("market_context")))
                self.replay.acknowledge_pending("regular",dec)
                continue
            action=dec["action"]
            reward,fee_cost,slippage_cost=net_action_reward(action,forward,dec.get("previous_position",0),
                                                            self.fee,self.slippage)
            self.metrics["paper_net_reward"]+=float(reward)
            action_name=ACTION_NAMES[action]
            rewards=self.metrics.setdefault("action_net_reward",{"SELL":0.0,"HOLD":0.0,"BUY":0.0})
            counts=self.metrics.setdefault("action_count",{"SELL":0,"HOLD":0,"BUY":0})
            rewards[action_name]=float(rewards.get(action_name,0.0))+float(reward)
            counts[action_name]=int(counts.get(action_name,0))+1
            self.metrics["fee_total"] = float(self.metrics.get("fee_total",0.0))+fee_cost
            self.metrics["slippage_total"] = float(self.metrics.get("slippage_total",0.0))+slippage_cost
            day=str(dec.get("timestamp","")[:10])
            daily=self.metrics.setdefault("daily_net_pnl",{})
            daily[day]=float(daily.get(day,0.0))+float(reward)
            regime=(float(dec["regime"]) if "regime" in dec
                    else abs(float(panel.features[dec["index"],i,6])))
            self.replay.add(Experience(dec["features"],dec["symbol_ids"],dec["market_ids"],dec["asset_ids"],
                dec["valid_mask"],i,action,float(reward),dec["timestamp"],"paper",max(regime,abs(forward)),
                reward_version=REWARD_VERSION,market_context=dec.get("market_context"),
                behavior_log_prob=dec.get("behavior_log_prob")),
                pending_ack=("regular",f"{dec.get('timestamp','')}|{dec.get('symbol',dec.get('symbol_index',''))}"))
        return keep

    def _mature_portfolio(self, pending, panel, end_index: int):
        """Turn completed paper-account transitions into reward experiences.

        The reward is the account's net-of-cost normalized equity change.  It
        is attached to the exact decision window and portfolio state, so the
        learner and promotion gate optimize the same virtual account result.
        """
        if not pending:
            return pending
        now = float(self.paper_account.normalized_equity())
        keep=[]; account_transition_added=set()
        for dec in pending:
            stamp=np.datetime64(dec["timestamp"]); current=panel.dates[end_index]
            if float((current-stamp)/np.timedelta64(1,"s"))>self.max_pending_age_seconds:
                self.metrics["portfolio_pending_expired_after_window"] = int(
                    self.metrics.get("portfolio_pending_expired_after_window",0))+1
                continue
            symbol=dec.get("symbol")
            if symbol not in panel.symbols:
                keep.append(dec); continue
            symbol_ix=panel.symbols.index(symbol)
            if current<=stamp or not panel.observed[end_index,symbol_ix]:
                keep.append(dec); continue
            if self.horizon_kind=="bars":
                dec["bars_elapsed"]=int(dec.get("bars_elapsed",0))+1
                matured=dec["bars_elapsed"]>=self.horizon_amount
            else:
                matured=current>=stamp+np.timedelta64(self.horizon_amount,"s")
            if not matured:
                keep.append(dec); continue
            account_reward = now - float(dec.get("equity_before", now))
            symbol_reward = (self.paper_account.symbol_net_pnl(symbol)
                             - float(dec.get("symbol_pnl_before", 0.0)))
            if dec.get("promotion_holdout",False):
                self.metrics["promotion_validation_outcomes"] = int(
                    self.metrics.get("promotion_validation_outcomes",0))+1
                self.replay.acknowledge_pending("portfolio",dec)
                continue
            entry=float(dec.get("entry_price",0.0)); exit_price=float(panel.closes[end_index,symbol_ix])
            forward_return=(exit_price/entry-1.0 if entry>0 and np.isfinite(entry*exit_price) else None)
            exp=Experience(dec["features"],dec["symbol_ids"],dec["market_ids"],dec["asset_ids"],
                    dec["valid_mask"],int(symbol_ix),int(dec["action"]),float(symbol_reward),
                    dec["timestamp"],"paper_account_symbol",float(dec.get("regime",0.0)),
                    market_context=dec.get("market_context"), portfolio_state=dec.get("portfolio_state"),
                    account_state=dec.get("account_state"),forward_return=forward_return,
                    behavior_log_prob=dec.get("behavior_log_prob"))
            if dec.get("is_validation",False):
                self._append_validation(self.portfolio_validation,self.portfolio_validation_dates,
                                        exp.timestamp,exp)
                self.replay.acknowledge_pending("portfolio",dec)
            else:
                self.replay.add(exp,pending_ack=("portfolio",
                    f"{dec.get('timestamp','')}|{dec.get('symbol',dec.get('symbol_index',''))}"))
                timestamp=dec["timestamp"]
                # Keep one allocation-credit row per symbol. It carries that
                # symbol's realized contribution separately from the shared
                # whole-account result; never attach the whole account return
                # to whichever symbol happened to mature first.
                with self.replay.lock:
                    first_account_transition=(timestamp not in account_transition_added and not any(
                        item.portfolio_value_transition and item.timestamp==timestamp
                        for item in self.replay.items))
                account_exp=Experience(dec["features"],dec["symbol_ids"],dec["market_ids"],dec["asset_ids"],
                    dec["valid_mask"],int(symbol_ix),int(dec["action"]),float(symbol_reward),timestamp,
                    "paper_account_portfolio",float(dec.get("regime",0.0)),
                    market_context=dec.get("market_context"),portfolio_state=dec.get("portfolio_state"),
                    account_state=dec.get("account_state"),portfolio_reward=float(account_reward),
                    portfolio_transition=True,portfolio_value_transition=first_account_transition,
                    forward_return=forward_return,
                    behavior_log_prob=dec.get("behavior_log_prob"))
                self.replay.add(account_exp)
                if first_account_transition:
                    account_transition_added.add(timestamp)
                    self.metrics["paper_account_reward"] = float(
                        self.metrics.get("paper_account_reward",0.0))+account_reward
                    self.metrics["portfolio_experiences"] = int(
                        self.metrics.get("portfolio_experiences",0))+1
                self.metrics["paper_experiences_seen"]=int(
                    self.metrics.get("paper_experiences_seen",0))+1
                self.metrics["paper_experiences_since_candidate"]=int(
                    self.metrics.get("paper_experiences_since_candidate",0))+1
                self.replay.note_paper_outcome()
        return keep

    def follow_csv(self, data_path:str|Path, poll_seconds:float=5.0, initial_lookback_bars:int=0):
        """Observe an append-only UTC timestamp CSV until Ctrl+C.

        Feed columns follow the same global schema as the historical downloader.
        New bars can be minute/hourly; all instruments for a timestamp should
        be appended as a batch. Candidate training runs on the learner thread.
        """
        data_path=Path(data_path); cursor_path=self.state_dir/"live_cursor.json"
        decisions_path=self.state_dir/"decisions.csv"
        cursor=None
        last_file_signature=None
        last_error_signature=None
        if cursor_path.exists(): cursor=json.loads(cursor_path.read_text(encoding="utf-8")).get("last_timestamp")
        account_cursor=self.paper_account.state.get("last_timestamp")
        if account_cursor and (cursor is None or account_cursor>cursor): cursor=account_cursor
        pending=self.replay.load_pending("regular")
        portfolio_pending=self.replay.load_pending("portfolio")
        market_reader=IncrementalMarketCSV(data_path,retain_timestamps=4096)
        self.metrics["pending_experiences"] = len(pending) + len(portfolio_pending)
        self.start()
        try:
            while not self.stop.is_set():
                if (self.state_dir/"stop.request").exists():
                    break
                if not data_path.exists():
                    self.stop.wait(poll_seconds); continue
                try:
                    stat=data_path.stat()
                    file_signature=(stat.st_size,stat.st_mtime_ns,getattr(stat,"st_ino",0))
                    if file_signature==last_file_signature:
                        self.stop.wait(poll_seconds); continue
                    self.metrics["observation_caught_up"]=False
                    self.metrics["agent_health"]="catching_up"
                    raw_frame,_reader_signature=market_reader.refresh()
                    panel=GlobalMarketPanel(data_path, max_symbols=self.cfg.max_symbols,
                        symbol_map=getattr(self.champion,"_stockrl_symbol_map",None),
                        # Keep closed-session indices, futures, yields, and other
                        # reference markets in the action set using their last
                        # known quote. The dashboard still marks their quote as
                        # stale; dropping them here hid their BUY/HOLD/SELL
                        # output entirely whenever their venue was closed.
                        recent_timestamps=4096, active_stale_seconds=604800,
                        raw_frame=raw_frame)
                except (OSError,ValueError) as exc:
                    # A producer may be in the middle of appending a CSV batch.
                    # Expose a stable error per file version; otherwise an
                    # input failure looks like a healthy but idle agent.
                    if file_signature != last_error_signature:
                        from datetime import datetime, timezone
                        self.metrics["agent_health"]="input_error"
                        self.metrics["agent_input_errors"]=int(
                            self.metrics.get("agent_input_errors",0))+1
                        self.metrics["agent_last_input_error"]=f"{type(exc).__name__}: {exc}"
                        self.metrics["agent_last_input_error_utc"]=datetime.now(timezone.utc).isoformat()
                        self.metrics["observation_caught_up"]=False
                        error_record={
                            "time_utc":self.metrics["agent_last_input_error_utc"],
                            "file_bytes":stat.st_size,"file_mtime_ns":stat.st_mtime_ns,
                            "error":self.metrics["agent_last_input_error"],
                        }
                        with (self.state_dir/"agent_errors.jsonl").open("a",encoding="utf-8") as log:
                            log.write(json.dumps(error_record,ensure_ascii=False)+"\n")
                            log.flush(); os.fsync(log.fileno())
                        self._write_metrics()
                        last_error_signature=file_signature
                    self.stop.wait(min(poll_seconds,1.0)); continue
                if last_error_signature is not None:
                    self.metrics["agent_health"]="catching_up"
                    self.metrics["agent_last_input_error"]=None
                    self.metrics["agent_last_input_error_utc"]=None
                    last_error_signature=None
                self.metrics["unmatched_live_symbols"] = list(
                    getattr(panel,"unmatched_symbols",()))
                self.metrics["unmatched_live_symbol_count"] = len(
                    self.metrics["unmatched_live_symbols"])
                if not len(panel.dates):
                    self.stop.wait(poll_seconds); continue
                oldest_available=(panel.recent_cutoff if panel.recent_cutoff is not None
                                  else panel.dates[0])
                if cursor is not None and np.datetime64(cursor) < oldest_available:
                    self.metrics["agent_health"]="history_gap"
                    self.metrics["observation_caught_up"]=False
                    self.metrics["agent_history_gap"]={
                        "saved_cursor":str(cursor),"oldest_available":str(oldest_available),
                        "retained_timestamps":4096,
                    }
                    self.metrics["candidate_skip_reason"]=(
                        "저장 cursor가 보존된 시세 범위보다 오래되어 연속 처리를 대기 중")
                    self._write_metrics()
                    last_file_signature=file_signature
                    self.stop.wait(poll_seconds)
                    continue
                self.metrics["agent_history_gap"]=None
                restore_windows={}
                expired_regular=expired_portfolio=0
                def restore_pending(items, is_portfolio):
                    nonlocal expired_regular, expired_portfolio
                    restored=[]
                    for dec in items:
                        if "features" in dec:
                            restored.append(dec)
                            continue
                        symbol=dec.get("symbol")
                        if symbol not in panel.symbols:
                            restored.append(dec)
                            continue
                        stamp=np.datetime64(dec["timestamp"])
                        index=int(np.searchsorted(panel.dates,stamp,side="left"))
                        if index>=len(panel.dates) or panel.dates[index]!=stamp:
                            if is_portfolio: expired_portfolio+=1
                            else: expired_regular+=1
                            continue
                        window=restore_windows.get(index)
                        if window is None:
                            window=self._window(panel,index)
                            restore_windows[index]=window
                        x0,sid0,mid0,aid0,mask0=window[:5]
                        dec.update({"index":index,"symbol_index":panel.symbols.index(symbol),
                            "features":x0[0].numpy().astype(np.float16),"symbol_ids":sid0[0].numpy(),
                            "market_ids":mid0[0].numpy(),"asset_ids":aid0[0].numpy(),
                            "valid_mask":mask0[0].numpy(),
                            "market_context":window[5][0].numpy().astype(np.float16) if len(window)>5 else None})
                        restored.append(dec)
                    return restored
                pending=restore_pending(pending,False)
                portfolio_pending=restore_pending(portfolio_pending,True)
                # Minute-live operation favors current observations over
                # replaying a stale feed backlog. Expire unresolved outcomes
                # older than five minutes and remove their old reward rows
                # from the same bounded journal before learning can resume.
                live_cutoff=panel.dates[-1]-np.timedelta64(300,"s")
                stale_regular=[dec for dec in pending
                    if np.datetime64(dec.get("timestamp"))<live_cutoff]
                stale_portfolio=[dec for dec in portfolio_pending
                    if np.datetime64(dec.get("timestamp"))<live_cutoff]
                if stale_regular or stale_portfolio:
                    stale_regular_ids={id(dec) for dec in stale_regular}
                    stale_portfolio_ids={id(dec) for dec in stale_portfolio}
                    pending=[dec for dec in pending if id(dec) not in stale_regular_ids]
                    portfolio_pending=[dec for dec in portfolio_pending
                                       if id(dec) not in stale_portfolio_ids]
                    self.metrics["live_stale_pending_discarded"]=int(
                        self.metrics.get("live_stale_pending_discarded",0))+len(stale_regular)+len(stale_portfolio)
                    self.replay.save_pending(pending,portfolio_pending)
                stale_replay=[exp for exp in self.replay.items
                    if np.datetime64(exp.timestamp)<live_cutoff and not exp.source.startswith("teacher")]
                if stale_replay:
                    stale_account_rows=sum(exp.source=="paper_account_symbol" for exp in stale_replay)
                    removed=self.replay.discard(stale_replay)
                    self.metrics["live_stale_replay_discarded"]=int(
                        self.metrics.get("live_stale_replay_discarded",0))+removed
                    self.metrics["paper_experiences_since_candidate"]=max(
                        0,int(self.metrics.get("paper_experiences_since_candidate",0))-stale_account_rows)
                if expired_regular or expired_portfolio:
                    self.metrics["pending_expired_after_window"] = int(
                        self.metrics.get("pending_expired_after_window",0))+expired_regular
                    self.metrics["portfolio_pending_expired_after_window"] = int(
                        self.metrics.get("portfolio_pending_expired_after_window",0))+expired_portfolio
                    self.metrics["pending_experiences"] = len(pending)+len(portfolio_pending)
                    self.replay.save_pending(pending,portfolio_pending)
                # The first feed batch establishes the ordered symbol universe;
                # migrate legacy integer-keyed paper positions to symbols.
                if any(isinstance(k,int) for k in self.positions):
                    self.positions={panel.symbols[k] if isinstance(k,int) and 0<=k<len(panel.symbols) else str(k):v
                                    for k,v in self.positions.items()}
                if cursor is None:
                    # Replay a small startup tail so delayed paper outcomes can
                    # mature immediately after the first launch. A persisted
                    # cursor always takes precedence after restart.
                    lookback=max(0,int(initial_lookback_bars))
                    start=max(0,len(panel.dates)-max(lookback,1))
                else:
                    cursor_dt=np.datetime64(cursor)
                    oldest=panel.recent_cutoff if panel.recent_cutoff is not None else panel.dates[0]
                    if cursor_dt<oldest:
                        self.metrics["market_history_skipped_after_window"] = int(
                            self.metrics.get("market_history_skipped_after_window",0))+1
                        self.metrics["pending_expired_after_window"] = int(
                            self.metrics.get("pending_expired_after_window",0))+len(pending)
                        self.metrics["portfolio_pending_expired_after_window"] = int(
                            self.metrics.get("portfolio_pending_expired_after_window",0))+len(portfolio_pending)
                        pending.clear(); portfolio_pending.clear()
                        start=max(0,len(panel.dates)-1)
                    else:
                        cursor_index=int(np.searchsorted(panel.dates,cursor_dt,side="right"))
                        lag_seconds=float((panel.dates[-1]-cursor_dt)/np.timedelta64(1,"s"))
                        if lag_seconds>300:
                            skipped=max(0,len(panel.dates)-cursor_index)
                            self.metrics["live_backlog_skipped_timestamps"]=int(
                                self.metrics.get("live_backlog_skipped_timestamps",0))+skipped
                            pending=[]; portfolio_pending=[]
                            self.replay.save_pending(pending,portfolio_pending)
                            start=max(0,len(panel.dates)-1)
                        else:
                            start=cursor_index
                for ti in range(start,len(panel.dates)):
                    if self.stop.is_set(): break
                    self.current_market_timestamp=str(panel.dates[ti])
                    # Autonomy is distinct from the model's HOLD action. When
                    # disabled, inference and counterfactual learning continue,
                    # but directional paper positions are left unchanged.
                    try:
                        mode_state=json.loads((self.state_dir/"autonomy.json").read_text(encoding="utf-8"))
                        paper_enabled=bool(mode_state.get("paper_enabled",mode_state.get("enabled",True)))
                        observe_enabled=bool(mode_state.get("observe_enabled",True))
                    except (OSError,json.JSONDecodeError,AttributeError):
                        paper_enabled=True; observe_enabled=True
                    self.metrics["autonomy_enabled"]=paper_enabled
                    self.metrics["paper_enabled"]=paper_enabled
                    self.metrics["observe_enabled"]=observe_enabled
                    self.paper_account.process_bar(panel,ti,paper_enabled)
                    portfolio_pending=self._mature_portfolio(portfolio_pending,panel,ti)
                    pending=self._mature(pending,panel,ti)
                    if not observe_enabled:
                        cursor=str(panel.dates[ti]); self.metrics["observations"]+=1
                        self.metrics["last_market_timestamp"]=cursor
                        self.metrics["pending_experiences"] = len(pending)+len(portfolio_pending)
                        self.replay.save_pending(pending,portfolio_pending)
                        self.paper_account.save()
                        _atomic_json({"last_timestamp":cursor},cursor_path)
                        continue
                    pstate,astate=self.paper_account.model_inputs(panel,ti)
                    logits,values,allocation=self._infer(panel,ti,pstate,astate)
                    model_probs=torch.softmax(torch.as_tensor(logits),-1).numpy()
                    probs=(1.0-PAPER_EXPLORATION_EPSILON)*model_probs+PAPER_EXPLORATION_EPSILON/3.0
                    actions=[self.policy_rng.choices((0,1,2),weights=row.tolist(),k=1)[0]
                             for row in probs]
                    window=self._window(panel,ti)
                    x,sid,mid,aid,mask=window[:5]
                    stamp=str(panel.dates[ti]); validation=self.validation_active
                    rows=[]
                    pending_inputs={"features":x[0].numpy().astype(np.float16),
                        "symbol_ids":sid[0].numpy(),"market_ids":mid[0].numpy(),
                        "asset_ids":aid[0].numpy(),"valid_mask":mask[0].numpy(),
                        "market_context":window[5][0].numpy().astype(np.float16) if len(window)>5 else None}
                    portfolio_inputs={**pending_inputs,
                        "portfolio_state":np.asarray(pstate,dtype=np.float16),
                        "account_state":np.asarray(astate,dtype=np.float32)}
                    for j,symbol in enumerate(panel.symbols):
                        # A closed venue may not print a bar at the current
                        # global timestamp. Use its forward-filled last quote
                        # for a visible BUY/HOLD/SELL decision, while the
                        # observed mask remains untouched for reward maturity.
                        if not panel.observed[:ti + 1, j].any(): continue
                        action=int(actions[j])
                        behavior_log_prob=float(np.log(max(float(probs[j,action]),1e-12)))
                        previous=self.positions.get(symbol,0)
                        if panel.observed[ti,j]:
                            pending.append({**pending_inputs,"index":ti,"symbol_index":j,"action":action,"previous_position":previous,
                              "timestamp":stamp,"symbol":symbol,"entry_price":float(panel.closes[ti,j]),
                              "behavior_log_prob":behavior_log_prob,
                              "bars_elapsed":0,"regime":abs(float(panel.features[ti,j,6])),
                              "is_validation":False,
                              "promotion_holdout":validation,
                              })
                        if paper_enabled and panel.observed[ti,j]:
                            portfolio_pending.append({**portfolio_inputs,"index":ti,"symbol_index":j,"symbol":symbol,
                              "action":action,"timestamp":stamp,
                              "behavior_log_prob":behavior_log_prob,
                              "entry_price":float(panel.closes[ti,j]),"bars_elapsed":0,
                              "regime":abs(float(panel.features[ti,j,6])),
                               "is_validation":False,"promotion_holdout":validation,
                              "equity_before":self.paper_account.normalized_equity(),
                              "symbol_pnl_before":self.paper_account.symbol_net_pnl(symbol),
                              })
                        rows.append({"date":stamp,"symbol":symbol,"action":ACTION_NAMES[action],"value":float(values[j]),
                          "p_sell":float(probs[j,0]),"p_hold":float(probs[j,1]),"p_buy":float(probs[j,2])})
                        self.metrics["decisions"]+=1
                        if paper_enabled:
                            self.positions[symbol]=action-1
                    self.paper_account.queue_decisions(panel,ti,probs,paper_enabled,
                                                       allocation=allocation,actions=actions)
                    self.metrics["pending_experiences"] = len(pending)+len(portfolio_pending)
                    self.replay.save_pending(pending,portfolio_pending)
                    self.paper_account.save()
                    self._collect_candidate_validation(panel,ti)
                    import pandas as pd
                    if rows:
                        pd.DataFrame(rows).to_csv(decisions_path,mode="a",header=not decisions_path.exists(),index=False)
                    self.metrics["observations"]+=1; cursor=stamp
                    self.metrics["last_market_timestamp"]=stamp
                    _atomic_json({"last_timestamp":cursor},cursor_path)
                    _atomic_json({str(k):v for k,v in self.positions.items()},self.state_dir/"live_positions.json")
                try:
                    newest_stat=data_path.stat()
                    newest_signature=(newest_stat.st_size,newest_stat.st_mtime_ns,
                                      getattr(newest_stat,"st_ino",0))
                except OSError:
                    newest_signature=None
                caught_up = (cursor is not None and np.datetime64(cursor) >= panel.dates[-1]
                             and newest_signature == file_signature)
                self.metrics["observation_caught_up"]=bool(caught_up)
                if caught_up:
                    self.metrics["agent_health"]=("observing_with_unmapped_symbols"
                        if self.metrics["unmatched_live_symbol_count"] else "observing")
                else:
                    self.metrics["agent_health"]="catching_up"
                self._write_metrics()
                last_file_signature=file_signature
                self.stop.wait(poll_seconds)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop.set()
            if self.thread.is_alive(): self.thread.join(timeout=300)
            if self.validation_queue_thread.is_alive():
                self.validation_queue_thread.join(timeout=300)
            self.metrics["candidate_training"]=False
            self.metrics["pending_experiences"] = len(pending)+len(portfolio_pending)
            self.replay.save_pending(pending,portfolio_pending)
            self.checkpoint()
            (self.state_dir/"stop.request").unlink(missing_ok=True)

    def run_panel(self,panel:GlobalMarketPanel, max_observations:int|None=None, stride:int=1):
        """Synchronous market observer; learner thread runs concurrently."""
        self.start(); pending=[]; history=[]
        # Chronological split: first 70% can enter replay, next 15% selects a
        # champion, and the final 15% stays sealed for a backtest report.
        cutoff=max(2,int(len(panel.dates)*.70))
        test_start=max(cutoff+1,int(len(panel.dates)*.85))
        self.min_validation_dates=max(2,min(32,int((test_start-cutoff)*.5)))
        count=0
        for ti in range(max(self.window-1,0),len(panel.dates),stride):
            if max_observations is not None and count>=max_observations and ti<cutoff: continue
            pending=self._mature(pending,panel,ti)
            if cutoff<=ti<test_start:
                # A chronological tail is held out from replay/gradient updates.
                # Save point-in-time states and their later market returns for
                # comparing candidate and incumbent on identical unseen bars.
                window=self._window(panel,ti)
                x,sid,mid,aid,mask=window[:5]
                for j in range(len(panel.symbols)):
                    target=self._target_index(panel,ti,j)
                    if not panel.observed[ti,j] or target is None or target>=test_start: continue
                    gross=panel.return_to(ti,target,j)
                    self._append_validation(self.validation,self.validation_dates,str(panel.dates[ti]),
                        (x[0].numpy().astype(np.float16),sid[0].numpy(),mid[0].numpy(),aid[0].numpy(),
                        mask[0].numpy(),j,gross,str(panel.dates[ti]),0,
                        window[5][0].numpy().astype(np.float16) if len(window)>5 else None))
                continue
            if ti>=test_start: continue
            logits,values,_allocation=self._infer(panel,ti); probs=torch.softmax(torch.as_tensor(logits),-1).numpy()
            window=self._window(panel,ti)
            x,sid,mid,aid,mask=window[:5]
            for j in range(len(panel.symbols)):
                target=self._target_index(panel,ti,j)
                if not panel.observed[ti,j] or target is None or target>=cutoff: continue
                action=int(np.argmax(probs[j])); previous=self.positions.get(j,0)
                pending.append({"index":ti,"symbol_index":j,"action":action,"previous_position":previous,
                  "timestamp":str(panel.dates[ti]),"features":x[0].numpy().astype(np.float16),
                  "symbol_ids":sid[0].numpy(),"market_ids":mid[0].numpy(),"asset_ids":aid[0].numpy(),
                  "valid_mask":mask[0].numpy(),
                  "market_context":window[5][0].numpy().astype(np.float16) if len(window)>5 else None})
                history.append({"date":str(panel.dates[ti]),"symbol":panel.symbols[j],"action":ACTION_NAMES[action],
                    "value":float(values[j]),"p_sell":float(probs[j,0]),"p_hold":float(probs[j,1]),"p_buy":float(probs[j,2])})
                self.metrics["decisions"]+=1
                self.positions[j]=action-1
            self.metrics["observations"]+=1; count+=1
        # Complete all pending decisions with known later observations.
        if len(panel.dates): pending=self._mature(pending,panel,len(panel.dates)-1)
        pd_frame=None
        import pandas as pd
        pd_frame=pd.DataFrame(history)
        pd_frame.to_csv(self.state_dir/"decisions.csv",index=False)
        return history

    def evaluate_panel(self,panel:GlobalMarketPanel,model=None,start_index:int|None=None,
                       end_index:int|None=None,stride:int=1)->dict:
        """Evaluate one policy with sequential positions and net-of-cost PnL."""
        model=model or self.champion; start=max(self.window-1,start_index or 0)
        end=len(panel.dates) if end_index is None else min(len(panel.dates),end_index)
        step_returns=[]; rows=[]; action_counts={name:0 for name in ACTION_NAMES}
        gross_total=fee_total=slippage_total=net_total=0.0
        was_training=model.training; model.eval()
        try:
            with torch.inference_mode():
                for ti in range(start,end,stride):
                    contextual=getattr(model,"_stockrl_uses_market_context",False)
                    args=[x.to(self.device) for x in panel.window(ti,self.window,include_context=contextual)]
                    args[0]=args[0].to(dtype=next(model.parameters()).dtype)
                    logits,values=model(*args); actions=logits[0].argmax(-1).cpu().tolist()
                    values=values[0].float().cpu().tolist(); rewards=[]
                    for j,symbol in enumerate(panel.symbols):
                        if not panel.observed[ti,j]: continue
                        target=self._target_index(panel,ti,j)
                        if target is None or target>=end: continue
                        action=int(actions[j])
                        reward,fee_cost,slip_cost=net_action_reward(action,panel.return_to(ti,target,j),
                                                                     0,self.fee,self.slippage)
                        action_counts[ACTION_NAMES[action]]+=1
                        gross=float(action-1)*panel.return_to(ti,target,j)
                        gross_total+=gross; fee_total+=fee_cost; slippage_total+=slip_cost; net_total+=reward
                        rewards.append(reward)
                        rows.append({"date":str(panel.dates[ti]),"symbol":symbol,"action":ACTION_NAMES[action],
                            "gross_pnl_return":gross,"fee_return":fee_cost,"slippage_return":slip_cost,
                            "net_pnl_return":reward,"value":float(values[j])})
                    if rewards: step_returns.append((str(panel.dates[ti])[:10],float(np.mean(rewards))))
        finally:
            model.train(was_training)
        equity=1.0; peak=1.0; max_dd=0.0; daily={}
        for day,ret in step_returns:
            equity*=max(0.0,1+ret); peak=max(peak,equity); max_dd=max(max_dd,(peak-equity)/peak)
            daily[day]=daily.get(day,0.0)+ret
        return {"start_index":start,"end_index":end,"timestamps":len(step_returns),"decisions":len(rows),
            "net_return":net_total,"net_pnl_return_sum":net_total,"compounded_step_return":equity-1,
            "gross_pnl_return_sum":gross_total,
            "fee_return_sum":fee_total,"slippage_return_sum":slippage_total,"max_drawdown":max_dd,
            "action_counts":action_counts,"daily_net_return":daily,"fee_rate":self.fee,
            "slippage_bps":self.slippage*10000,"horizon":str(self.horizon)}

    def backtest_panel(self,panel:GlobalMarketPanel,stride:int=1)->dict:
        """Run the final 15% as a sealed, equal-weight paper backtest."""
        start=max(1,int(len(panel.dates)*.85))
        out=self.evaluate_panel(panel,self.champion,start_index=start,stride=stride)
        import pandas as pd
        # Recreate row-level scores only for the final report CSV.
        rows=[]; model=self.champion; was_training=model.training; model.eval()
        try:
            with torch.inference_mode():
                for ti in range(max(self.window-1,start),len(panel.dates),stride):
                    contextual=getattr(model,"_stockrl_uses_market_context",False)
                    args=[x.to(self.device) for x in panel.window(ti,self.window,include_context=contextual)]
                    args[0]=args[0].to(dtype=next(model.parameters()).dtype)
                    logits,values=model(*args); acts=logits[0].argmax(-1).cpu().tolist()
                    for j,symbol in enumerate(panel.symbols):
                        if not panel.observed[ti,j]: continue
                        target=self._target_index(panel,ti,j)
                        if target is None or target>=len(panel.dates): continue
                        action=int(acts[j])
                        reward,fee_cost,slip_cost=net_action_reward(action,panel.return_to(ti,target,j),0,self.fee,self.slippage)
                        rows.append({"date":str(panel.dates[ti]),"symbol":symbol,"action":ACTION_NAMES[action],
                            "gross_pnl_return":(action-1)*panel.return_to(ti,target,j),"fee_return":fee_cost,
                            "slippage_return":slip_cost,"net_pnl_return":reward,"value":float(values[0,j])})
        finally: model.train(was_training)
        pd.DataFrame(rows).to_csv(self.state_dir/"backtest.csv",index=False)
        self.metrics["backtest"]=out
        return out

    def import_teacher_csv(self, panel:GlobalMarketPanel, path:str|Path, symbol="MSFT", max_index:int|None=None,
                           source_name:str|None=None):
        """Import normalized SELL/HOLD/BUY labels as replayed imitation data."""
        import pandas as pd
        df=pd.read_csv(path)
        if not {"date","action"}.issubset(df.columns): raise ValueError("teacher CSV needs date,action columns")
        source_name=source_name or f"teacher:{Path(path).stem}"
        symbol_to_ix={s:i for i,s in enumerate(panel.symbols)}
        if symbol not in symbol_to_ix: raise ValueError(f"teacher symbol {symbol} is absent from panel")
        date_to_ix={pd.Timestamp(d):i for i,d in enumerate(panel.dates)}
        with self.replay.lock:
            existing={(e.source,e.timestamp,e.symbol_index,e.action) for e in self.replay.items}
        actions={"SELL":0,"HOLD":1,"WAIT":1,"BUY":2}
        added=0
        for row in df.itertuples(index=False):
            date=pd.Timestamp(row.date)
            if date.tzinfo is not None: date=date.tz_convert("UTC").tz_localize(None)
            ti=date_to_ix.get(date)
            if ti is None: ti=date_to_ix.get(date.normalize())
            row_symbol=str(getattr(row,"symbol",symbol)); j=symbol_to_ix.get(row_symbol)
            action=actions.get(str(row.action).upper())
            if ti is None or action is None or j is None or (max_index is not None and ti>=max_index): continue
            key=(source_name,str(date.normalize()),j,action)
            if key in existing: continue
            window=self._window(panel,ti); x,sid,mid,aid,mask=window[:5]
            self.replay.add(Experience(x[0].numpy().astype(np.float16),sid[0].numpy(),mid[0].numpy(),aid[0].numpy(),
                mask[0].numpy(),j,action,0.0,str(date),source_name,0.0,
                market_context=window[5][0].numpy().astype(np.float16) if len(window)>5 else None)); added+=1; existing.add(key)
        return added

    def _pack(self,batch):
        def tensor(key,dtype=None):
            a=np.stack([getattr(x,key) for x in batch]); t=torch.as_tensor(a,device=self.device)
            return t.to(dtype) if dtype else t
        feature_dtype=torch.float16 if self.device.type=="cuda" else torch.float32
        args=(tensor("features",feature_dtype),tensor("symbol_ids",torch.long),tensor("market_ids",torch.long),
              tensor("asset_ids",torch.long),tensor("valid_mask",torch.bool))
        if getattr(self.champion,"_stockrl_uses_market_context",False):
            from .market_training import CONTEXT_FEATURES
            rows=[]
            for experience in batch:
                context=getattr(experience,"market_context",None)
                if context is None:
                    context=np.zeros((self.window,len(CONTEXT_FEATURES)),dtype=np.float16)
                rows.append(context)
            args=args+(torch.as_tensor(np.stack(rows),device=self.device,dtype=torch.float32),)
        p_rows=[]; a_rows=[]
        for experience in batch:
            p_rows.append(experience.portfolio_state if experience.portfolio_state is not None
                          else np.zeros((experience.features.shape[1], 8), dtype=np.float16))
            a_rows.append(experience.account_state if experience.account_state is not None
                          else np.zeros(8, dtype=np.float32))
        return args, torch.as_tensor(np.stack(p_rows),device=self.device,dtype=torch.float32), torch.as_tensor(np.stack(a_rows),device=self.device,dtype=torch.float32)

    def _learner(self):
        while not self.stop.wait(.1):
            if not bool(self.metrics.get("observation_caught_up",False)):
                self.metrics["candidate_skip_reason"]="시장 데이터 따라잡기 완료 전 candidate 학습 대기"
                continue
            new_experiences=int(self.metrics.get("paper_experiences_since_candidate",0))
            self.metrics["candidate_replay_since_last_update"]=new_experiences
            # Do not spend GPU time on legacy isolated-action samples. The
            # candidate trains on realized paper-account outcomes. Promotion
            # uses a separate, future sequential ledger episode.
            if int(self.metrics.get("paper_experiences_seen",0)) < self.min_replay:
                self.metrics["candidate_skip_reason"] = f"paper 경험 {self.metrics.get('paper_experiences_seen',0)}/{self.min_replay}건 대기"
                continue
            if self.replay.trainable_count()<self.min_replay:
                self.metrics["candidate_skip_reason"] = f"학습 가능 경험 {self.replay.trainable_count()}/{self.min_replay}건 대기"
                continue
            if self.candidate is not None:
                self.metrics["candidate_skip_reason"] = (
                    f"순차 paper 검증 {self.validation_bars}/{self.validation_window_bars}개 bar 대기"
                    if self.validation_active else "candidate 학습 진행 중")
                continue
            if time.monotonic()<self.candidate_retry_after:
                self.metrics["candidate_skip_reason"]="candidate 학습 오류 후 메모리 회복 대기"
                continue
            if new_experiences<max(self.min_replay,self.candidate_interval):
                self.metrics["candidate_skip_reason"] = f"다음 candidate까지 새 경험 {new_experiences}/{max(self.min_replay,self.candidate_interval)}건"
                continue
            self.last_train_replay_size=len(self.replay)
            self.metrics["candidate_skip_reason"] = None
            try:
                self._train_candidate()
            except Exception as exc:
                self.metrics["candidate_training"]=False; self.candidate=None
                self.metrics["candidate_errors"]=int(self.metrics.get("candidate_errors",0))+1
                self.metrics["last_candidate_error"]=f"{type(exc).__name__}: {exc}"[:1200]
                exc.__traceback__=None
                del exc
                self._schedule_candidate_retry()
                self._release_cuda_cache()
                try:
                    self._reset_candidate_file()
                    self._write_metrics()
                except Exception: pass

    def _schedule_candidate_retry(self):
        self.candidate_retry_attempts+=1
        delay=min(300.0,15.0*(2**min(self.candidate_retry_attempts-1,5)))
        self.candidate_retry_after=time.monotonic()+delay
        self.metrics["candidate_retry_delay_seconds"]=delay

    def _release_cuda_cache(self):
        gc.collect()
        if self.device.type=="cuda":
            try: torch.cuda.empty_cache()
            except RuntimeError: pass

    def _train_candidate(self):
        from datetime import datetime, timezone
        trigger_experience_count=int(self.metrics.get("paper_experiences_since_candidate",0))
        self.metrics["candidate_training"]=True
        self.metrics["candidate_skip_reason"] = None
        with self.lock: source_champion=self.champion
        candidate_path=self.model_dir/"candidate.pt"
        self.candidate=None
        if getattr(source_champion,"_stockrl_uses_market_context",False):
            from .market_training import ContextConditionedTransformer
            backbone=GlobalMarketTransformer(self.cfg)
            if self.device.type=="cuda": backbone=backbone.half()
            self.candidate=ContextConditionedTransformer(backbone)
            self.candidate.context_policy.float(); self.candidate.context_value.float()
            self.candidate._stockrl_uses_market_context=True
            self.candidate._stockrl_symbol_map=source_champion._stockrl_symbol_map
        else:
            self.candidate=GlobalMarketTransformer(self.cfg)
            if self.device.type=="cuda": self.candidate=self.candidate.half()
        self.candidate=self.candidate.to(self.device)
        # Every candidate starts from the current champion. A rejected candidate
        # is reset and is never the starting point of the next attempt.
        self.candidate.load_state_dict(source_champion.state_dict())
        original=self.champion
        candidate=self.candidate.train(); opt=torch.optim.AdamW(
            candidate.parameters(),lr=self.lr,weight_decay=.01,
            eps=1e-4 if self.device.type=="cuda" else 1e-8,foreach=False)
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            training_baseline_allocated=int(torch.cuda.memory_allocated(self.device))
        else:
            training_baseline_allocated=0
        elapsed=[]
        candidate_samples=0; candidate_sample_keys=set(); used_experiences=[]
        sampled_ids=set()
        self.metrics["last_candidate_optimizer_steps"]=0
        self.metrics["last_candidate_samples_trained"]=0
        self.metrics["candidate_optimizer_steps_target"]=self.updates_per_candidate
        self.metrics["candidate_samples_target"]=self.updates_per_candidate*self.batch_size
        for update_ix in range(self.updates_per_candidate):
            batch=self.replay.sample(self.batch_size,exclude_ids=sampled_ids)
            if not batch: break
            sampled_ids.update(id(e) for e in batch)
            candidate_sample_keys.update((e.source,e.timestamp,e.symbol_index,e.action,e.portfolio_transition)
                                         for e in batch)
            self.metrics["teacher_examples_trained"]+=sum(e.source.startswith("teacher") for e in batch)
            self.metrics["paper_examples_trained"]+=sum(not e.source.startswith("teacher") for e in batch)
            ti=time.perf_counter(); opt.zero_grad(set_to_none=True); valid_samples=0; loss_values=[]
            use_portfolio=getattr(candidate,"_stockrl_uses_market_context",False)
            # Accumulate the requested replay batch as one optimizer update,
            # while holding only one sequence's activations on the 8 GB GPU.
            successful_batch=[]
            for experience in batch:
                args,pstate,astate=self._pack([experience])
                if use_portfolio:
                    logits,values,allocations=candidate(*args,portfolio_state=pstate,
                                                         account_state=astate,return_allocation=True)
                else:
                    logits,values=candidate(*args); allocations=None
                chosen=logits[0,int(experience.symbol_index)].float()
                predicted=values[0,int(experience.symbol_index)].float()
                if not torch.isfinite(chosen).all() or not torch.isfinite(predicted).all():
                    self.metrics["nonfinite_updates"]+=1
                    continue
                action=torch.as_tensor([experience.action],device=self.device,dtype=torch.long)
                action_probs=(1.0-PAPER_EXPLORATION_EPSILON)*torch.softmax(chosen,dim=-1)
                action_probs=action_probs+PAPER_EXPLORATION_EPSILON/3.0
                dist=Categorical(probs=action_probs[None])
                symbol_reward=torch.as_tensor(float(experience.reward),device=self.device,dtype=torch.float32)
                loss=chosen.sum()*0
                if not experience.portfolio_transition:
                    advantage=(symbol_reward-predicted.detach()).clamp(-1,1)
                    if experience.behavior_log_prob is not None:
                        new_log_prob=dist.log_prob(action)[0]
                        old_log_prob=torch.as_tensor(float(experience.behavior_log_prob),
                                                     device=self.device,dtype=torch.float32)
                        ratio=torch.exp((new_log_prob-old_log_prob).clamp(-20,20))
                        clipped=ratio.clamp(.8,1.2)
                        policy_loss=-torch.minimum(ratio*advantage,clipped*advantage)
                    else:
                        # Legacy deterministic actions have no valid behavior
                        # probability; use their outcome only for the critic.
                        policy_loss=chosen.sum()*0
                    loss=(policy_loss
                          +.5*nn.functional.smooth_l1_loss(predicted[None],symbol_reward[None])
                          -.005*dist.entropy().mean())
                else:
                    loss=chosen.sum()*0
                    if experience.portfolio_value_transition:
                        account_reward=torch.as_tensor(float(experience.portfolio_reward or 0.0),
                                                        device=self.device,dtype=torch.float32)
                        portfolio_value=values[0].float().mean()
                        loss=loss+.25*nn.functional.smooth_l1_loss(portfolio_value,account_reward)
                    if allocations is not None:
                        selected=allocations[0,int(experience.symbol_index)].clamp_min(1e-7)
                        contribution=symbol_reward.detach().clamp(-1,1)
                        loss=loss-.10*contribution*torch.log(selected)
                if experience.source.startswith("teacher"):
                    loss=loss+nn.functional.cross_entropy(chosen[None],action)
                if not torch.isfinite(loss):
                    self.metrics["nonfinite_updates"]+=1
                    continue
                (loss/len(batch)).backward()
                valid_samples+=1
                successful_batch.append(experience)
                loss_values.append(float(loss.detach().cpu()))
                del args,pstate,astate,logits,values,allocations,chosen,predicted,dist,loss
            if not valid_samples:
                continue
            nn.utils.clip_grad_norm_(candidate.parameters(),1.0); opt.step()
            candidate_samples+=valid_samples
            used_experiences.extend(successful_batch)
            self.metrics["update_losses"].append(float(np.mean(loss_values)))
            self.metrics["update_losses"]=self.metrics["update_losses"][-2000:]
            if self.device.type=="cuda": torch.cuda.synchronize(self.device)
            elapsed.append(time.perf_counter()-ti); self.steps+=1
        self.metrics["update_seconds"].extend(elapsed); self.metrics["updates"]+=len(elapsed)
        self.metrics["last_candidate_update_seconds"]=float(sum(elapsed))
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
            self.metrics["last_candidate_peak_allocated_bytes"]=int(
                torch.cuda.max_memory_allocated(self.device))
            self.metrics["last_candidate_baseline_allocated_bytes"]=training_baseline_allocated
            self.metrics["last_candidate_peak_reserved_bytes"]=int(
                torch.cuda.max_memory_reserved(self.device))
        self.metrics["last_candidate_optimizer_steps"]=len(elapsed)
        self.metrics["last_candidate_samples_trained"]=candidate_samples
        self.metrics["last_candidate_unique_samples_trained"]=len(candidate_sample_keys)
        self.metrics["candidate_optimizer_steps_target"]=self.updates_per_candidate
        self.metrics["candidate_samples_target"]=self.updates_per_candidate*self.batch_size
        self.metrics["update_seconds"]=self.metrics["update_seconds"][-2000:]
        self.metrics["teacher_mix_probability"]=self.replay.teacher_fraction()
        if not elapsed:
            self.metrics["candidate_training"]=False
            self.metrics["candidate_errors"]=int(self.metrics.get("candidate_errors",0))+1
            self.metrics["last_candidate_error"]="no finite optimizer update; replay retained for retry"
            self._schedule_candidate_retry()
            self.candidate=None
            self._reset_candidate_file()
            self._write_metrics()
            del candidate,opt
            self._release_cuda_cache()
            return
        if not all(torch.isfinite(p).all() for p in candidate.parameters()):
            self.metrics["nonfinite_updates"]+=1
            self.metrics["rejections"]+=1
            self._schedule_candidate_retry()
            self.metrics["last_rejection_utc"]=datetime.now(timezone.utc).isoformat()
            history=self.metrics.setdefault("candidate_gate_history",[])
            history.append({"time_utc":self.metrics["last_rejection_utc"],"applied":False,
                            "reason":"가중치에 계산 불가능한 값이 발생해 적용하지 않음",
                            "candidate_score":None,"champion_score":None})
            self.metrics["candidate_gate_history"]=history[-20:]
            self.metrics["candidate_training"]=False
            self.candidate=None
            self._reset_candidate_file()
            self._write_metrics()
            del candidate,opt
            self._release_cuda_cache()
            return
        delta=sum((candidate.state_dict()[k].float()-v.detach().float()).abs().sum().item()
                  for k,v in original.state_dict().items())
        self.metrics["weight_delta_l1"].append(delta)
        self.metrics["weight_delta_l1"]=self.metrics["weight_delta_l1"][-2000:]
        self.metrics["last_update_utc"]=datetime.now(timezone.utc).isoformat()
        self.state_dir.mkdir(exist_ok=True,parents=True)
        # Candidate optimizer state is intentionally ephemeral across runs;
        # omit it from the artifact so evaluation/recovery only loads weights.
        trained_replay_row_ids=self.replay.row_ids_for(used_experiences)
        save_model(self.model_dir/"candidate.pt",candidate,self.cfg,step=self.steps,temp_dir=self.state_dir)
        self._begin_candidate_validation(candidate,trained_replay_row_ids)
        # Keep replay rows available through validation. A rejected or
        # interrupted candidate must not erase the only copy of its experience.
        self.metrics["candidate_replay_rows_held_for_validation"]=len(trained_replay_row_ids)
        self.metrics["candidate_replay_rows_consumed"]=0
        self.last_train_replay_size=len(self.replay)
        self.metrics["last_train_replay_size"]=self.last_train_replay_size
        self.metrics["paper_experiences_since_candidate"]=max(
            0,int(self.metrics.get("paper_experiences_since_candidate",0))-trigger_experience_count)
        self.metrics["candidate_replay_since_last_update"]=int(
            self.metrics["paper_experiences_since_candidate"])
        self.metrics["candidate_training"]=False
        self.metrics["candidate_stage"]="sequential_paper_validation"
        self.candidate_retry_attempts=0; self.candidate_retry_after=0.0
        self._write_metrics()
        del candidate,opt

    def _begin_candidate_validation(self,candidate,trained_replay_row_ids=None):
        """Start a fresh, future-only comparison in two identical paper ledgers."""
        self.validation_generation+=1
        generation=self.validation_generation
        self.validation_queue_invalid_reason=None
        while True:
            try:
                self.validation_queue.get_nowait()
                self.validation_queue.task_done()
            except queue.Empty:
                break
        self.validation_champion_account.reset()
        self.validation_candidate_account.reset()
        self.validation_start_after=(self.current_market_timestamp or
                                     self.metrics.get("last_market_timestamp"))
        self.validation_bars=0
        self.validation_active=True
        self.candidate=candidate.eval()
        source_sha=self._sha256_file(self.champion_path)
        self.validation_source_sha256=source_sha
        self._atomic_json({"status":"collecting","start_after":self.validation_start_after,
                           "bars":0,"source_champion_sha256":source_sha,
                           "trained_replay_row_ids":sorted(set(trained_replay_row_ids or [])),
                           "replay_rows_finalized":False,
                           "generation":generation,
                           "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())},
                          self.validation_state_path)
        self.metrics["candidate_stage"]="sequential_paper_validation"
        self.metrics["candidate_validation_bars"]=0
        self.metrics["promotion_blocked_reason"]=None

    @staticmethod
    def _atomic_json(value,path):
        temporary=Path(path).with_suffix(Path(path).suffix+".tmp")
        temporary.parent.mkdir(parents=True,exist_ok=True)
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
        os.replace(temporary,path)

    def _collect_candidate_validation(self,panel,index):
        if not self.validation_active or self.candidate is None:
            return
        stamp=str(panel.dates[index])
        if self.validation_start_after and stamp<=str(self.validation_start_after):
            return
        item=(self.validation_generation,panel,int(index),stamp,time.monotonic())
        try:
            self.validation_queue.put_nowait(item)
        except queue.Full:
            self.validation_queue_invalid_reason="검증 대기열이 가득 차 연속 시세를 놓쳤습니다"
            self.metrics["candidate_validation_queue_overflows"]=(
                int(self.metrics.get("candidate_validation_queue_overflows",0))+1)
            return
        depth=self.validation_queue.qsize()
        self.metrics["candidate_validation_queue_depth"]=depth
        self.metrics["candidate_validation_queue_max_depth"]=max(
            depth,int(self.metrics.get("candidate_validation_queue_max_depth",0)))

    def _validation_worker(self):
        while not self.stop.is_set():
            try:
                generation,panel,index,stamp,queued_at=self.validation_queue.get(timeout=.25)
            except queue.Empty:
                continue
            try:
                if generation!=self.validation_generation or not self.validation_active:
                    continue
                invalid=self.validation_queue_invalid_reason
                delay=time.monotonic()-queued_at
                self.metrics["candidate_validation_queue_delay_seconds"]=delay
                if invalid is None and delay>self.validation_queue_delay_limit_seconds:
                    invalid=(f"검증 처리 지연 {delay:.1f}초가 허용 한도 "
                             f"{self.validation_queue_delay_limit_seconds:.0f}초를 넘었습니다")
                if invalid is not None:
                    self.validation_queue_invalid_reason=None
                    self._finish_candidate_validation(None,None,invalid)
                    continue
                self._process_candidate_validation(panel,index,generation,stamp)
            except Exception as exc:
                if generation==self.validation_generation and self.validation_active:
                    self._finish_candidate_validation(None,None,
                        f"{type(exc).__name__}: {exc}")
            finally:
                self.validation_queue.task_done()
                self.metrics["candidate_validation_queue_depth"]=self.validation_queue.qsize()

    def _process_candidate_validation(self,panel,index,generation,stamp):
        if generation!=self.validation_generation or not self.validation_active or self.candidate is None:
            return
        champ_account=self.validation_champion_account
        cand_account=self.validation_candidate_account
        if champ_account.state.get("last_timestamp")!=cand_account.state.get("last_timestamp"):
            self._finish_candidate_validation(None,None,"paper ledger timestamps diverged")
            return
        state=json.loads(self.validation_state_path.read_text(encoding="utf-8"))
        ledger_timestamp=champ_account.state.get("last_timestamp")
        if state.get("bars")!=self.validation_bars or state.get("last_timestamp")!=ledger_timestamp:
            self._finish_candidate_validation(None,None,"validation state and account timestamps diverged")
            return
        if ledger_timestamp==stamp:
            return
        if ledger_timestamp is not None and stamp<=ledger_timestamp:
            self._finish_candidate_validation(None,None,"validation market timestamps are not increasing")
            return
        completion_scores=None
        try:
            # Both evaluators start from identical seed cash and see this same
            # chronological bar. Existing paper positions and live orders are
            # never used by this isolated comparison.
            champ_account.process_bar(panel,index,True)
            cand_account.process_bar(panel,index,True)
            window=self._window(panel,index)
            args=[x.to(self.device) for x in window]
            with self.lock:
                champion=self.champion
            accounts=((champion,champ_account),(self.candidate,cand_account))
            for model,account in accounts:
                pstate,astate=account.model_inputs(panel,index)
                model_args=list(args)
                model_args[0]=model_args[0].to(dtype=next(model.parameters()).dtype)
                inference_started=time.perf_counter()
                with torch.inference_mode():
                    if getattr(model,"_stockrl_uses_market_context",False):
                        pt=torch.as_tensor(np.asarray(pstate,dtype=np.float32)[None],device=self.device)
                        at=torch.as_tensor(np.asarray(astate,dtype=np.float32)[None],device=self.device)
                        logits,_,allocation=model(*model_args,portfolio_state=pt,
                                                  account_state=at,return_allocation=True)
                        allocation=allocation[0].float().cpu().numpy()
                    else:
                        logits,_=model(*model_args); allocation=None
                    probabilities=torch.softmax(logits[0].float(),dim=-1).cpu().numpy()
                elapsed=time.perf_counter()-inference_started
                prefix=("champion" if model is champion else "candidate")+"_validation_inference_"
                self.metrics[prefix+"count"]=(int(self.metrics.get(prefix+"count",0))+1)
                self.metrics[prefix+"seconds_total"]=(
                    float(self.metrics.get(prefix+"seconds_total",0.0))+elapsed)
                account.queue_decisions(panel,index,probabilities,True,allocation=allocation)
                account.save()
            self.validation_bars+=1
            self.metrics["candidate_validation_bars"]=self.validation_bars
            self.metrics["candidate_skip_reason"]=(
                f"순차 paper 검증 {self.validation_bars}/{self.validation_window_bars}개 bar 대기")
            state.update({"status":"collecting","start_after":self.validation_start_after,
                          "last_timestamp":stamp,"bars":self.validation_bars,
                          "source_champion_sha256":self.validation_source_sha256,
                          "generation":generation})
            self._atomic_json(state,self.validation_state_path)
            if self.validation_bars>=self.validation_window_bars:
                # KRW and USD returns are normalized by their matching seed
                # cash, so unrelated currencies are never added as raw money.
                candidate_score=cand_account.normalized_equity()-2.0
                champion_score=champ_account.normalized_equity()-2.0
                completion_scores=(candidate_score,champion_score)
        except Exception as exc:
            self._finish_candidate_validation(None,None,
                f"{type(exc).__name__}: {exc}")
            return
        if completion_scores is not None:
            self._finish_candidate_validation(*completion_scores,None)

    def _finish_candidate_validation(self,candidate_score,champion_score,error):
        candidate=self.candidate
        source_sha=self._sha256_file(self.champion_path)
        if error is None and self.validation_bars<self.validation_window_bars:
            error="sequential paper validation window is incomplete"
        if error is None:
            state=json.loads(self.validation_state_path.read_text(encoding="utf-8"))
            if state.get("source_champion_sha256")!=source_sha:
                error="champion changed during candidate validation"
        if error is not None:
            self.metrics["promotion_blocked_reason"]=error
            self.metrics["candidate_validation_error"]=error
            self.metrics["rejections"]+=1
            from datetime import datetime,timezone
            self.metrics["last_rejection_utc"]=datetime.now(timezone.utc).isoformat()
            self.metrics["last_candidate_promoted"]=False
            self._reset_candidate_file()
            self.candidate=None
            self.validation_active=False
            prior_state={}
            try: prior_state=json.loads(self.validation_state_path.read_text(encoding="utf-8"))
            except (OSError,json.JSONDecodeError): pass
            self._atomic_json({**prior_state,"status":"rejected","reason":error,
                               "bars":self.validation_bars,
                               "source_champion_sha256":source_sha,
                               "replay_rows_consumed":0,
                               "replay_rows_finalized":True},self.validation_state_path)
        else:
            state=json.loads(self.validation_state_path.read_text(encoding="utf-8"))
            try:
                promoted=self._commit_candidate(candidate,float(candidate_score),float(champion_score),
                                                self.validation_bars,source_sha,
                                                state.get("trained_replay_row_ids",[]))
            except Exception as exc:
                error=f"{type(exc).__name__}: {exc}"
                self.metrics["promotion_blocked_reason"]=error
                self.metrics["candidate_validation_error"]=error
                self.metrics["rejections"]+=1
                self.metrics["last_candidate_promoted"]=False
                self._reset_candidate_file()
                self.candidate=None
                from datetime import datetime,timezone
                self.metrics["last_rejection_utc"]=datetime.now(timezone.utc).isoformat()
                prior_state={}
                try: prior_state=json.loads(self.validation_state_path.read_text(encoding="utf-8"))
                except (OSError,json.JSONDecodeError): pass
                self._atomic_json({**prior_state,"status":"rejected","reason":error,
                                   "bars":self.validation_bars,
                                   "source_champion_sha256":source_sha,
                                   "replay_rows_consumed":0,
                                   "replay_rows_finalized":True},self.validation_state_path)
            else:
                if promoted:
                    try:
                        consumed=self.replay.discard_row_ids(state.get("trained_replay_row_ids",[]))
                        self.metrics["candidate_replay_rows_consumed"]=consumed
                        state=json.loads(self.validation_state_path.read_text(encoding="utf-8"))
                        state["replay_rows_consumed"]=consumed
                        state["replay_rows_finalized"]=True
                        self._atomic_json(state,self.validation_state_path)
                    except Exception as exc:
                        # The champion and validation state already say promoted.
                        # Leave the rows and recovery marker intact for next start.
                        self.metrics["candidate_replay_cleanup_pending"]=True
                        self.metrics["last_candidate_replay_cleanup_error"]=f"{type(exc).__name__}: {exc}"[:1200]
                else:
                    self.metrics["candidate_replay_rows_consumed"]=0
            self.validation_active=False
        self.metrics["candidate_training"]=False
        self.metrics["candidate_stage"]="waiting"
        self.metrics["candidate_skip_reason"]=None
        self.validation_generation+=1
        self._write_metrics()
        del candidate
        self._release_cuda_cache()

    def _commit_candidate(self,candidate,score_new:float,score_old:float,validation_bars:int,
                          source_champion_sha256:str,trained_replay_row_ids=None)->bool:
        """Stage a valid champion checkpoint, then atomically swap the reader."""
        promote=(validation_bars>=self.validation_window_bars
                 and self._sha256_file(self.champion_path)==source_champion_sha256
                 and should_promote(score_new,score_old,1e-9))
        self.metrics["last_candidate_validation_score"]=score_new
        self.metrics["last_champion_validation_score"]=score_old
        self.metrics["last_candidate_promoted"]=promote
        if promote:
            staged=self.state_dir/"champion.next"
            save_model(staged,candidate,self.cfg,step=self.steps,temp_dir=self.state_dir)
            promoted_sha=self._sha256_file(staged).upper()
            baseline_next=self.state_dir/"promotion_baseline.next"
            self._atomic_json({"sha256":promoted_sha,"set_reason":"candidate passed sequential paper gate"},
                              baseline_next)
            with self.lock:
                if self._sha256_file(self.champion_path)!=source_champion_sha256:
                    raise RuntimeError("champion changed before atomic promotion")
                os.replace(staged,self.champion_path)
                os.replace(baseline_next,self.promotion_baseline_path)
                self.promotion_baseline_sha256=promoted_sha
                self.champion=candidate.eval(); self.metrics["promotions"]+=1
                from datetime import datetime, timezone
                self.metrics["last_promotion_utc"]=datetime.now(timezone.utc).isoformat()
            self.candidate=None
        else:
            self.metrics["rejections"]+=1
            self._reset_candidate_file()
            self.candidate=None
            from datetime import datetime, timezone
            self.metrics["last_rejection_utc"]=datetime.now(timezone.utc).isoformat()
        self.metrics["promotion_blocked_reason"]=None
        history=self.metrics.setdefault("candidate_gate_history",[])
        reason=("sequential paper-account net return improved" if promote else
                "champion changed during comparison" if self._sha256_file(self.champion_path)!=source_champion_sha256 else
                "candidate paper-account net return did not beat champion")
        history.append({"time_utc":self.metrics["last_promotion_utc" if promote else "last_rejection_utc"],
                        "applied":promote,"reason":reason,
                        "candidate_score":float(score_new),"champion_score":float(score_old)})
        self.metrics["candidate_gate_history"]=history[-20:]
        self._atomic_json({"status":"promoted" if promote else "rejected",
                           "bars":validation_bars,"candidate_score":float(score_new),
                           "champion_score":float(score_old),"applied":promote,
                           "trained_replay_row_ids":sorted(set(trained_replay_row_ids or [])),
                           "replay_rows_consumed":0,
                           "replay_rows_finalized":not promote,
                           "source_champion_sha256":source_champion_sha256,
                           "result_champion_sha256":self._sha256_file(self.champion_path),
                           "reason":reason},self.validation_state_path)
        return promote

    def _write_metrics(self):
        proc=psutil.Process(); metrics=dict(self.metrics)
        metrics.update({"parameters":parameter_count(self.champion),"device":str(self.device),
          "replay_persistence":"bounded_sqlite_runtime",
          "promotion_score_mode":"sequential_paper_account_net_return",
          "promotion_uses_sequential_paper_account":True,
          "promotion_gate_ready":not bool(metrics.get("promotion_blocked_reason")),
          "promotion_blocked_reason":metrics.get("promotion_blocked_reason"),
          "promotion_baseline_sha256":self.promotion_baseline_sha256,
          "candidate_learning_enabled":bool(self.metrics.get("observation_caught_up",False)),
          "candidate_every":self.candidate_interval,
          "candidate_min_replay":self.min_replay,
          "candidate_batch_size":self.batch_size,
          "candidate_optimizer_steps_target":self.updates_per_candidate,
          "candidate_samples_target":self.updates_per_candidate*self.batch_size,
          "candidate_min_validation_dates":self.validation_window_bars,
          "candidate_validation_bars":self.validation_bars,
          "candidate_validation_queue_depth":self.validation_queue.qsize(),
          "candidate_validation_queue_capacity":self.validation_queue.maxsize,
          "candidate_validation_queue_delay_limit_seconds":self.validation_queue_delay_limit_seconds,
          "candidate_replay_since_last_update":int(self.metrics.get("candidate_replay_since_last_update",0)),
          "last_train_replay_size":self.last_train_replay_size,
          "candidate_stage":("sequential_paper_validation" if self.validation_active else
                             "training" if self.metrics.get("candidate_training") else "waiting"),
          "champion_model_parameter_bytes":sum(p.numel()*p.element_size() for p in self.champion.parameters()),
          "last_candidate_peak_extra_allocated_bytes":(
              max(0,int(self.metrics["last_candidate_peak_allocated_bytes"])-
                      int(self.metrics["last_candidate_baseline_allocated_bytes"]))
              if self.metrics.get("last_candidate_peak_allocated_bytes") is not None and
                 self.metrics.get("last_candidate_baseline_allocated_bytes") is not None else None),
          "validation_window_dates":len(self.validation_dates),
          "portfolio_validation_window_dates":len(self.portfolio_validation_dates),
          "horizon":str(self.horizon),"fee_rate":self.fee,"slippage_bps":self.slippage*10000,
          "rss_bytes":proc.memory_info().rss,"replay_count":len(self.replay),
          "trainable_replay_count":self.replay.trainable_count(),
          "replay_file_bytes":self.replay.disk_bytes(),
          "champion_path":str(self.champion_path),"candidate_path":str(self.model_dir/"candidate.pt")})
        if self.device.type=="cuda":
            metrics.update({"cuda_device":torch.cuda.get_device_name(self.device),
                "cuda_memory_allocated_bytes":torch.cuda.memory_allocated(self.device),
                "cuda_memory_reserved_bytes":torch.cuda.memory_reserved(self.device),
                "cuda_peak_allocated_bytes":max(int(metrics.get("cuda_peak_allocated_bytes",0)),
                                                  int(torch.cuda.max_memory_allocated(self.device))),
                "cuda_total_memory_bytes":torch.cuda.get_device_properties(self.device).total_memory})
        for key in ("inference_seconds","update_seconds"):
            a=metrics[key]; metrics[key+"_p50"]=float(np.percentile(a,50)) if a else 0
            metrics[key+"_p95"]=float(np.percentile(a,95)) if a else 0
        _atomic_json(metrics,self.state_dir/"metrics.json")

    def checkpoint(self):
        # Do not rewrite champion on shutdown. Only the promotion gate may replace it.
        self._write_metrics()


def benchmark_model(model, panel, window=128, repeats=3, device=None):
    from .core import device_for
    dev=device or device_for(); model.to(dev)
    if dev.type=="cuda": model.half()
    model.eval(); contextual=getattr(model,"_stockrl_uses_market_context",False)
    args=[x.to(dev) for x in panel.window(min(len(panel.dates)-1,window),window,include_context=contextual)]
    args[0]=args[0].to(dtype=next(model.parameters()).dtype)
    samples=[]
    with torch.inference_mode():
        model(*args)
        if dev.type=="cuda": torch.cuda.synchronize(dev)
        for _ in range(repeats):
            if dev.type=="cuda": torch.cuda.synchronize(dev)
            t=time.perf_counter(); model(*args); samples.append(time.perf_counter()-t)
            if dev.type=="cuda":
                torch.cuda.synchronize(dev); samples[-1]=time.perf_counter()-t
    return {"inference_seconds_p50":float(np.percentile(samples,50)),"inference_seconds_p95":float(np.percentile(samples,95)),
            "inference_repeats":repeats}
