"""One durable FIFO of rolling-window experiences, including unfinished work.

The cache capacity limits RAM, never the number of unlearned database rows.
Overlapping windows share compressed frames. Only checkpoint-confirmed training
can consume rows; market closures and restarts do not expire them.
"""
from collections import OrderedDict, deque
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import hashlib
import pickle
import random
import sqlite3
import threading
import time
import zlib

import numpy as np


CURRENT_REWARD_VERSION = "symbol_and_portfolio_v5"
ARRAY_NAMES = ("features", "symbol_ids", "market_ids", "asset_ids", "valid_mask",
               "market_context", "multiscale_state", "portfolio_state", "account_state","daily_history","goal_state")
TRAINING_SOURCES = ("paper_account_symbol", "paper_account_portfolio")


class ReplayStorageFull(RuntimeError):
    """Leave unlearned work intact when its storage budget is exhausted."""


class GlobalReplayBuffer:
    # A warning level, never an eviction or write limit for unlearned work.
    storage_warning_bytes = 50 * 1024 * 1024

    def __init__(self, capacity=4096, seed=7, journal_path=None, dual_learning=False):
        self.capacity = max(1, int(capacity))
        self.items = deque(maxlen=min(self.capacity, 128) if journal_path else None)
        self.rng = random.Random(seed)
        self.lock = threading.RLock()
        self.paper_outcomes_seen = 0
        self.journal_path = Path(journal_path) if journal_path else None
        self.row_ids = {}
        self.window_cache = OrderedDict()
        self.frame_cache = OrderedDict()
        self.long_history_cache=OrderedDict()
        self._memory_uses = {}
        self._champion_memory_uses = {}
        self.dual_learning=bool(dual_learning)
        self._account_scores={}
        if self.journal_path:
            from .runtime_backup import restore_sqlite_backup
            restore_sqlite_backup(self.journal_path.resolve())
            self._load_journal()

    def _connect(self):
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.journal_path, timeout=30)
        db.execute("PRAGMA journal_size_limit=524288")
        db.execute("PRAGMA wal_autocheckpoint=128")
        # Undo old page limits. Learning completion, not size, permits deletion.
        db.execute("PRAGMA max_page_count=4294967294")
        return db

    @staticmethod
    def _day(timestamp):
        try:
            stamp = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            return stamp.astimezone(timezone(timedelta(hours=9))).date().isoformat()
        except ValueError:
            return str(timestamp)[:10]

    @staticmethod
    def _window_key(exp):
        digest = hashlib.sha256()
        for name in ARRAY_NAMES:
            value = getattr(exp, name, None)
            digest.update(name.encode())
            if value is None:
                digest.update(b"none")
            else:
                array = np.ascontiguousarray(value)
                digest.update(str(array.dtype).encode())
                digest.update(repr(array.shape).encode())
                digest.update(array.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _metadata(exp):
        names = ("symbol_index", "action", "reward", "timestamp", "source", "regime",
                 "reward_version", "portfolio_reward", "portfolio_transition",
                 "portfolio_value_transition", "forward_return", "behavior_log_prob",
                 "trade_executed", "origin_model", "credit_observations",
                 "bootstrap_window_key", "bootstrap_symbol_index", "bootstrap_discount",
                 "goal_reward_points", "portfolio_goal_reward_points", "goal_terminal", "goal_episode_id",
                 "reward_settlement", "reward_end_timestamp", "reward_quote_timestamp")
        return pickle.dumps({name: getattr(exp, name) for name in names
                             if hasattr(exp, name)}, protocol=5)

    def _store_window(self, db, exp):
        key = self._window_key(exp)
        if db.execute("SELECT 1 FROM windows WHERE key=?", (key,)).fetchone():
            return key
        refs = []
        frame_rows = {}
        for index in range(len(exp.features)):
            frame = {"features": np.ascontiguousarray(exp.features[index]),
                     "valid_mask": np.ascontiguousarray(exp.valid_mask[index]),
                     "market_context": (None if getattr(exp, "market_context", None) is None
                                        else np.ascontiguousarray(exp.market_context[index]))}
            raw = pickle.dumps(frame, protocol=5)
            frame_key = hashlib.sha256(raw).digest()
            refs.append(frame_key)
            frame_rows[frame_key] = zlib.compress(raw, level=1)
        db.executemany("INSERT OR IGNORE INTO frames(key,payload,refs) VALUES(?,?,0)",
                       frame_rows.items())
        db.executemany("UPDATE frames SET refs=refs+1 WHERE key=?",
                       ((ref,) for ref in frame_rows))
        arrays = {name: getattr(exp, name, None) for name in ARRAY_NAMES
                  if name not in ("features", "valid_mask", "market_context")}
        history=arrays.get("daily_history")
        history_key=None
        if history is not None:
            raw=pickle.dumps(history,protocol=5)
            history_key=hashlib.sha256(raw).hexdigest()
            db.execute("INSERT OR IGNORE INTO long_history_inputs VALUES(?,?)",(history_key,zlib.compress(raw,1)))
            arrays["daily_history"]=None
        payload = zlib.compress(pickle.dumps({"encoding": "rolling_frames_v1",
            "frames": refs, "arrays": arrays,"long_history_key":history_key}, protocol=5), level=1)
        db.execute("INSERT INTO windows(key,payload,long_history_key) VALUES(?,?,?)", (key, payload,history_key))
        return key

    def _read_window(self, db, key):
        if key in self.window_cache:
            self.window_cache.move_to_end(key)
            return self.window_cache[key]
        row = db.execute("SELECT payload FROM windows WHERE key=?", (key,)).fetchone()
        if row is None:
            raise ValueError(f"Replay input is missing: {key}")
        stored = pickle.loads(zlib.decompress(row[0]))
        if stored.get("encoding") != "rolling_frames_v1":
            arrays = stored  # Existing journals retain their exact inputs.
        else:
            frames = []
            for frame_key in stored["frames"]:
                frame = self.frame_cache.get(frame_key)
                if frame is None:
                    row = db.execute("SELECT payload FROM frames WHERE key=?", (frame_key,)).fetchone()
                    if row is None:
                        raise ValueError("Replay rolling frame is missing")
                    frame = pickle.loads(zlib.decompress(row[0]))
                    self.frame_cache[frame_key] = frame
                    if len(self.frame_cache) > 512:
                        self.frame_cache.popitem(last=False)
                frames.append(frame)
            arrays = dict(stored["arrays"])
            arrays["features"] = np.stack([frame["features"] for frame in frames])
            arrays["valid_mask"] = np.stack([frame["valid_mask"] for frame in frames])
            arrays["market_context"] = (None if frames[0]["market_context"] is None else
                np.stack([frame["market_context"] for frame in frames]))
        history_key=stored.get("long_history_key")
        if history_key:
            if history_key not in self.long_history_cache:
                row=db.execute("SELECT payload FROM long_history_inputs WHERE key=?",(history_key,)).fetchone()
                if row is None:
                    raise ValueError("unlearned long-history input is missing")
                self.long_history_cache[history_key]=pickle.loads(zlib.decompress(row[0]))
                if len(self.long_history_cache)>2:
                    self.long_history_cache.popitem(last=False)
            arrays["daily_history"]=self.long_history_cache[history_key]
        self.window_cache[key] = arrays
        if len(self.window_cache) > 4:
            self.window_cache.popitem(last=False)
        return arrays

    @staticmethod
    def _training_eligible(metadata):
        return (metadata.get("source", "").startswith("teacher") or
                (metadata.get("source") in TRAINING_SOURCES and
                 metadata.get("reward_version") == CURRENT_REWARD_VERSION))

    def _load_journal(self):
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS windows (key TEXT PRIMARY KEY,payload BLOB NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS long_history_inputs (key TEXT PRIMARY KEY,payload BLOB NOT NULL)")
            if "long_history_key" not in {r[1] for r in db.execute("PRAGMA table_info(windows)")}:
                db.execute("ALTER TABLE windows ADD COLUMN long_history_key TEXT")
            db.execute("CREATE TABLE IF NOT EXISTS experiences (id INTEGER PRIMARY KEY AUTOINCREMENT,window_key TEXT NOT NULL,metadata BLOB NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS pending_records (kind TEXT NOT NULL,record_key TEXT NOT NULL,metadata BLOB NOT NULL,PRIMARY KEY(kind,record_key))")
            db.execute("CREATE TABLE IF NOT EXISTS market_observations (stamp TEXT PRIMARY KEY,window_key TEXT NOT NULL,payload BLOB NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS observer_state (name TEXT PRIMARY KEY,payload BLOB NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS origin_counts (role TEXT PRIMARY KEY,created INTEGER NOT NULL DEFAULT 0)")
            db.execute("CREATE TABLE IF NOT EXISTS observation_counts (name TEXT PRIMARY KEY,total INTEGER NOT NULL DEFAULT 0)")
            db.execute("CREATE TABLE IF NOT EXISTS frames (key BLOB PRIMARY KEY,payload BLOB NOT NULL,refs INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS portfolio_value_stamps (timestamp TEXT PRIMARY KEY)")
            db.execute("CREATE TABLE IF NOT EXISTS daily_learning (day TEXT PRIMARY KEY,enqueued INTEGER NOT NULL DEFAULT 0,first_trained INTEGER NOT NULL DEFAULT 0,completed INTEGER NOT NULL DEFAULT 0,exposures INTEGER NOT NULL DEFAULT 0)")
            columns = {row[1] for row in db.execute("PRAGMA table_info(experiences)")}
            additions = {"timestamp": "TEXT", "day": "TEXT", "eligible": "INTEGER DEFAULT 0",
                         "training_uses": "INTEGER NOT NULL DEFAULT 0", "error": "TEXT",
                         "experience_key": "TEXT", "champion_training_uses":"INTEGER NOT NULL DEFAULT 0",
                         "bootstrap_window_key":"TEXT"}
            with db:
                for name, declaration in additions.items():
                    if name not in columns:
                        db.execute(f"ALTER TABLE experiences ADD COLUMN {name} {declaration}")
                for row_id, blob in db.execute("SELECT id,metadata FROM experiences WHERE timestamp IS NULL").fetchall():
                    metadata = pickle.loads(blob)
                    db.execute("UPDATE experiences SET timestamp=?,day=?,eligible=? WHERE id=?",
                        (metadata["timestamp"], self._day(metadata["timestamp"]),
                         int(self._training_eligible(metadata)), row_id))
                db.execute("CREATE UNIQUE INDEX IF NOT EXISTS experience_key_idx ON experiences(experience_key)")
                db.execute("CREATE INDEX IF NOT EXISTS experience_fifo_idx ON experiences(eligible,error,training_uses,timestamp,id)")
                self._ensure_performance_indexes(db)
                db.execute("INSERT OR IGNORE INTO daily_learning(day,enqueued) SELECT day,COUNT(*) FROM experiences GROUP BY day")
                db.execute("CREATE TABLE IF NOT EXISTS replay_settings(key TEXT PRIMARY KEY,value TEXT)")
                if self.dual_learning and not db.execute("SELECT 1 FROM replay_settings WHERE key='dual_learning'").fetchone():
                    # Remaining partially learned rows now require both models.
                    # Completed history before this change retains its original meaning.
                    for day,count in db.execute("SELECT day,COUNT(*) FROM experiences WHERE training_uses>0 GROUP BY day").fetchall():
                        db.execute("UPDATE daily_learning SET first_trained=MAX(0,first_trained-?) WHERE day=?",(count,day))
                    db.execute("INSERT INTO replay_settings VALUES('dual_learning','1')")
            # Load metadata and at most a small RAM cache. No startup pruning.
            with db:
                for (blob,) in db.execute("SELECT metadata FROM experiences"):
                    metadata=pickle.loads(blob)
                    if metadata.get("portfolio_value_transition"):
                        db.execute("INSERT OR IGNORE INTO portfolio_value_stamps VALUES(?)",(metadata["timestamp"],))
            rows = db.execute("SELECT id,window_key,metadata,training_uses FROM experiences ORDER BY id LIMIT ?",
                              (self.items.maxlen,)).fetchall()
            for row in rows:
                self._decode(db, row)

    def _decode(self, db, row):
        from .experience import Experience
        row_id, key, blob, uses = row
        metadata = pickle.loads(blob)
        if metadata.get("portfolio_transition") and "portfolio_value_transition" not in metadata:
            metadata["portfolio_value_transition"] = False
        exp = Experience(**self._read_window(db, key), **metadata)
        if exp.bootstrap_window_key:
            exp._bootstrap_inputs=self._read_window(db,exp.bootstrap_window_key)
        exp._replay_row_id = int(row_id)
        exp._replay_window_key = key
        exp._replay_training_uses = int(uses)
        self.items.append(exp)
        self.row_ids[id(exp)] = int(row_id)
        self._prune_row_ids()
        return exp

    def _prune_row_ids(self):
        live = {id(row) for row in self.items}
        self.row_ids = {key: value for key, value in self.row_ids.items() if key in live}

    def _collect_unused_windows(self, db):
        pending_keys = set()
        pending_stamps=set()
        for (blob,) in db.execute("SELECT metadata FROM pending_records"):
            metadata=pickle.loads(blob)
            pending_stamps.add(metadata.get("timestamp"))
            if metadata.get("origin_model")=="candidate":
                pending_stamps.add("candidate|"+metadata.get("timestamp",""))
            key = metadata.get("_window_key")
            if key:
                pending_keys.add(key)
        pending_keys.update(row[0] for row in db.execute("SELECT window_key FROM market_observations"))
        unused = db.execute("SELECT key,payload FROM windows WHERE key NOT IN (SELECT window_key FROM experiences) AND key NOT IN (SELECT bootstrap_window_key FROM experiences WHERE bootstrap_window_key IS NOT NULL)").fetchall()
        for key, blob in unused:
            if key in pending_keys:
                continue
            stored = pickle.loads(zlib.decompress(blob))
            if stored.get("encoding") == "rolling_frames_v1":
                db.executemany("UPDATE frames SET refs=refs-1 WHERE key=?",
                               ((ref,) for ref in set(stored["frames"])))
            db.execute("DELETE FROM windows WHERE key=?", (key,))
            self.window_cache.pop(key, None)
        db.execute("DELETE FROM frames WHERE refs<=0")
        db.execute("DELETE FROM long_history_inputs WHERE key NOT IN (SELECT long_history_key FROM windows WHERE long_history_key IS NOT NULL)")
        for (stamp,) in db.execute("SELECT timestamp FROM portfolio_value_stamps WHERE timestamp NOT IN (SELECT timestamp FROM experiences)").fetchall():
            if stamp not in pending_stamps:
                db.execute("DELETE FROM portfolio_value_stamps WHERE timestamp=?",(stamp,))

    def add(self, exp, pending_ack=None):
        self.add_many([exp], pending_ack)

    def add_many(self, experiences, pending_ack=None, pending_acks=()):
        with self.lock:
            if not self.journal_path:
                for exp in experiences:
                    successor=getattr(exp,"_bootstrap_experience",None)
                    if successor is not None:
                        exp.bootstrap_window_key=self._window_key(successor)
                        exp._bootstrap_inputs={name:getattr(successor,name,None) for name in ARRAY_NAMES}
                self.items.extend(experiences)
                return
            saved=[]
            # All symbol outcomes from one market state share immutable inputs.
            # Hash/store each input once within this transaction, including its
            # successor, instead of re-hashing five years of history per symbol.
            stored_inputs={}
            def store_input(db, exp):
                identity=tuple(id(getattr(exp,name,None)) for name in ARRAY_NAMES)
                if identity not in stored_inputs:
                    stored_inputs[identity]=self._store_window(db,exp)
                return stored_inputs[identity]
            try:
                with closing(self._connect()) as db, db:
                    for exp in experiences:
                        key=store_input(db,exp)
                        successor=getattr(exp,"_bootstrap_experience",None)
                        if successor is not None:
                            exp.bootstrap_window_key=store_input(db,successor)
                            exp._bootstrap_inputs={name:getattr(successor,name,None) for name in ARRAY_NAMES}
                        symbol_id=int(exp.symbol_ids[exp.symbol_index])
                        identity_fields=(exp.source,exp.timestamp,symbol_id,exp.action)
                        if getattr(exp,"origin_model","champion")!="champion":
                            identity_fields=(*identity_fields,exp.origin_model)
                        identity=hashlib.sha256(repr(identity_fields).encode()).hexdigest()
                        cursor=db.execute("INSERT OR IGNORE INTO experiences(window_key,metadata,timestamp,day,eligible,experience_key,bootstrap_window_key) VALUES(?,?,?,?,?,?,?)",
                            (key,self._metadata(exp),exp.timestamp,self._day(exp.timestamp),
                             int(self._training_eligible(vars(exp))),identity,exp.bootstrap_window_key))
                        if cursor.rowcount:
                            saved.append((exp,int(cursor.lastrowid),key))
                            origin=getattr(exp,"origin_model","champion")
                            db.execute("INSERT INTO origin_counts(role,created) VALUES(?,1) ON CONFLICT(role) DO UPDATE SET created=created+1",(origin,))
                            db.execute("INSERT INTO daily_learning(day,enqueued) VALUES(?,1) ON CONFLICT(day) DO UPDATE SET enqueued=enqueued+1",
                                       (self._day(exp.timestamp),))
                            if exp.portfolio_value_transition:
                                value_stamp=(exp.timestamp if origin=="champion" else origin+"|"+exp.timestamp)
                                db.execute("INSERT OR IGNORE INTO portfolio_value_stamps VALUES(?)",(value_stamp,))
                    if pending_ack:
                        db.execute("DELETE FROM pending_records WHERE kind=? AND record_key=?",pending_ack)
                    db.executemany("DELETE FROM pending_records WHERE kind=? AND record_key=?",pending_acks)
                for exp,row_id,key in saved:
                    exp._replay_row_id=row_id
                    exp._replay_window_key=key
                    exp._replay_training_uses=0
                    self.items.append(exp)
                    self.row_ids[id(exp)]=row_id
                self._prune_row_ids()
            except sqlite3.OperationalError as exc:
                if "full" in str(exc).lower():
                    raise ReplayStorageFull("Replay disk is full; unlearned rows were retained.") from exc
                raise

    def has_portfolio_value(self,timestamp,origin="champion"):
        if not self.journal_path:
            return any(row.portfolio_value_transition and row.timestamp==timestamp and getattr(row,"origin_model","champion")==origin for row in self.items)
        with self.lock, closing(self._connect()) as db:
            key=timestamp if origin=="champion" else origin+"|"+timestamp
            return db.execute("SELECT 1 FROM portfolio_value_stamps WHERE timestamp=?",(key,)).fetchone() is not None

    def enqueue_market_observation(self, panel, paper_enabled, uniforms):
        """One durable FIFO observation; retained until Candidate commits its account."""
        if not self.journal_path:
            raise ValueError("mandatory observation needs the shared replay journal")
        stamp=str(panel.dates[-1])
        arrays={name:getattr(panel,name,None) for name in ARRAY_NAMES}
        arrays["features"]=panel.features.astype(np.float16)
        arrays["valid_mask"]=panel.observed
        arrays["multiscale_state"]=panel.multiscale.astype(np.float16)
        arrays["market_context"]=panel.market_context.astype(np.float16) if panel.market_context is not None else None
        arrays["daily_history"]=getattr(panel,"daily_history",None)
        metadata={"dates":panel.dates,"symbols":panel.symbols,"groups":panel.groups,
            "closes":panel.closes,"ever_observed":panel.ever_observed,
            "paper_enabled":paper_enabled,"uniforms":uniforms}
        with self.lock, closing(self._connect()) as db, db:
            if db.execute("SELECT 1 FROM market_observations WHERE stamp=?",(stamp,)).fetchone():
                return
            key=self._store_window(db,SimpleNamespace(**arrays))
            db.execute("INSERT INTO market_observations VALUES(?,?,?)",
                (stamp,key,zlib.compress(pickle.dumps(metadata,protocol=5),1)))
            db.execute("INSERT INTO observation_counts VALUES('common',1) ON CONFLICT(name) DO UPDATE SET total=total+1")

    def next_market_observation(self):
        with self.lock, closing(self._connect()) as db:
            row=db.execute("SELECT stamp,window_key,payload FROM market_observations ORDER BY stamp LIMIT 1").fetchone()
            if not row:
                return None
            data=pickle.loads(zlib.decompress(row[2])); data.update(self._read_window(db,row[1]))
            data["observed"]=data.pop("valid_mask");data["multiscale"]=data.pop("multiscale_state")
            return row[0],data

    def market_observation_stats(self):
        if not self.journal_path:
            return {"pending":0,"oldest":None}
        with self.lock, closing(self._connect()) as db:
            count,oldest=db.execute("SELECT COUNT(*),MIN(stamp) FROM market_observations").fetchone()
            totals=dict(db.execute("SELECT name,total FROM observation_counts"))
            origins=dict(db.execute("SELECT role,created FROM origin_counts"))
            return {"pending":count,"oldest":oldest,"common":totals.get("common",0),
                "candidate_completed":totals.get("candidate",0),"experience_origins":origins}

    def commit_observer(self, stamp, account_state, pending):
        with self.lock, closing(self._connect()) as db, db:
            self._save_pending_rows(db,"candidate_portfolio",pending)
            db.execute("INSERT OR REPLACE INTO observer_state VALUES('candidate',?)",
                (pickle.dumps(account_state,protocol=5),))
            deleted=db.execute("DELETE FROM market_observations WHERE stamp=?",(stamp,)).rowcount
            if deleted:
                db.execute("INSERT INTO observation_counts VALUES('candidate',1) ON CONFLICT(name) DO UPDATE SET total=total+1")

    def save_pending_kind(self,kind,pending):
        with self.lock, closing(self._connect()) as db, db:
            self._save_pending_rows(db,kind,pending)

    def record_account_score(self,role,stamp,episode,points):
        """Persist one score delta per actual timestamp; restart is not reward."""
        def record(previous):
            if previous and previous["episode"]==episode and previous["timestamp"]>=stamp:
                return previous
            same=bool(previous and previous["episode"]==episode)
            return {"timestamp":stamp,"episode":episode,"points":dict(points),
                "change":({c:float(points[c])-float(previous["points"][c]) for c in points}
                          if same else None)}
        with self.lock:
            if not self.journal_path:
                value=record(self._account_scores.get(role));self._account_scores[role]=value
                return value
            with closing(self._connect()) as db,db:
                key="reward_score:"+role
                row=db.execute("SELECT payload FROM observer_state WHERE name=?",(key,)).fetchone()
                value=record(pickle.loads(row[0]) if row else None)
                db.execute("INSERT OR REPLACE INTO observer_state VALUES(?,?)",
                    (key,pickle.dumps(value,protocol=5)))
                return value

    def set_observer_account(self,account_state):
        with self.lock, closing(self._connect()) as db, db:
            db.execute("INSERT OR REPLACE INTO observer_state VALUES('candidate',?)",
                (pickle.dumps(account_state,protocol=5),))

    def observer_account(self):
        if not self.journal_path:
            return None
        with self.lock, closing(self._connect()) as db:
            row=db.execute("SELECT payload FROM observer_state WHERE name='candidate'").fetchone()
            return pickle.loads(row[0]) if row else None

    def _save_pending_rows(self,db,kind,rows):
        keep=set(); window_by_arrays={}
        for item in rows:
            key=f"{item.get('timestamp','')}|{item.get('symbol',item.get('symbol_index',''))}"
            keep.add(key)
            metadata={name:value for name,value in item.items() if name not in ARRAY_NAMES}
            if "features" in item:
                identity=tuple(id(item.get(name)) for name in ARRAY_NAMES)
                if identity not in window_by_arrays:
                    window_by_arrays[identity]=self._store_window(db,SimpleNamespace(**item))
                metadata["_window_key"]=window_by_arrays[identity]
            db.execute("INSERT OR REPLACE INTO pending_records VALUES(?,?,?)",
                (kind,key,pickle.dumps(metadata,protocol=5)))
        for (key,) in db.execute("SELECT record_key FROM pending_records WHERE kind=?",(kind,)).fetchall():
            if key not in keep:
                db.execute("DELETE FROM pending_records WHERE kind=? AND record_key=?",(kind,key))

    def save_pending(self, regular, portfolio):
        if not self.journal_path:
            return
        with self.lock, closing(self._connect()) as db, db:
            window_by_arrays={}
            for kind, rows in (("regular", regular), ("portfolio", portfolio)):
                keep = set()
                for item in rows:
                    key = f"{item.get('timestamp','')}|{item.get('symbol',item.get('symbol_index',''))}"
                    keep.add(key)
                    metadata = {name: value for name, value in item.items() if name not in ARRAY_NAMES}
                    if "features" in item:
                        array_identity=tuple(id(item.get(name)) for name in ARRAY_NAMES)
                        if array_identity not in window_by_arrays:
                            window_by_arrays[array_identity]=self._store_window(db,SimpleNamespace(**item))
                        metadata["_window_key"]=window_by_arrays[array_identity]
                    db.execute("INSERT OR REPLACE INTO pending_records VALUES(?,?,?)",
                               (kind, key, pickle.dumps(metadata, protocol=5)))
                for (key,) in db.execute("SELECT record_key FROM pending_records WHERE kind=?", (kind,)).fetchall():
                    if key not in keep:
                        db.execute("DELETE FROM pending_records WHERE kind=? AND record_key=?", (kind, key))

    def load_pending(self, kind):
        if not self.journal_path:
            return []
        with self.lock, closing(self._connect()) as db:
            out = []
            for (blob,) in db.execute("SELECT metadata FROM pending_records WHERE kind=? ORDER BY record_key", (kind,)):
                metadata = pickle.loads(blob)
                key = metadata.pop("_window_key", None)
                if key:
                    metadata.update(self._read_window(db, key))
                out.append(metadata)
            return out

    def acknowledge_pending(self, kind, decision):
        if not self.journal_path:
            return
        key = (decision if isinstance(decision, str) else
               f"{decision.get('timestamp','')}|{decision.get('symbol',decision.get('symbol_index',''))}")
        with self.lock, closing(self._connect()) as db, db:
            db.execute("DELETE FROM pending_records WHERE kind=? AND record_key=?", (kind, key))

    def pending_batch(self, batch_size, passes=1, exclude_row_ids=(), learner="candidate", *, timestamps=None):
        if learner not in ("candidate","champion"):
            raise ValueError("unknown learner")
        uses_by_id=self._champion_memory_uses if learner=="champion" else self._memory_uses
        column="champion_training_uses" if learner=="champion" else "training_uses"
        with self.lock:
            if not self.journal_path:
                rows = [row for row in self.items if self._training_eligible(vars(row)) and
                        uses_by_id.get(id(row), 0) < passes and id(row) not in exclude_row_ids
                        and (timestamps is None or row.timestamp in timestamps)]
                for row in rows:
                    row._replay_training_uses=uses_by_id.get(id(row),0)
                return sorted(rows, key=lambda row: (row.timestamp, uses_by_id.get(id(row), 0)))[:batch_size]
            with closing(self._connect()) as db:
                excluded = sorted(set(exclude_row_ids))
                clause = (" AND id NOT IN (" + ",".join("?" for _ in excluded) + ")") if excluded else ""
                parameters = [passes, *excluded]
                if timestamps is not None:
                    import json
                    # Filter before LIMIT: missing evidence cannot starve ready MoE work.
                    clause += " AND timestamp IN (SELECT value FROM json_each(?))"
                    parameters.append(json.dumps(list(timestamps)))
                rows = db.execute(f"SELECT id,window_key,metadata,{column} FROM experiences WHERE eligible=1 AND error IS NULL AND {column}<?" +
                    clause + f" ORDER BY timestamp,{column},id LIMIT ?", (*parameters, batch_size)).fetchall()
                return [self._decode(db, row) for row in rows]

    def sample(self, batch_size, exclude_ids=None):
        excluded = [self.row_ids.get(key, key) for key in (exclude_ids or ())]
        return self.pending_batch(batch_size, exclude_row_ids=excluded)

    def retained_row_count(self, row_ids):
        """Rows updated in RAM, retained until their checkpoint is durable."""
        if not row_ids or not self.journal_path:
            return 0
        import json
        with self.lock, closing(self._connect()) as db:
            return db.execute("SELECT COUNT(*) FROM experiences WHERE id IN "
                "(SELECT value FROM json_each(?))",(json.dumps([int(k) for k in row_ids]),)).fetchone()[0]

    @staticmethod
    def _ensure_performance_indexes(db):
        for name, columns in (
            ("experience_window_idx", "window_key"),
            ("experience_successor_idx", "bootstrap_window_key"),
            ("experience_timestamp_idx", "timestamp"),
            ("experience_champion_fifo_idx", "eligible,error,champion_training_uses,timestamp,id"),
        ):
            db.execute(f"CREATE INDEX IF NOT EXISTS {name} ON experiences({columns})")

    def acknowledge_training(self, uses_by_id, passes=1, learner="candidate"):
        """Consume only after every required learner confirms its checkpoint."""
        if learner not in ("candidate","champion"):
            raise ValueError("unknown learner")
        column="champion_training_uses" if learner=="champion" else "training_uses"
        if not self.journal_path:
            memory=self._champion_memory_uses if learner=="champion" else self._memory_uses
            memory.update(uses_by_id)
            self.items = deque(row for row in self.items if self._memory_uses.get(id(row),0)<passes or
                (self.dual_learning and self._champion_memory_uses.get(id(row),0)<passes))
            return 0
        completed = 0
        deleted_ids=set()
        with self.lock, closing(self._connect()) as db, db:
            targets = {int(row_id): int(target) for row_id, target in uses_by_id.items()}
            rows = {}
            keys = list(targets)
            for offset in range(0, len(keys), 500):
                chunk = keys[offset:offset+500]
                marks = ",".join("?" for _ in chunk)
                rows.update((row[0], row[1:]) for row in db.execute(
                    f"SELECT id,training_uses,champion_training_uses,day FROM experiences WHERE id IN ({marks})", chunk))
            updates = []
            daily = {}
            for row_id, target in targets.items():
                row = rows.get(row_id)
                previous=(row[1] if learner=="champion" else row[0]) if row else 0
                if row is None or int(target) <= previous:
                    continue
                candidate_uses,champion_uses,day=row
                before=min(candidate_uses,champion_uses) if self.dual_learning else candidate_uses
                if learner=="champion": champion_uses=int(target)
                else: candidate_uses=int(target)
                after=min(candidate_uses,champion_uses) if self.dual_learning else candidate_uses
                done=int(after>=passes)
                counters = daily.setdefault(day, [0, 0, 0])
                counters[0] += int(before==0 and after>0)
                counters[1] += done
                counters[2] += int(target)-previous
                updates.append((int(target), int(row_id)))
                if done:
                    deleted_ids.add(int(row_id))
                    completed += 1
            db.executemany(f"UPDATE experiences SET {column}=? WHERE id=?", updates)
            db.executemany("UPDATE daily_learning SET first_trained=first_trained+?,completed=completed+?,exposures=exposures+? WHERE day=?",
                ((*values, day) for day, values in daily.items()))
            if completed:
                db.executemany("DELETE FROM experiences WHERE id=?", ((row_id,) for row_id in deleted_ids))
                self._collect_unused_windows(db)
        if completed:
            self.items = deque((row for row in self.items if getattr(row,"_replay_row_id",None) not in deleted_ids),maxlen=min(self.capacity,128))
            self._prune_row_ids()
            self.compact()
        return completed

    def compact(self):
        if not self.journal_path:
            return
        started = time.monotonic()
        with self.lock, closing(self._connect()) as db:
            free = db.execute("PRAGMA freelist_count").fetchone()[0]
            page = db.execute("PRAGMA page_size").fetchone()[0]
            total = db.execute("PRAGMA page_count").fetchone()[0]
            empty = not db.execute("SELECT 1 FROM experiences LIMIT 1").fetchone()
            due = started-getattr(self, "_last_vacuum_time", -300) >= 300
            worthwhile = free*page >= 64*1024*1024 and free >= total//4
            # Completed rows are already deleted. SQLite reuses their free pages.
            # Do not rewrite the entire growing DB for every 256-row batch.
            if free*page > 2*1024*1024 and (empty or (due and worthwhile)):
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                db.execute("VACUUM")
                self._last_vacuum_time = time.monotonic()

    def finalize_completed(self, passes):
        """Apply a changed pass target only to already checkpoint-confirmed rows."""
        if not self.journal_path:
            before=len(self.items)
            self.items=deque(row for row in self.items if self._memory_uses.get(id(row),0)<passes or
                (self.dual_learning and self._champion_memory_uses.get(id(row),0)<passes))
            return before-len(self.items)
        completed=0
        with self.lock, closing(self._connect()) as db, db:
            count_expr="MIN(training_uses,champion_training_uses)" if self.dual_learning else "training_uses"
            predicate=f"eligible=1 AND error IS NULL AND {count_expr}>=?"
            groups=db.execute(f"SELECT day,COUNT(*) FROM experiences WHERE {predicate} GROUP BY day",(passes,)).fetchall()
            for day,count in groups:
                db.execute("UPDATE daily_learning SET completed=completed+? WHERE day=?",(count,day))
                completed+=count
            if completed:
                db.execute(f"DELETE FROM experiences WHERE {predicate}",(passes,))
                self._collect_unused_windows(db)
                # These are decoded caches, never the durable queue.
                self.items.clear(); self.row_ids.clear()
            db.execute("INSERT INTO replay_settings(key,value) VALUES('required_passes',?) "
                       "ON CONFLICT(key) DO UPDATE SET value=excluded.value",(str(passes),))
        if completed: self.compact()
        return completed

    def quarantine(self, row_ids, reason):
        if not self.journal_path:
            return
        with self.lock, closing(self._connect()) as db, db:
            db.executemany("UPDATE experiences SET error=? WHERE id=?", ((reason, int(row_id)) for row_id in row_ids))

    def stats(self, passes=1):
        if not self.journal_path:
            remaining={name:sum(self._training_eligible(vars(row)) and uses.get(id(row),0)<passes for row in self.items)
                for name,uses in (("candidate",self._memory_uses),("champion",self._champion_memory_uses))}
            eligible=sum(self._training_eligible(vars(row)) and (self._memory_uses.get(id(row),0)<passes or
                (self.dual_learning and self._champion_memory_uses.get(id(row),0)<passes)) for row in self.items)
            return {"total": len(self.items), "eligible": eligible, "untrained": eligible,
                    "quarantined":0,"unsupported":0,"daily":[],"model_remaining":remaining,
                    "model_untrained":remaining}
        with self.lock, closing(self._connect()) as db:
            count_expr="MIN(training_uses,champion_training_uses)" if self.dual_learning else "training_uses"
            total, eligible, untrained, quarantine, unsupported = db.execute(
                f"SELECT COUNT(*),COALESCE(SUM(eligible=1 AND error IS NULL AND {count_expr}<?),0),"
                f"COALESCE(SUM(eligible=1 AND error IS NULL AND {count_expr}=0),0),"
                "COALESCE(SUM(error IS NOT NULL),0),COALESCE(SUM(eligible=0),0) FROM experiences", (passes,)).fetchone()
            model_remaining={}; model_untrained={}
            for name,column in (("candidate","training_uses"),("champion","champion_training_uses")):
                model_remaining[name],model_untrained[name]=db.execute(
                    f"SELECT COALESCE(SUM(eligible=1 AND error IS NULL AND {column}<?),0),COALESCE(SUM(eligible=1 AND error IS NULL AND {column}=0),0) FROM experiences",(passes,)).fetchone()
            blocked_by_day=dict(db.execute("SELECT day,COUNT(*) FROM experiences WHERE error IS NOT NULL OR eligible=0 GROUP BY day"))
            blocked_reasons=dict(db.execute("SELECT COALESCE(error,'unsupported reward schema'),COUNT(*) FROM experiences WHERE error IS NOT NULL OR eligible=0 GROUP BY COALESCE(error,'unsupported reward schema')"))
            completed_retained=db.execute(f"SELECT COUNT(*) FROM experiences WHERE eligible=1 AND error IS NULL AND {count_expr}>=?",(passes,)).fetchone()[0]
            remaining_by_day=dict(db.execute(
                f"SELECT day,COUNT(*) FROM experiences WHERE {count_expr}<? OR error IS NOT NULL OR eligible=0 GROUP BY day",
                (passes,)))
            retained_by_day=dict(db.execute("SELECT day,COUNT(*) FROM experiences GROUP BY day"))
            daily = [{"day": day, "enqueued": queued, "first_trained": first,
                      "completed":done,"exposures":exposures,"remaining":max(0,remaining_by_day.get(day,0)-blocked_by_day.get(day,0)),
                      "blocked":blocked_by_day.get(day,0),"remaining_total":remaining_by_day.get(day,0),
                      "removed_without_completion":max(0,queued-done-retained_by_day.get(day,0))}
                     for day, queued, first, done, exposures in db.execute(
                         "SELECT day,enqueued,first_trained,completed,exposures FROM daily_learning ORDER BY day DESC LIMIT 14")]
            oldest = db.execute(f"SELECT MIN(timestamp) FROM experiences WHERE eligible=1 AND error IS NULL AND {count_expr}<?", (passes,)).fetchone()[0]
            pending = db.execute("SELECT COUNT(*) FROM pending_records").fetchone()[0]
        return {"total": total, "eligible": eligible, "untrained": untrained,
                "quarantined": quarantine, "unsupported": unsupported, "daily": daily,
                "blocked_reasons":blocked_reasons,"completed_retained":completed_retained,
                "oldest": oldest, "pending": pending, "bytes": self.disk_bytes(),
                "model_remaining":model_remaining,"model_untrained":model_untrained,
                "storage_pressure": self.disk_bytes() >= self.storage_warning_bytes}

    def __len__(self):
        if not self.journal_path:
            return len(self.items)
        with self.lock, closing(self._connect()) as db:
            return db.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]

    def trainable_count(self):
        return self.stats()["eligible"]

    def disk_bytes(self):
        if not self.journal_path:
            return 0
        total=0
        for path in (self.journal_path,Path(str(self.journal_path)+"-wal"),
                     Path(str(self.journal_path)+"-shm")):
            try:
                total+=path.stat().st_size
            except FileNotFoundError:
                # SQLite may remove WAL/SHM between existence and size checks.
                # Their absence is normal; no replay data is discarded here.
                continue
        return total

    def row_ids_for(self, experiences):
        return sorted({getattr(row, "_replay_row_id", self.row_ids.get(id(row), id(row))) for row in experiences})

    def discard_row_ids(self, row_ids):
        ids = set(map(int, row_ids))
        if not ids:
            return 0
        removed = 0
        with self.lock:
            if self.journal_path:
                with closing(self._connect()) as db, db:
                    for row_id in ids:
                        removed += db.execute("DELETE FROM experiences WHERE id=?", (row_id,)).rowcount
                    self._collect_unused_windows(db)
            self.items = deque((row for row in self.items if getattr(row, "_replay_row_id", self.row_ids.get(id(row))) not in ids),
                               maxlen=self.items.maxlen)
            self._prune_row_ids()
        return removed

    def discard(self, experiences):
        if self.journal_path:
            return self.discard_row_ids(self.row_ids_for(experiences))
        ids = {id(row) for row in experiences}
        before = len(self.items)
        self.items = deque(row for row in self.items if id(row) not in ids)
        return before-len(self.items)

    def teacher_fraction(self):
        return max(0.0, 1.0-self.paper_outcomes_seen/10000.0) if any(row.source.startswith("teacher") for row in self.items) else 0.0

    def note_paper_outcome(self):
        self.paper_outcomes_seen += 1

    def sample_source(self, source):
        return next((row for row in self.items if row.source.startswith(source)), None)

    def load(self, path, manifest_path=None):
        import json
        import torch
        path = Path(path)
        if not path.exists():
            return
        manifest_path = manifest_path or path.with_suffix(".manifest.json")
        paths = [path]
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            root = path.parent/(path.stem+"_chunks")/manifest["generation"]
            paths = [root/name for name in manifest["chunks"]]
        for chunk in paths:
            for exp in torch.load(chunk, map_location="cpu", weights_only=False):
                self.add(exp)
