"""Serialize writers and tolerate short Windows file-sharing conflicts."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
import threading
import time
from weakref import WeakValueDictionary

_locks = WeakValueDictionary()
_guard = threading.Lock()


def read_json(path: str | Path, fallback=None):
    """Read persisted state; each failure returns a caller-owned default."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return deepcopy(fallback) if fallback is not None else {}


def atomic_json(value, path: str | Path, default=None) -> None:
    path = Path(path)
    with _guard:
        key = str(path.resolve())
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default), encoding="utf-8")
            for attempt in range(8):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(0.025 * (attempt + 1))
        finally:
            temporary.unlink(missing_ok=True)
