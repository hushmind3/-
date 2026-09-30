"""Serialize writers and tolerate short Windows file-sharing conflicts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time

_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


def atomic_json(value, path: str | Path, default=None) -> None:
    path = Path(path)
    with _guard:
        lock = _locks.setdefault(str(path.resolve()), threading.Lock())
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
