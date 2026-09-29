"""Create a recoverable project-local snapshot of a market runtime.

SQLite databases are copied with SQLite's online backup API, which includes
committed WAL transactions without copying the live WAL/SHM sidecars. The
default mode refuses to snapshot while the StockRL web/feed/agent is running.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
SIDECAR_SUFFIXES = ("-wal", "-shm")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_counts(connection: sqlite3.Connection) -> dict[str, int]:
    names = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    counts: dict[str, int] = {}
    for name in names:
        quoted = '"' + name.replace('"', '""') + '"'
        counts[name] = connection.execute(
            f"SELECT COUNT(*) FROM {quoted}"
        ).fetchone()[0]
    return counts


def sqlite_backup(source: Path, destination: Path) -> dict[str, object]:
    source_connection = sqlite3.connect(
        f"{source.as_uri()}?mode=ro", uri=True, timeout=30
    )
    destination_connection = sqlite3.connect(destination, timeout=30)
    try:
        source_connection.backup(destination_connection)
        destination_connection.commit()
        source_counts = table_counts(source_connection)
        destination_counts = table_counts(destination_connection)
        integrity = destination_connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]
        if integrity != "ok" or source_counts != destination_counts:
            raise RuntimeError(
                f"SQLite snapshot verification failed for {source}: "
                f"integrity={integrity}, source_rows={source_counts}, "
                f"snapshot_rows={destination_counts}"
            )
        destination_connection.execute("PRAGMA journal_mode=DELETE")
        return {
            "integrity_check": integrity,
            "table_rows": destination_counts,
            "sha256": sha256(destination),
        }
    finally:
        destination_connection.close()
        source_connection.close()


def running_stockrl_processes() -> list[str]:
    found: list[str] = []
    try:
        import psutil

        for process in psutil.process_iter(["pid", "cmdline"]):
            try:
                command = " ".join(process.info.get("cmdline") or [])
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
            lowered = command.lower()
            if "stockrl" in lowered and any(
                name in lowered for name in ("live-feed", "global-online")
            ):
                found.append(f"pid={process.info['pid']} {command}")
    except ImportError:
        pass
    return found


def web_running() -> bool | None:
    try:
        with urlopen("http://127.0.0.1:8766/api/status", timeout=1) as response:
            payload = json.load(response)
    except (OSError, URLError, ValueError):
        return False
    required = ("running", "feed_running", "agent_running")
    if not all(key in payload for key in required):
        return None
    return any(bool(payload[key]) for key in required)


def stable_copy(source: Path, destination: Path, attempts: int = 3) -> str:
    for _ in range(attempts):
        before = source.stat()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        after = source.stat()
        copied = destination.stat()
        source_signature = (before.st_size, before.st_mtime_ns, before.st_ino)
        after_signature = (after.st_size, after.st_mtime_ns, after.st_ino)
        if (source_signature == after_signature
                and copied.st_size == after.st_size):
            copied_hash = sha256(destination)
            if copied_hash == sha256(source) and source.stat().st_mtime_ns == after.st_mtime_ns:
                return copied_hash
        destination.unlink(missing_ok=True)
    raise RuntimeError(f"File kept changing while snapshotting: {source}")


def create_snapshot(market: str, allow_live: bool) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", market):
        raise ValueError("Market must use simple lowercase letters, digits, - or _")

    active_processes = running_stockrl_processes()
    web_state = web_running()
    if not allow_live and (active_processes or web_state is True or web_state is None):
        details = "; ".join(active_processes) or "8766 status is running or unknown"
        raise RuntimeError(
            "Stop the StockRL web/feed/agent before snapshotting, or pass "
            f"--allow-live for per-file consistent backups. Found: {details}"
        )

    source_root = (ROOT / "runtime" / "markets" / market / "live").resolve()
    if not source_root.is_relative_to(ROOT.resolve()) or not source_root.is_dir():
        raise ValueError(f"Runtime source must exist inside the project: {source_root}")

    snapshots = ROOT / "runtime" / "markets" / market / "snapshots"
    if not snapshots.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"Snapshot destination must stay inside the project: {snapshots}")
    snapshots.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final = snapshots / stamp
    staging = snapshots / f".{stamp}.tmp"
    if final.exists() or staging.exists():
        raise FileExistsError(f"Snapshot path already exists: {final}")
    staging.mkdir()

    manifest: dict[str, object] = {
        "market": market,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source_root.relative_to(ROOT)),
        "web_running": web_state,
        "active_processes": active_processes,
        "live_snapshot": bool(allow_live),
        "files": {},
        "excluded": {"model_checkpoints": [], "sqlite_sidecars": []},
        "note": (
            "Each SQLite database is a consistent online backup. Other files "
            "are copied only when their size and modification time stay stable; "
            "a live snapshot is not a cross-file transaction."
        ),
    }
    try:
        source_files = sorted(path for path in source_root.rglob("*") if path.is_file())
        for source in source_files:
            if source.is_symlink():
                raise RuntimeError(f"Refusing to follow runtime link: {source}")
            relative = source.relative_to(source_root)
            destination = staging / relative
            lower_name = source.name.lower()
            if source.suffix.lower() in CHECKPOINT_SUFFIXES:
                manifest["excluded"]["model_checkpoints"].append(str(relative))  # type: ignore[index]
                continue
            if lower_name.endswith(SIDECAR_SUFFIXES):
                manifest["excluded"]["sqlite_sidecars"].append(str(relative))  # type: ignore[index]
                continue
            if source.name.lower().endswith(".sqlite3"):
                destination.parent.mkdir(parents=True, exist_ok=True)
                metadata = sqlite_backup(source, destination)
                manifest["files"][str(relative)] = {  # type: ignore[index]
                    "kind": "sqlite_online_backup", **metadata
                }
            else:
                digest = stable_copy(source, destination)
                manifest["files"][str(relative)] = {  # type: ignore[index]
                    "kind": "file", "bytes": destination.stat().st_size,
                    "sha256": digest,
                }

        (staging / "snapshot_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", default="korea")
    parser.add_argument(
        "--allow-live", action="store_true",
        help="Allow a live snapshot; each SQLite DB stays consistent, but files "
             "as a group may represent slightly different times.",
    )
    args = parser.parse_args()
    destination = create_snapshot(args.market, args.allow_live)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
