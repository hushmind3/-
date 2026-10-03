"""Reconnect a restarted web controller to its existing worker processes."""
import json
from pathlib import Path
import psutil
from ..state_io import atomic_json


class AttachedWorker:
    def __init__(self, record):
        self.process = psutil.Process(int(record["pid"]))
        if self.process.create_time()!=record["created"] or self.process.cmdline()!=record["command"]:
            raise ValueError("worker process identity changed")
        self.pid = self.process.pid

    def poll(self):
        try:
            return None if self.process.is_running() and self.process.status()!=psutil.STATUS_ZOMBIE else 0
        except psutil.NoSuchProcess:
            return 0

    def wait(self, timeout=None):
        try:
            return self.process.wait(timeout)
        except psutil.TimeoutExpired as exc:
            import subprocess
            raise subprocess.TimeoutExpired(self.pid,timeout) from exc

    def terminate(self):
        self.process.terminate()

    def kill(self):
        self.process.kill()


def handoff(supervisor):
    records={}
    for name,worker in supervisor.children.items():
        if worker.poll() is None:
            process=psutil.Process(worker.pid)
            records[name]={"pid":worker.pid,"created":process.create_time(),"command":process.cmdline()}
    path=supervisor.runtime/"web_workers.json"
    atomic_json({"mode":supervisor.mode,"profile":str(supervisor.profile),
        "run_requested":supervisor.run_requested,"workers":records},path)


def adopt(supervisor):
    path=supervisor.runtime/"web_workers.json"
    if not path.exists():
        return False
    data=json.loads(path.read_text(encoding="utf-8"))
    if data["profile"] in (None,"None"):
        supervisor.mode=data["mode"]
        path.unlink()
        return False
    profile=Path(data["profile"]).resolve()
    if not profile.is_relative_to(supervisor.runtime.resolve()):
        raise ValueError("worker profile is outside runtime")
    children={}
    for name,record in data["workers"].items():
        if name == "agent":
            # Retired Transformer workers are never re-adopted or restarted.
            (profile/"agent"/"stop.request").parent.mkdir(parents=True,exist_ok=True)
            (profile/"agent"/"stop.request").touch()
            continue
        if name not in {"feed","champion","candidate"}:
            raise ValueError("unknown worker role")
        try:
            worker=AttachedWorker(record)
            if worker.poll() is None:
                children[name]=worker
        except psutil.NoSuchProcess:
            pass
    supervisor.mode=data["mode"]
    supervisor.profile=profile
    supervisor.children=children
    supervisor.run_requested=bool(data["run_requested"])
    path.unlink()
    return supervisor.run_requested
