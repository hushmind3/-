"""Dedicated PT worker lifecycle. Status reads never import torch or load weights."""
import json
import os
from pathlib import Path
import subprocess
import threading
import psutil
from .resources import ROOT
from ..expert_registry import atomic_json


class TradingMoELifecycle:
    def __init__(self):
        self.artifacts=Path.home()/"Desktop"/"모델"/"heterogeneous-experts"
        self.state=ROOT/"runtime/trading_moe/native_vertical_run"
        self.record=ROOT/"runtime/trading_moe/worker.json"
        self.lock=threading.RLock()

    @staticmethod
    def read(path):
        try:return json.loads(path.read_text(encoding="utf-8"))
        except (OSError,ValueError):return {}

    def process(self):
        record=self.read(self.record)
        try:
            p=psutil.Process(record["pid"])
            return p if p.is_running() and p.create_time()==record["created"] and p.cmdline()==record["command"] else None
        except (psutil.Error,KeyError):return None

    def status(self):
        with self.lock:
            data=self.read(self.state/"worker_status.json")
            process=self.process()
            if process is None:
                previous=data.get("status")
                data["status"]="stopped"
                if previous in ("loading","running","saving"):
                    data["error"]=data.get("error") or "worker가 종료됐습니다. 마지막 저장 상태에서 다시 시작할 수 있습니다."
            elif data.get("pid")!=process.pid:
                # Windows venv's launcher spawns the actual Python worker.
                # Only accept a reported PID in this launcher's process tree.
                try:
                    worker=psutil.Process(int(data.get("pid",0)))
                    if process.pid in [p.pid for p in worker.parents()]:process=worker
                    else:data["status"]="loading"
                except (psutil.Error,ValueError,TypeError):data["status"]="loading"
            data.update(pid=process.pid if process else None,alive=process is not None,
                checkpoint=str(self.artifacts/"TradingMoE.pt"),paper=True)
            try:data["worker_ram_bytes"]=process.memory_info().rss if process else 0
            except psutil.Error:data["worker_ram_bytes"]=0
            checkpoint=self.artifacts/"TradingMoE.pt"
            data["checkpoint_bytes"]=checkpoint.stat().st_size if checkpoint.exists() else 0
            if not data.get("books"):
                from ..paper_account import PaperAccount
                account=self.read(self.state/"paper_account.json")
                if account:
                    paper=PaperAccount(self.state/"paper_account.json",.001,.0001)
                    data["books"]=paper.snapshot()["books"]
                    data["reward_points"]=paper.reward_points()
                    data["fills"]=account.get("fills",[])[-20:]
                    report=self.read(self.state/"report.json")
                    data.setdefault("optimizer_updates",report.get("optimizer_updates",0))
                data.setdefault("parameters",2227295521)
            if process and (self.state/"stop.request").exists():data["stop_requested"]=True
            return data

    def start(self):
        with self.lock:
            if self.process():return {"ok":True,"already_running":True,"state":self.status()}
            checkpoint=self.artifacts/"TradingMoE.pt"
            python=self.artifacts/"venv/Scripts/python.exe"
            if not checkpoint.is_file() or not python.is_file():
                return {"ok":False,"error":"TradingMoE.pt 또는 전용 Python 환경을 찾을 수 없습니다."}
            self.state.mkdir(parents=True,exist_ok=True)
            (self.state/"stop.request").unlink(missing_ok=True)
            previous=self.read(self.state/"worker_status.json")
            atomic_json(self.state/"worker_status.json",{**previous,"status":"loading","error":None,"stop_requested":False})
            command=[str(python),"-u",str(ROOT/"scripts/run_native_vertical_trading.py"),
                "--root",str(self.artifacts),"--state",str(self.state),"--resume","--continuous","--interval","0.1"]
            try:
                with (self.state/"worker.log").open("ab") as log:
                    worker=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
                process=psutil.Process(worker.pid)
                atomic_json(self.record,{"pid":worker.pid,"created":process.create_time(),"command":process.cmdline()})
                return {"ok":True,"state":self.status()}
            except OSError as exc:
                return {"ok":False,"error":str(exc)}

    def stop(self):
        with self.lock:
            if not self.process():return {"ok":True,"already_stopped":True,"state":self.status()}
            (self.state/"stop.request").write_text("save and exit",encoding="utf-8")
            return {"ok":True,"message":"현재 사이클 후 계좌·replay·optimizer·PT를 저장하고 종료합니다.","state":self.status()}
