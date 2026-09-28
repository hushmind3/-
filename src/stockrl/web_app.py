"""Responsive local web dashboard and process supervisor for StockRL."""
from __future__ import annotations

import csv
from collections import deque
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime, time as day_time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]

PAGE = Path(__file__).with_name("web_dashboard.html").read_text(encoding="utf-8")


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _market_group(item: dict) -> str:
    market=item.get("market",""); asset=item.get("asset_class","")
    if market in ("KRX","KOSDAQ") and asset=="index": return "korea_index"
    if market in ("KRX","KOSDAQ"): return "korea"
    if asset in ("index_future","commodity_future"): return "futures"
    if asset=="yield": return "bonds"
    if asset=="currency": return "fx"
    if asset=="crypto": return "crypto"
    if market=="US": return "us"
    return f"global:{market}"


def _session(group: str) -> str:
    # Session labels are a clock-based guide only; exchange holidays and halts
    # are not available from the current free data providers.
    from dateutil import tz
    if group=="crypto": return "가상자산 · 24시간 · 연중"
    if group.startswith("global:"):
        venue=group.split(":",1)[1]
        specs={"Japan":("Asia/Tokyo","일본 현물 09:00–11:30 / 12:30–15:30"),
               "HongKong":("Asia/Hong_Kong","홍콩 현물 09:30–12:00 / 13:00–16:00"),
               "Germany":("Europe/Berlin","독일 현물 09:00–17:30"),
               "UK":("Europe/London","영국 현물 08:00–16:30")}
        if venue in specs:
            zone,hours=specs[venue]; now=datetime.now(tz.gettz(zone)); open_now=now.weekday()<5
            t=now.time()
            if venue=="Japan": open_now &= day_time(9)<=t<day_time(11,30) or day_time(12,30)<=t<day_time(15,30)
            elif venue=="HongKong": open_now &= day_time(9,30)<=t<day_time(12) or day_time(13)<=t<day_time(16)
            elif venue=="Germany": open_now &= day_time(9)<=t<day_time(17,30)
            else: open_now &= day_time(8)<=t<day_time(16,30)
            return f"{hours} · 현지 {now:%H:%M} · {'장중' if open_now else '장외'} · 휴장일 미반영"
        return "현지 거래소별 시간"
    if group=="fx":
        now=datetime.now(tz.gettz("America/New_York")); wd=now.weekday(); t=now.time()
        opened=(wd==6 and t>=day_time(17)) or (0<=wd<4) or (wd==4 and t<day_time(17))
        return f"FX · 주중 24/5 · 뉴욕 {now:%H:%M} · {'거래 시간대' if opened else '주말 휴장 시간대'}"
    if group in ("futures","bonds","us"):
        now=datetime.now(tz.gettz("America/New_York")); wd=now.weekday(); t=now.time()
        if group=="us": opened=wd<5 and day_time(9,30)<=t<day_time(16)
        elif group=="bonds": opened=wd<5 and day_time(8)<=t<day_time(17)
        else: opened=((wd==6 and t>=day_time(18)) or (0<=wd<4) or (wd==4 and t<day_time(17))) and not day_time(17)<=t<day_time(18)
        hours={"us":"미국 현물 09:30–16:00 ET","bonds":"미 국채지표 08:00–17:00 ET","futures":"선물 주중 23시간(정비시간 제외)"}[group]
        return f"{hours} · 뉴욕 {now:%H:%M} · {'장중' if opened else '장외/정비'} · 휴장일 미반영"
    if group=="korea_index":
        now=datetime.now(tz.gettz("Asia/Seoul")); t=now.time()
        opened=now.weekday()<5 and day_time(9)<=t<day_time(15,30)
        return f"한국 지수 09:00–15:30 KST · 서울 {now:%H:%M} · {'장중' if opened else '장외'} · 휴장일 미반영"
    if group=="korea":
        now=datetime.now(tz.gettz("Asia/Seoul")); t=now.time()
        if now.weekday()>=5: phase="주말 장외"
        elif day_time(8)<=t<day_time(8,50): phase="NXT 프리마켓"
        elif day_time(8,50)<=t<day_time(9): phase="NXT 종료 · KRX 개장 대기"
        elif day_time(9,0,30)<=t<day_time(15,20): phase="KRX 정규장 · NXT 메인마켓"
        elif day_time(9)<=t<day_time(15,30): phase="KRX 정규장"
        elif day_time(15,30)<=t<day_time(15,40): phase="NXT 호가접수 · 체결 대기"
        elif day_time(15,40)<=t<day_time(20): phase="NXT 애프터마켓"
        else: phase="장외"
        return ("KRX 09:00–15:30 · NXT 프리 08:00–08:50 / 메인 09:00:30–15:20 / "
                f"애프터 15:40–20:00 KST · 서울 {now:%H:%M} · {phase} · 휴장일 미반영")
    return "현지 거래소별 시간"


def _market_overview(instruments: list[dict], decisions: list[dict],
                     fresh_symbols: set[str] | None = None) -> list[dict]:
    fresh_symbols = fresh_symbols or set()
    titles={"korea":"한국 주식","korea_index":"한국 지수","us":"미국 주식·NASDAQ·지수",
            "futures":"지수·원자재 선물","bonds":"국채 금리","fx":"환율","crypto":"가상자산"}
    grouped={}
    for item in instruments:
        key=_market_group(item); label=titles.get(key,key.split(":",1)[1]+" 지수" if key.startswith("global:") else key)
        grouped.setdefault(key,{"key":key,"label":label,"symbols":[]})["symbols"].append(item.get("symbol",""))
    for value in grouped.values():
        symbols=set(value["symbols"]); rows=[d for d in decisions if d.get("symbol") in symbols]
        value["count"]=len(symbols); value["session"]=_session(value["key"])
        value["fresh_count"]=len(symbols & fresh_symbols)
        value["latest_decision"]=max((row.get("date", "") for row in rows), default=None)
        value["actions"]={name:sum(1 for row in rows if row.get("action")==name) for name in ("BUY","HOLD","SELL")}
    order=("korea","korea_index","us","global:Japan","global:HongKong","global:Germany","global:UK","futures","bonds","fx","crypto")
    return [grouped[key] for key in order if key in grouped]


class Supervisor:
    def __init__(self, runtime: Path, device: str, candidate_every: int, fee: float,
                 horizon: str = "1m", config: str = "configs/live_symbols.json",
                 initial_champion: str | None = None):
        self.runtime, self.device = runtime, device
        self.candidate_every, self.fee = candidate_every, fee
        self.config = Path(config)
        if not self.config.is_absolute():
            self.config = ROOT / self.config
        self.initial_champion = Path(initial_champion) if initial_champion else None
        if self.initial_champion is not None and not self.initial_champion.is_absolute():
            self.initial_champion = ROOT / self.initial_champion
        self.lock = threading.RLock()
        self.profile: Path | None = None
        self.mode = "live"
        self.children: dict[str, subprocess.Popen] = {}
        self.run_requested = False
        self.stopping = False
        self.restart_request: tuple[str, str | None] | None = None
        self.log_tail: list[str] = []
        self.log_handles = {}
        self._market_row_cache = {"path": None, "offset": 0, "lines": 0}
        self._latest_csv_cache = {}
        self._gpu_snapshot = {"sampled": 0.0}
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.settings_path = self.runtime / "web_settings.json"
        settings = _json(self.settings_path)
        self.mode = settings.get("mode", "live")
        self.horizon = settings.get("horizon", horizon)
        self.autonomy_enabled = bool(settings.get("paper_enabled", settings.get("autonomy_enabled", True)))
        self.observe_enabled = bool(settings.get("observe_enabled", True))
        self.worker = threading.Thread(target=self._monitor, daemon=True, name="web-supervisor")
        self.worker.start()

    def _log(self, line: str):
        self.log_tail.append(line)
        self.log_tail = self.log_tail[-30:]

    def _market_row_count(self, path: Path) -> int:
        """Count only bytes appended since the previous dashboard refresh."""
        try:
            size = path.stat().st_size
        except OSError:
            self._market_row_cache = {"path": None, "offset": 0, "lines": 0}
            return 0
        cache = self._market_row_cache
        if cache["path"] != path or size < cache["offset"]:
            cache = {"path": path, "offset": 0, "lines": 0}
        if size > cache["offset"]:
            try:
                with path.open("rb") as stream:
                    stream.seek(cache["offset"])
                    while chunk := stream.read(1024 * 1024):
                        cache["lines"] += chunk.count(b"\n")
                    cache["offset"] = stream.tell()
            except OSError:
                return max(0, cache["lines"] - 1)
        self._market_row_cache = cache
        return max(0, cache["lines"] - 1)

    def _latest_by_symbol(self, path: Path, cache_key: str) -> dict:
        """Read only appended complete CSV lines after the first scan."""
        try:
            size = path.stat().st_size
        except OSError:
            self._latest_csv_cache.pop(cache_key, None)
            return {"latest": {}, "recent": []}
        cache = self._latest_csv_cache.get(cache_key)
        if cache is None or cache["path"] != path or size < cache["offset"]:
            cache = {"path": path, "offset": 0, "fields": None,
                     "latest": {}, "recent": deque(maxlen=60)}
        if size > cache["offset"]:
            try:
                with path.open("rb") as stream:
                    stream.seek(cache["offset"])
                    chunk = stream.read()
                end = chunk.rfind(b"\n")
                if end >= 0:
                    complete = chunk[:end + 1]
                    cache["offset"] += end + 1
                    lines = complete.decode("utf-8").splitlines()
                    reader = csv.reader(lines)
                    if cache["fields"] is None:
                        cache["fields"] = [name.lstrip("\ufeff") for name in next(reader)]
                    fields = cache["fields"]
                    for values in reader:
                        if len(values) != len(fields):
                            continue
                        row = dict(zip(fields, values))
                        symbol = row.get("symbol")
                        if symbol:
                            cache["latest"][symbol] = row
                            cache["recent"].append(row)
            except (OSError, UnicodeDecodeError, csv.Error, StopIteration):
                pass
        self._latest_csv_cache[cache_key] = cache
        return cache

    def _physical_gpu(self) -> dict:
        """Sample actual GPU activity separately from PyTorch allocation."""
        now = time.monotonic()
        if now - self._gpu_snapshot["sampled"] < 10:
            return self._gpu_snapshot
        command = shutil.which("nvidia-smi")
        if command is None and os.name == "nt":
            for path in (r"C:\Windows\System32\nvidia-smi.exe",
                         r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
                if Path(path).exists():
                    command = path
                    break
        snapshot = {"sampled": now}
        if command:
            try:
                result = subprocess.run(
                    [command, "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2, check=True)
                usage, used, total = (int(part.strip()) for part in result.stdout.splitlines()[0].split(","))
                snapshot.update({"utilization_percent": usage,
                                 "memory_used_mb": used, "memory_total_mb": total})
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
        self._gpu_snapshot = snapshot
        return snapshot

    def _spawn(self, name: str, args: list[str], log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8", buffering=1)
        self.log_handles[name] = handle
        self.children[name] = subprocess.Popen(args, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._log(f"{time.strftime('%H:%M:%S')} started: {name} (PID {self.children[name].pid})")

    def _launch(self, name: str):
        assert self.profile is not None
        data = self.profile / "market.csv"
        stop = self.profile / "feed.stop"
        state = self.profile / "agent"
        if name == "feed":
            if self.mode == "mock":
                source = ROOT / "data/global_market_daily.csv"
                args = [sys.executable, "-u", "-m", "stockrl", "mock-feed", "--source", str(source),
                        "--output", str(data), "--bars", "24", "--interval-seconds", "0.25", "--stop-file", str(stop)]
            else:
                args = [sys.executable, "-u", "-m", "stockrl", "live-feed", "--config",
                        str(self.config), "--output", str(data), "--poll-seconds", "15",
                        "--stop-file", str(stop)]
            self._spawn(name, args, self.profile / "logs" / "feed.log")
            return
        args = [sys.executable, "-u", "-m", "stockrl", "global-online", "--data", str(data),
                "--state-dir", str(state), "--follow", "--poll-seconds", "1", "--window", "128",
                "--initial-lookback-bars", "8",
                "--candidate-every", str(self.candidate_every), "--fee", str(self.fee), "--horizon", self.horizon,
                "--device", self._device()]
        seed = self.initial_champion or (ROOT / "runtime-global-cuda-final/champion.pt")
        if seed.is_file():
            args.extend(["--initial-champion", str(seed)])
        self._spawn(name, args, self.profile / "logs" / "agent.log")

    def _device(self):
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def start(self, mode: str = "live", horizon: str | None = None) -> dict:
        if mode not in ("live", "mock"):
            return {"error": "mode must be live or mock"}
        from .global_online import parse_horizon
        try:
            parse_horizon(horizon or self.horizon)
        except ValueError as exc:
            return {"error": str(exc)}
        with self.lock:
            if self.stopping:
                return {"error": "System is stopping; wait for completion."}
            if self.run_requested:
                return {"ok": True, "message": "System is already running."}
            self.mode, self.profile = mode, self.runtime / mode
            # Bundled mock history is daily, so one source bar is its meaningful horizon.
            self.horizon = "1bar" if mode == "mock" else (horizon or self.horizon)
            self.profile.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(json.dumps({"mode": mode, "horizon": self.horizon,
                "autonomy_enabled":self.autonomy_enabled,"paper_enabled":self.autonomy_enabled,
                "observe_enabled":self.observe_enabled}, ensure_ascii=False, indent=2), encoding="utf-8")
            self._write_autonomy()
            for path in (self.profile / "feed.stop", self.profile / "agent" / "stop.request"):
                path.unlink(missing_ok=True)
            self.run_requested = True
            self.stopping = False
            self.children.clear()
            self._launch("feed")
            self._launch("agent")
            return {"ok": True, "message": "paper system started."}

    def _write_autonomy(self):
        profile=self.profile or (self.runtime/self.mode)
        target=profile/"agent"/"autonomy.json"
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"enabled":self.autonomy_enabled,
                                          "paper_enabled":self.autonomy_enabled,
                                          "observe_enabled":self.observe_enabled},ensure_ascii=False),encoding="utf-8")
        temporary.replace(target)
        settings=_json(self.settings_path)
        settings.update({"mode":self.mode,"horizon":self.horizon,"autonomy_enabled":self.autonomy_enabled,
                         "paper_enabled":self.autonomy_enabled,"observe_enabled":self.observe_enabled})
        self.settings_path.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding="utf-8")

    def set_autonomy(self, enabled: bool) -> dict:
        with self.lock:
            self.autonomy_enabled=bool(enabled)
            self._write_autonomy()
            return {"ok":True,"autonomy_enabled":self.autonomy_enabled}

    def set_modes(self, paper_enabled=None, observe_enabled=None) -> dict:
        with self.lock:
            if paper_enabled is not None:
                self.autonomy_enabled = bool(paper_enabled)
            if observe_enabled is not None:
                self.observe_enabled = bool(observe_enabled)
            self._write_autonomy()
            return {"ok": True, "paper_enabled": self.autonomy_enabled,
                    "observe_enabled": self.observe_enabled}

    def restart(self, mode: str, horizon: str | None = None) -> dict:
        from .global_online import parse_horizon
        if mode not in ("live", "mock"):
            return {"error": "mode must be live or mock"}
        try:
            parse_horizon(horizon or self.horizon)
        except ValueError as exc:
            return {"error": str(exc)}
        with self.lock:
            if self.stopping:
                return {"error": "System is stopping; wait for completion."}
            if not self.run_requested:
                return self.start(mode, horizon)
            self.restart_request = (mode, horizon)
            self.stop(keep_restart=True)
            return {"ok": True, "message": "Restart requested; current state is being saved."}

    def reload_feed(self) -> None:
        """Apply credential/provider changes without interrupting the model."""
        with self.lock:
            if self.run_requested and self.mode == "live" and self.profile:
                (self.profile / "feed.stop").touch()
                self._log(f"{time.strftime('%H:%M:%S')} market feed reloading after provider change")

    def stop(self, keep_restart: bool = False):
        with self.lock:
            if not keep_restart:
                self.restart_request = None
            if self.stopping:
                return {"ok": True, "message": "System is already stopping."}
            self.run_requested = False
            self.stopping = True
            if self.profile:
                (self.profile / "feed.stop").touch()
                (self.profile / "agent" / "stop.request").parent.mkdir(parents=True, exist_ok=True)
                (self.profile / "agent" / "stop.request").touch()
            threading.Thread(target=self._stop_children, daemon=True).start()
        return {"ok": True, "message": "Stopping; learning state is being saved."}

    def _stop_children(self):
        for name, proc in list(self.children.items()):
            try:
                proc.wait(timeout=30 if name == "agent" else 8)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        with self.lock:
            for handle in self.log_handles.values():
                try:
                    handle.close()
                except OSError:
                    pass
            self.log_handles.clear()
            self.children.clear()
            self.stopping = False
            request = self.restart_request
            self.restart_request = None
            self._log("Stopped; learning state saved.")
            if request is not None:
                result = self.start(*request)
                self._log("Restarted." if result.get("ok") else f"Restart failed: {result.get('error')}")

    def _monitor(self):
        while True:
            time.sleep(2)
            with self.lock:
                if not self.run_requested or not self.profile:
                    continue
                for name, proc in list(self.children.items()):
                    code = proc.poll()
                    if code is None:
                        continue
                    self._log(f"{time.strftime('%H:%M:%S')} {name} exited ({code}); restarting")
                    self.children.pop(name, None)
                    handle = self.log_handles.pop(name, None)
                    if handle:
                        handle.close()
                    if name == "feed" and self.mode == "mock" and code == 0:
                        continue
                    if name == "feed":
                        (self.profile / "feed.stop").unlink(missing_ok=True)
                    time.sleep(.1)
                    self._launch(name)

    def status(self) -> dict:
        with self.lock:
            profile = self.profile or (self.runtime / self.mode)
            state, data = profile / "agent", profile / "market.csv"
            metrics = _json(state / "metrics.json")
            feed_metrics = _json(profile / ("live_feed_metrics.json" if self.mode == "live" else "mock_feed_metrics.json"))
            instruments=_json(self.config).get("instruments",[])
            path = state / "decisions.csv"
            decision_cache = self._latest_by_symbol(path, "decisions")
            quote_cache = self._latest_by_symbol(data, "quotes")
            decisions = list(decision_cache["recent"])[::-1]
            latest_decisions = decision_cache["latest"]
            latest_quotes = quote_cache["latest"]
            fresh_symbols = set(feed_metrics.get("fresh_symbols_5m", []))
            instrument_status = []
            for item in instruments:
                symbol = item.get("symbol", "")
                quote = latest_quotes.get(symbol)
                decision = latest_decisions.get(symbol)
                instrument_status.append({
                    "symbol": symbol, "name": item.get("name"),
                    "market": item.get("market"), "asset_class": item.get("asset_class"),
                    "provider": item.get("provider"), "group": _market_group(item),
                    "fresh": symbol in fresh_symbols,
                    "quote": {key: quote.get(key) for key in ("date", "close", "volume")}
                             if quote else None,
                    "decision": {key: decision.get(key) for key in
                                 ("date", "action", "p_sell", "p_hold", "p_buy", "value")}
                                if decision else None})
            positions = _json(state / "live_positions.json")
            paper_account = _json(state / "paper_account.json")
            rows = self._market_row_count(data)
            checkpoint = state / "champion.pt"
            gpu = metrics.get("cuda_device", metrics.get("device", "CPU"))
            if metrics.get("cuda_total_memory_bytes"):
                gpu += f" VRAM {metrics.get('cuda_memory_allocated_bytes',0)/1024**3:.1f}/{metrics['cuda_total_memory_bytes']/1024**3:.1f} GB"
            return {"running": self.run_requested, "stopping": self.stopping,
                    "restarting": self.restart_request is not None,
                    "mode": self.mode, "horizon": self.horizon,
                    "feed_running": bool(self.children.get("feed") and self.children["feed"].poll() is None),
                    "agent_running": bool(self.children.get("agent") and self.children["agent"].poll() is None),
                    "feed_rows": max(0, rows),
                    "configured_instruments": len(instruments),
                    "markets": _market_overview(instruments,list(latest_decisions.values()),
                        fresh_symbols),
                    "instruments": instrument_status,
                    "provider":__import__("stockrl.provider_credentials",fromlist=["public_status"]).public_status(self.runtime),
                    "feed_metrics": feed_metrics, "metrics": metrics,
                    "autonomy_enabled":self.autonomy_enabled,
                    "paper_enabled":self.autonomy_enabled,
                    "observe_enabled":self.observe_enabled,
                    "champion_version": datetime.fromtimestamp(checkpoint.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if checkpoint.exists() else "seed pending",
                    "decisions": decisions, "positions": positions,
                    "paper_account": paper_account, "gpu": gpu,
                    "physical_gpu": self._physical_gpu(),
                    "logs": "\n".join(self.log_tail[-8:]) or "Broker API is not connected; orders remain OFF."}


def serve(host: str = "127.0.0.1", port: int = 8765, runtime: str = "runtime-global-web",
          device: str = "auto", candidate_every: int = 256, fee: float = .001,
          auto_start: bool = True, open_browser: bool = True, horizon: str = "1m",
          config: str = "configs/live_symbols.json", initial_champion: str | None = None):
    runtime_path = Path(runtime)
    if not runtime_path.is_absolute():
        runtime_path = ROOT / runtime_path
    supervisor = Supervisor(runtime_path, device, candidate_every, fee, horizon, config, initial_champion)

    class Handler(BaseHTTPRequestHandler):
        server_version = "StockRLWeb/1.0"

        def log_message(self, fmt, *args):
            # Suppress routine HTTP polling access logs; operational events are logged by Supervisor.
            return

        def _send(self, payload, code=200, content_type="application/json; charset=utf-8"):
            data = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code); self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/":
                return self._send(PAGE, content_type="text/html; charset=utf-8")
            if route == "/api/status":
                return self._send(supervisor.status())
            if route == "/api/provider":
                from .provider_credentials import public_status
                return self._send(public_status(supervisor.runtime))
            if route == "/api/provider/public-ip":
                try:
                    import ipaddress
                    import requests
                    value = requests.get("https://checkip.amazonaws.com/", timeout=8).text.strip()
                    ipaddress.ip_address(value)
                    return self._send({"ip": value})
                except Exception as exc:
                    return self._send({"error": f"Public IP lookup failed: {type(exc).__name__}"}, 502)
            self._send({"error": "not found"}, 404)

        def do_POST(self):
            try:
                n = min(int(self.headers.get("Content-Length", "0")), 4096)
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._send({"error": "invalid json"}, 400)
            route = urlparse(self.path).path
            if route == "/api/start":
                result = supervisor.start(payload.get("mode", "live"),payload.get("horizon"))
                return self._send(result, 200 if result.get("ok") else 400)
            if route == "/api/restart":
                result = supervisor.restart(payload.get("mode", "live"),payload.get("horizon"))
                return self._send(result, 200 if result.get("ok") else 400)
            if route == "/api/stop":
                return self._send(supervisor.stop())
            if route == "/api/autonomy":
                if not isinstance(payload.get("enabled"),bool):
                    return self._send({"error":"enabled must be a boolean"},400)
                return self._send(supervisor.set_autonomy(payload["enabled"]))
            if route == "/api/modes":
                if any(key in payload and not isinstance(payload[key], bool)
                       for key in ("paper_enabled", "observe_enabled")):
                    return self._send({"error":"mode flags must be boolean"},400)
                return self._send(supervisor.set_modes(payload.get("paper_enabled"),
                                                       payload.get("observe_enabled")))
            if route == "/api/feed/reconnect":
                if supervisor.mode != "live" or not supervisor.run_requested:
                    return self._send({"error": "Live market feed is not running."}, 400)
                supervisor.reload_feed()
                return self._send({"ok": True, "message": "Market feed reconnect requested."})
            if route == "/api/provider/save":
                try:
                    from .provider_credentials import save_credentials
                    result=save_credentials(supervisor.runtime,str(payload.get("provider","")),
                        str(payload.get("environment","paper")),str(payload.get("app_key","")),
                        str(payload.get("secret","")),str(payload.get("account","")))
                    supervisor.reload_feed()
                    return self._send({"ok":True,"provider":result})
                except (ValueError,RuntimeError) as exc:
                    return self._send({"error":str(exc)},400)
            if route == "/api/provider/test":
                try:
                    from .provider_credentials import test_connection
                    result=test_connection(supervisor.runtime,str(payload.get("provider", "")) or None,str(payload.get("environment", "")) or None)
                    if result.get("ok"):
                        supervisor.reload_feed()
                    return self._send(result)
                except (ValueError,RuntimeError) as exc:
                    return self._send({"error":str(exc)},400)
                except Exception as exc:
                    return self._send({"error":f"Connection check failed: {type(exc).__name__}: {exc}"},502)
            if route == "/api/provider/connect":
                try:
                    from .provider_credentials import connect_credentials
                    result = connect_credentials(supervisor.runtime,
                        str(payload.get("environment", "real")),
                        str(payload.get("app_key", "")),
                        str(payload.get("secret", "")),
                        str(payload.get("account", "")))
                    if result.get("ok"):
                        supervisor.reload_feed()
                    return self._send(result)
                except (ValueError, RuntimeError) as exc:
                    return self._send({"error": str(exc)}, 400)
                except Exception as exc:
                    return self._send({"error": f"Connection check failed: {type(exc).__name__}: {exc}"}, 502)
            if route == "/api/provider/clear":
                try:
                    from .provider_credentials import clear_credentials
                    result=clear_credentials(supervisor.runtime,str(payload.get("provider","")))
                    supervisor.reload_feed()
                    return self._send({"ok":True,"provider":result})
                except (ValueError,RuntimeError) as exc:
                    return self._send({"error":str(exc)},400)
            self._send({"error": "not found"}, 404)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    address = server.server_address
    url = f"http://127.0.0.1:{address[1]}/" if host in ("0.0.0.0", "") else f"http://{host}:{address[1]}/"
    if auto_start:
        supervisor.start(supervisor.mode)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="stockrl-web")
    thread.start()
    print(f"StockRL web dashboard: {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url, new=1, autoraise=True)
    try:
        while thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        supervisor.stop()
        server.shutdown(); server.server_close()


