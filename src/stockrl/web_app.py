"""Responsive local web dashboard and process supervisor for StockRL."""
from __future__ import annotations

import csv
from collections import deque
import json
import os
import sqlite3
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime, time as day_time, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from .paths import default_runtime_dir, ensure_project_path, validate_model_dir
from .paper_account import KR_SELL_TAX_ASSUMPTION, SEED_CASH

ROOT = Path(__file__).resolve().parents[2]

PAGE = Path(__file__).with_name("web_dashboard.html").read_text(encoding="utf-8")


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _utc_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)


def _agent_progress_health(data_path: Path, state_dir: Path,
                           process_running: bool) -> dict:
    """Compare the persisted agent cursor with the newest indexed feed bar."""
    cursor = _utc_datetime(_json(state_dir / "live_cursor.json").get("last_timestamp"))
    latest = None
    lag_bars = None
    lookup_error = None
    index_path = data_path.with_suffix(data_path.suffix + ".sqlite3")
    if index_path.exists():
        connection = None
        try:
            connection = sqlite3.connect(
                f"{index_path.resolve().as_uri()}?mode=ro", uri=True, timeout=0.5
            )
            connection.execute("PRAGMA query_only=ON")
            cursor_ns = int(cursor.timestamp() * 1_000_000_000) if cursor else -1
            latest_ns, lag_bars = connection.execute(
                "SELECT MAX(stamp_ns), COUNT(DISTINCT CASE WHEN stamp_ns > ? "
                "THEN stamp_ns END) FROM seen",
                (cursor_ns,),
            ).fetchone()
            if latest_ns is not None:
                latest = datetime.fromtimestamp(latest_ns / 1_000_000_000, timezone.utc)
        except (OSError, sqlite3.Error) as exc:
            lookup_error = f"market index unavailable: {type(exc).__name__}"
        finally:
            if connection is not None:
                connection.close()

    lag_seconds = max(0, int((latest - cursor).total_seconds())) if latest and cursor else None
    if not process_running:
        health = "stopped"
        reason = "agent process is stopped"
    elif lookup_error:
        health = "unknown"
        reason = lookup_error
    elif latest is None or cursor is None:
        health = "unknown"
        reason = "feed timestamp or agent cursor is unavailable"
    elif lag_seconds is not None and lag_seconds > 300:
        health = "stale"
        reason = f"agent cursor trails feed by {lag_seconds}s across {lag_bars} bars"
    else:
        health = "healthy"
        reason = "agent cursor is within five minutes of the newest feed bar"
    return {
        "status": health,
        "reason": reason,
        "latest_feed_timestamp_utc": latest.isoformat() if latest else None,
        "agent_cursor_timestamp_utc": cursor.isoformat() if cursor else None,
        "lag_seconds": lag_seconds,
        "lag_bars": lag_bars,
        "threshold_seconds": 300,
    }


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
                 initial_champion: str | None = None, model_dir: str | Path | None = None,
                 settings_dir: str | Path | None = None):
        self.runtime = ensure_project_path(runtime, "runtime")
        self.device = device
        self.candidate_every, self.fee = candidate_every, fee
        self.config = Path(config)
        if not self.config.is_absolute():
            self.config = ROOT / self.config
        self.initial_champion = Path(initial_champion) if initial_champion else None
        if self.initial_champion is not None and not self.initial_champion.is_absolute():
            self.initial_champion = ROOT / self.initial_champion
        requested_model_dir = model_dir or os.environ.get("STOCKRL_MODEL_DIR")
        self.model_dir = validate_model_dir(requested_model_dir)
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
        self.settings_path = (ensure_project_path(settings_dir, "settings") / "web_settings.json"
                              if settings_dir else ROOT / "configs" / "local" / "web_settings.json")
        ensure_project_path(self.settings_path, "settings")
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings = _json(self.settings_path)
        self.mode = settings.get("mode", "live")
        self.horizon = settings.get("horizon", horizon)
        self.autonomy_enabled = bool(settings.get("paper_enabled", settings.get("autonomy_enabled", True)))
        self.observe_enabled = bool(settings.get("observe_enabled", True))
        if self.autonomy_enabled:
            self.observe_enabled = True
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
        if log_path.exists() and log_path.stat().st_size >= 8*1024*1024:
            log_path.write_text("",encoding="utf-8")
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
                "--state-dir", str(state), "--model-dir", str(self.model_dir), "--follow", "--poll-seconds", "1", "--window", "128",
                "--initial-lookback-bars", "8",
                "--candidate-every", str(self.candidate_every), "--batch-size", "8", "--updates", "8",
                "--fee", str(self.fee), "--horizon", self.horizon,
                "--device", self._device()]
        seed = self.initial_champion or (self.model_dir / "champion.pt")
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
            if self.autonomy_enabled:
                self.observe_enabled=True
            self._write_autonomy()
            return {"ok":True,"autonomy_enabled":self.autonomy_enabled,
                    "paper_enabled":self.autonomy_enabled,"observe_enabled":self.observe_enabled}

    def set_modes(self, paper_enabled=None, observe_enabled=None) -> dict:
        with self.lock:
            if paper_enabled is not None:
                self.autonomy_enabled = bool(paper_enabled)
            if observe_enabled is not None:
                self.observe_enabled = bool(observe_enabled)
            if self.autonomy_enabled:
                self.observe_enabled = True
            elif not self.observe_enabled:
                # Pending paper orders are cleared on the next bar; existing
                # holdings remain in the paper account.
                self.autonomy_enabled = False
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

    def reset_paper_accounts(self) -> dict:
        """Reset the live Champion and Candidate observer paper ledgers safely."""
        with self.lock:
            if self.stopping:
                return {"error": "System is stopping; wait before resetting accounts."}
            was_running = self.run_requested
            mode, horizon = self.mode, self.horizon
            profile = self.profile or (self.runtime / self.mode)

        if was_running:
            self.stop()
            deadline = time.monotonic() + 45.0
            while time.monotonic() < deadline:
                with self.lock:
                    stopped = not self.stopping and not self.run_requested
                if stopped:
                    break
                time.sleep(0.1)
            else:
                return {"error": "Live workers did not stop; paper accounts were not reset."}

        state = profile / "agent"
        state.mkdir(parents=True, exist_ok=True)
        from .paper_account import PaperAccount

        for filename in ("paper_account.json", "candidate_observer_account.json"):
            PaperAccount(state / filename, self.fee, 0.0).reset()

        decisions_path = state / "decisions.csv"
        decisions_path.write_text("date,symbol,action,value,p_sell,p_hold,p_buy\n",
                                  encoding="utf-8")
        positions_path = state / "live_positions.json"
        temporary = positions_path.with_suffix(".json.reset.tmp")
        temporary.write_text("{}", encoding="utf-8")
        os.replace(temporary, positions_path)
        with self.lock:
            self._latest_csv_cache.pop("decisions", None)

        observer_path = state / "candidate_observer_state.json"
        observer = _json(observer_path)
        observer.update({"status": "waiting_for_candidate_update",
                         "last_timestamp": None, "last_decisions": [],
                         "last_inference_seconds": None, "error": None})
        temporary = observer_path.with_suffix(".json.reset.tmp")
        temporary.write_text(json.dumps(observer, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, observer_path)

        metrics_path = state / "metrics.json"
        metrics = _json(metrics_path)
        metrics.update({"paper_net_reward": 0.0, "paper_account_reward": 0.0,
                        "fee_total": 0.0, "slippage_total": 0.0,
                        "paper_account_reset_utc": datetime.now(timezone.utc).isoformat(),
                        "candidate_live_status": "waiting_for_candidate",
                        "candidate_live_last_timestamp": None})
        temporary = metrics_path.with_suffix(".json.reset.tmp")
        temporary.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, metrics_path)

        if was_running:
            result = self.start(mode, horizon)
            if not result.get("ok"):
                return {"error": "Accounts reset, but system restart failed: " +
                        str(result.get("error", "unknown error")), "reset": True}
        return {"ok": True, "reset": ["champion", "candidate_observer"],
                "system_restarted": was_running,
                "message": "Champion and Candidate observer paper accounts reset to seed cash."}

    def _stop_children(self):
        for name, proc in list(self.children.items()):
            try:
                proc.wait(timeout=90 if name == "agent" else 8)
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
            action_direction = {"BUY": 1, "SELL": -1, "HOLD": 0}
            model_directions = {
                symbol: action_direction.get(row.get("action"), 0)
                for symbol, row in latest_decisions.items()
            }
            paper_account = _json(state / "paper_account.json")
            candidate_observer_account = _json(state / "candidate_observer_account.json")
            candidate_observer_state = _json(state / "candidate_observer_state.json")
            candidate_observer_books = {}
            for currency, book in candidate_observer_account.get("books", {}).items():
                positions = book.get("positions", {})
                marks = book.get("marks", {})
                holdings_value = sum(float(position.get("quantity", 0.0)) * float(
                    marks.get(symbol, position.get("average_cost", 0.0)))
                    for symbol, position in positions.items())
                initial_cash = float(book.get("initial_cash", 0.0))
                equity = float(book.get("cash", 0.0)) + holdings_value
                unrealized = sum(float(position.get("quantity", 0.0)) * (
                    float(marks.get(symbol, position.get("average_cost", 0.0)))
                    - float(position.get("average_cost", 0.0)))
                    for symbol, position in positions.items())
                candidate_observer_books[currency] = {
                    "initial_cash": initial_cash,
                    "cash": float(book.get("cash", 0.0)),
                    "net_pnl": equity - initial_cash,
                    "net_return_rate": ((equity - initial_cash) / initial_cash
                                        if initial_cash else 0.0),
                    "realized_pnl": float(book.get("realized_pnl", 0.0)),
                    "unrealized_pnl": unrealized,
                    "costs": sum(float(book.get(key, 0.0)) for key in
                                 ("fees", "sell_tax", "spread", "slippage")),
                    "trade_count": int(book.get("trade_count", 0)),
                    "position_count": sum(1 for position in positions.values()
                                          if float(position.get("quantity", 0.0)) != 0.0),
                    "positions": [{"symbol": symbol,
                                   "quantity": float(position.get("quantity", 0.0)),
                                   "average_cost": float(position.get("average_cost", 0.0)),
                                   "mark": float(marks.get(symbol, position.get("average_cost", 0.0)))}
                                  for symbol, position in sorted(positions.items())
                                  if float(position.get("quantity", 0.0)) != 0.0],
                }
            candidate_live_account = {
                "available": (bool(candidate_observer_books) and
                    candidate_observer_state.get("status") in
                    ("observing", "training_and_observing")),
                "status": candidate_observer_state.get("status", "waiting_for_candidate_update"),
                "last_timestamp": candidate_observer_state.get("last_timestamp",
                    candidate_observer_account.get("last_timestamp")),
                "candidate_version": candidate_observer_state.get("candidate_version"),
                "candidate_training": bool(candidate_observer_state.get("candidate_training",
                    metrics.get("candidate_training"))),
                "last_inference_seconds": candidate_observer_state.get("last_inference_seconds"),
                "inference_count": int(metrics.get("candidate_live_inference_count", 0)),
                "inference_seconds_total": float(metrics.get(
                    "candidate_live_inference_seconds_total", 0.0)),
                "queue_drops": int(metrics.get("candidate_live_queue_drops", 0)),
                "policy_mode": candidate_observer_state.get(
                    "policy_mode", "same_epsilon_sampling_and_random_draws_as_champion"),
                "error": candidate_observer_state.get("error",
                    metrics.get("candidate_live_error")),
                "decisions": candidate_observer_state.get("last_decisions", []),
                "recent_fills": list(candidate_observer_account.get("fills", []))[-12:],
                "books": candidate_observer_books,
            }
            validation_state = _json(state / "candidate_validation.json")
            validation_books = {}
            validation_accounts_available = True
            validation_initial_cash = {}
            validation_last_timestamps = {}
            for model_name, account_file in (
                    ("champion", "candidate_validation_champion.json"),
                    ("candidate", "candidate_validation_candidate.json")):
                account_path = state / account_file
                account = _json(account_path)
                validation_last_timestamps[model_name] = account.get("last_timestamp")
                validation_initial_cash[model_name] = {
                    currency: float(book.get("initial_cash", 0.0))
                    for currency, book in account.get("books", {}).items()
                }
                validation_accounts_available = (
                    validation_accounts_available and account_path.is_file()
                    and bool(account.get("books")))
                model_books = {}
                for currency, book in account.get("books", {}).items():
                    positions = book.get("positions", {})
                    marks = book.get("marks", {})
                    holdings_value = sum(
                        float(position.get("quantity", 0.0)) * float(
                            marks.get(symbol, position.get("average_cost", 0.0)))
                        for symbol, position in positions.items())
                    initial_cash = float(book.get("initial_cash", 0.0))
                    cash = float(book.get("cash", 0.0))
                    equity = cash + holdings_value
                    unrealized = sum(
                        float(position.get("quantity", 0.0)) * (
                            float(marks.get(symbol, position.get("average_cost", 0.0)))
                            - float(position.get("average_cost", 0.0)))
                        for symbol, position in positions.items())
                    model_books[currency] = {
                        "initial_cash": initial_cash,
                        "cash": cash,
                        "equity": equity,
                        "holdings_value": holdings_value,
                        "net_pnl": equity - initial_cash,
                        "net_return_rate": ((equity - initial_cash) / initial_cash
                                            if initial_cash else 0.0),
                        "realized_pnl": float(book.get("realized_pnl", 0.0)),
                        "unrealized_pnl": unrealized,
                        "costs": sum(float(book.get(key, 0.0)) for key in
                                     ("fees", "sell_tax", "spread", "slippage")),
                        "trade_count": int(book.get("trade_count", 0)),
                        "position_count": sum(
                            1 for position in positions.values()
                            if float(position.get("quantity", 0.0)) != 0.0),
                        "positions": [
                            {"symbol": symbol,
                             "quantity": float(position.get("quantity", 0.0))}
                            for symbol, position in sorted(positions.items())
                            if float(position.get("quantity", 0.0)) != 0.0
                        ],
                        "recent_fills": list(account.get("fills", []))[-12:],
                    }
                validation_books[model_name] = model_books
            validation_comparison = {
                "status": validation_state.get("status", "not_started"),
                "active": bool(metrics.get("candidate_validation_active")) and
                          validation_state.get("status") == "collecting",
                "bars_current": int(metrics.get("candidate_validation_bars",
                                                 validation_state.get("bars", 0)) or 0),
                "bars_required": int(metrics.get("candidate_min_validation_dates", 128) or 128),
                "snapshot_version": metrics.get("candidate_validation_snapshot_version",
                                                 validation_state.get("source_candidate_version")),
                "accounts_available": validation_accounts_available,
                "last_timestamp": (validation_state.get("last_timestamp") or
                                   validation_last_timestamps.get("champion")),
                "start_after": validation_state.get("start_after"),
                "same_market_timeline": bool(
                    validation_state.get("same_market_timeline", False) and
                    validation_last_timestamps.get("champion") is not None and
                    validation_last_timestamps.get("champion") ==
                    validation_last_timestamps.get("candidate")),
                "same_market_input": bool(
                    validation_state.get("same_market_input", False)),
                "same_last_bar": bool(
                    validation_last_timestamps.get("champion") is not None and
                    validation_last_timestamps.get("champion") ==
                    validation_last_timestamps.get("candidate")),
                "same_starting_cash": bool(validation_accounts_available and
                    validation_initial_cash.get("champion") ==
                    validation_initial_cash.get("candidate") == SEED_CASH),
                "fee_rate": float(metrics.get("fee_rate", self.fee)),
                "slippage_bps": float(metrics.get("slippage_bps", 1.0)),
                "krw_sell_tax_rate": KR_SELL_TAX_ASSUMPTION,
                "same_cost_rules": bool(validation_accounts_available),
                "action_rule": "same highest-probability action; no exploration draw",
                "same_action_rule": bool(validation_state.get("same_action_rule", False)),
                "comparison_valid": bool(validation_state.get("comparison_valid", False)),
                "reason": validation_state.get("reason"),
                "last_decisions": validation_state.get("last_decisions", {}),
                "champion": validation_books["champion"],
                "candidate": validation_books["candidate"],
            }
            live_seed = {name: {
                currency: float(book.get("initial_cash", 0.0))
                for currency, book in account.get("books", {}).items()
            } for name, account in (("champion", paper_account),
                                    ("candidate", candidate_observer_account))}
            live_account_comparison = {
                "score_is_promotion_gate": False,
                "same_seed_cash": live_seed["champion"] == live_seed["candidate"] == SEED_CASH,
                "same_fee_rate": float(metrics.get("fee_rate", self.fee)),
                "same_slippage_bps": float(metrics.get("slippage_bps", 1.0)),
                "same_krw_sell_tax_rate": KR_SELL_TAX_ASSUMPTION,
                "same_action_sampling": "same exploration probability and random draws for each queued market observation",
                "candidate_snapshot_version": candidate_observer_state.get("candidate_version"),
                "candidate_skipped_observations": int(metrics.get("candidate_live_queue_drops", 0)),
                "last_bar_timestamps_equal": (
                    paper_account.get("last_timestamp") ==
                    candidate_observer_account.get("last_timestamp")),
                "reason_not_a_fair_score": (
                    "운영 관찰 계좌는 Candidate 가중치가 바뀐 여러 시점과 일부 건너뛴 관측을 누적합니다. "
                    "승급 점수는 별도 128분 동일 구간 시험만 사용합니다."),
            }
            paper_positions = {}
            paper_financials = {}
            for currency, book in paper_account.get("books", {}).items():
                held = book.get("positions", {})
                for symbol, position in held.items():
                    quantity = float(position.get("quantity", 0.0))
                    if quantity:
                        paper_positions[f"{currency}:{symbol}"] = {
                            "symbol": symbol, "currency": currency, "quantity": quantity,
                            "average_cost": float(position.get("average_cost", 0.0)),
                            "mark": float(book.get("marks", {}).get(symbol, position.get("average_cost", 0.0)))
                        }
                unrealized = sum(
                    float(position.get("quantity", 0.0)) * (
                        float(book.get("marks", {}).get(symbol, position.get("average_cost", 0.0)))
                        - float(position.get("average_cost", 0.0)))
                    for symbol, position in held.items())
                paper_financials[currency] = {
                    "initial_cash": float(book.get("initial_cash", 0.0)),
                    "cash": float(book.get("cash", 0.0)),
                    "net_pnl": float(book.get("net_pnl", 0.0)),
                    "realized_pnl": float(book.get("realized_pnl", 0.0)),
                    "unrealized_pnl": unrealized,
                    "trade_count": int(book.get("trade_count", 0)),
                    "position_count": sum(1 for position in held.values()
                                           if float(position.get("quantity", 0.0)) != 0.0),
                    "fees": float(book.get("fees", 0.0)),
                    "sell_tax": float(book.get("sell_tax", 0.0)),
                    "spread": float(book.get("spread", 0.0)),
                    "slippage": float(book.get("slippage", 0.0)),
                }
            probability_counts = {}
            for row in latest_decisions.values():
                try:
                    signature = tuple(round(float(row[key]), 6) for key in ("p_sell", "p_hold", "p_buy"))
                except (KeyError, TypeError, ValueError):
                    continue
                probability_counts[signature] = probability_counts.get(signature, 0) + 1
            repeated = max(probability_counts.values(), default=0)
            output_diagnostics = {
                "symbols_with_probabilities": sum(probability_counts.values()),
                "unique_probability_vectors": len(probability_counts),
                "largest_identical_group": repeated,
                "warning": repeated >= 3,
                "note": ("확률이 같은 종목이 반복됩니다. 입력·중간 출력 원인은 별도 진단이 필요하며 행동은 자동 차단하지 않습니다."
                         if repeated >= 3 else None),
            }
            learning_candidate_every = int(metrics.get("candidate_every", self.candidate_every))
            learning_min_replay = int(metrics.get("candidate_min_replay", 8))
            learning_min_holdout = int(metrics.get("candidate_min_validation_dates", 128))
            learning_replay = int(metrics.get("candidate_eligible_replay_count",
                metrics.get("trainable_replay_count", metrics.get("replay_count", 0))))
            learning_holdout = int(metrics.get("candidate_validation_bars", metrics.get("validation_window_dates", 0)))
            learning_skip = metrics.get("candidate_skip_reason")
            if metrics.get("candidate_training"):
                learning_skip = (f"Candidate 학습 중 · optimizer "
                    f"{metrics.get('candidate_optimizer_steps_current',0)}/"
                    f"{metrics.get('candidate_optimizer_steps_target',0)}회")
            elif not learning_skip:
                learning_skip = (f"학습 가능한 replay {learning_replay}건 · 다음 batch 조건 확인 중")
            learning_blocker = metrics.get("promotion_blocked_reason")
            rows = self._market_row_count(data)
            checkpoint = self.model_dir / "champion.pt"
            gpu = metrics.get("cuda_device", metrics.get("device", "CPU"))
            if metrics.get("cuda_total_memory_bytes"):
                gpu += f" VRAM {metrics.get('cuda_memory_allocated_bytes',0)/1024**3:.1f}/{metrics['cuda_total_memory_bytes']/1024**3:.1f} GB"
            provider_status=__import__("stockrl.provider_credentials",fromlist=["public_status"]).public_status(self.runtime)
            feed_running=bool(self.children.get("feed") and self.children["feed"].poll() is None)
            agent_process_running=bool(self.children.get("agent") and self.children["agent"].poll() is None)
            if self.mode == "live":
                agent_health=_agent_progress_health(data,state,agent_process_running)
            else:
                agent_health={"status":"healthy" if agent_process_running else "stopped",
                              "reason":"mock agent process status" if agent_process_running else "agent process is stopped",
                              "latest_feed_timestamp_utc":None,"agent_cursor_timestamp_utc":None,
                              "lag_seconds":None,"lag_bars":None,"threshold_seconds":300}
            agent_running=(agent_process_running and agent_health["status"]=="healthy")
            feed_metrics["broker_provider"]=provider_status["provider"]
            feed_metrics["provider_environment"]=provider_status["environment"]
            if not feed_running:
                feed_metrics["broker_connected"]=False
            status_logs=self.log_tail[-8:]
            if agent_health["status"] in ("stale","unknown"):
                alert=f"AGENT ALERT: process {'alive' if agent_process_running else 'stopped'}, progress {agent_health['status']}: {agent_health['reason']}"
                status_logs=[alert]+status_logs[-7:]
            return {"running": self.run_requested, "stopping": self.stopping,
                    "restarting": self.restart_request is not None,
                    "mode": self.mode, "horizon": self.horizon,
                    "feed_running": feed_running,
                    "agent_running": agent_running,
                    "agent_process_running": agent_process_running,
                    "agent_health": agent_health,
                    "feed_rows": max(0, rows),
                    "configured_instruments": len(instruments),
                    "markets": _market_overview(instruments,list(latest_decisions.values()),
                        fresh_symbols),
                    "instruments": instrument_status,
                    "provider":provider_status,
                    "feed_metrics": feed_metrics, "metrics": metrics,
                    "backtest": _json(state / "backtest.json"),
                    "autonomy_enabled":self.autonomy_enabled,
                    "paper_enabled":self.autonomy_enabled,
                    "observe_enabled":self.observe_enabled,
                    "learning": {
                        "candidate_learning_enabled": metrics.get("candidate_learning_enabled", True),
                        "champion_learning_enabled":metrics.get("champion_learning_enabled",False),
                        "dual_learning_enabled":metrics.get("dual_learning_enabled",False),
                        "champion_training":metrics.get("champion_training",False),
                        "champion_model_version":metrics.get("champion_training_version",0),
                        "champion_remaining":metrics.get("champion_eligible_replay_count",0),
                        "eligible_backlog":metrics.get("replay_eligible_backlog",learning_replay),
                        "blocked_replay":int(metrics.get("replay_quarantined_count",0))+int(metrics.get("replay_unsupported_count",0)),
                        "candidate_stage": metrics.get("candidate_stage",
                            "training" if metrics.get("candidate_training") else "waiting"),
                        "candidate_every": learning_candidate_every,
                        "replay_current": learning_replay,
                        "daily_learning": metrics.get("daily_learning", []),
                        "daily_learning_timezone": "Asia/Seoul",
                        "replay_untrained_count": metrics.get("replay_untrained_count"),
                        "replay_quarantined_count": metrics.get("replay_quarantined_count", 0),
                        "replay_unsupported_count": metrics.get("replay_unsupported_count", 0),
                        "replay_oldest_unfinished_timestamp": metrics.get("replay_oldest_unfinished_timestamp"),
                        "replay_size_limit_enabled": metrics.get("replay_size_limit_enabled", False),
                        "candidate_window_forwards": metrics.get("last_candidate_window_forwards"),
                        "candidate_min_replay": learning_min_replay,
                        "holdout_timestamps_current": learning_holdout,
                        "holdout_timestamps_required": learning_min_holdout,
                        "candidate_skip_reason": learning_skip,
                        "promotion_gate_ready": metrics.get("promotion_gate_ready", False),
                        "promotion_blocked_reason": learning_blocker,
                        "validation_bars_current": learning_holdout,
                        "validation_bars_required": learning_min_holdout,
                        "candidate_training_samples": metrics.get("last_candidate_samples_trained"),
                        "candidate_training_unique_samples": metrics.get("last_candidate_unique_samples_trained"),
                        "candidate_training_samples_target": metrics.get("candidate_samples_target"),
                        "candidate_optimizer_steps": metrics.get("last_candidate_optimizer_steps"),
                        "candidate_optimizer_steps_target": metrics.get("candidate_optimizer_steps_target"),
                        "candidate_update_seconds": metrics.get("last_candidate_update_seconds"),
                        "candidate_peak_allocated_bytes": metrics.get("last_candidate_peak_allocated_bytes"),
                        "candidate_baseline_allocated_bytes": metrics.get("last_candidate_baseline_allocated_bytes"),
                        "candidate_peak_reserved_bytes": metrics.get("last_candidate_peak_reserved_bytes"),
                        "candidate_validation_score": metrics.get("last_candidate_validation_score"),
                        "champion_validation_score": metrics.get("last_champion_validation_score"),
                        "candidate_score_unit": "net_return_rate",
                        "paper_trade_count": sum(int(book.get("trade_count", 0))
                                                  for book in paper_account.get("books", {}).values()),
                        "paper_trade_counts_by_currency": {
                            currency: int(book.get("trade_count", 0))
                            for currency, book in paper_account.get("books", {}).items()},
                        "gpu_device": metrics.get("cuda_device", metrics.get("device", "CPU")),
                        "gpu_allocated_bytes": metrics.get("cuda_memory_allocated_bytes"),
                        "gpu_reserved_bytes": metrics.get("cuda_memory_reserved_bytes"),
                        "gpu_total_bytes": metrics.get("cuda_total_memory_bytes"),
                    },
                    "output_diagnostics": output_diagnostics,
                    "champion_version": datetime.fromtimestamp(checkpoint.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if checkpoint.exists() else "seed pending",
                    "decisions": decisions, "model_directions": model_directions,
                    "positions": paper_positions, "paper_positions": paper_positions,
                    "paper_financials": paper_financials, "paper_account": paper_account, "gpu": gpu,
                    "validation_comparison": validation_comparison,
                    "live_account_comparison": live_account_comparison,
                    "candidate_live_account": candidate_live_account,
                    "real_orders_enabled": False,
                    "physical_gpu": self._physical_gpu(),
                    "logs": "\n".join(status_logs) or "Broker API is not connected; orders remain OFF."}


def serve(host: str = "127.0.0.1", port: int = 8766, runtime: str | None = None,
          device: str = "auto", candidate_every: int = 16, fee: float = .001,
          auto_start: bool = True, open_browser: bool = True, horizon: str = "1m",
          config: str = "configs/live_symbols.json", initial_champion: str | None = None,
          model_dir: str | None = None, settings_dir: str | None = None):
    runtime_path = Path(runtime) if runtime is not None else default_runtime_dir()
    if not runtime_path.is_absolute():
        runtime_path = ROOT / runtime_path
    supervisor = Supervisor(runtime_path, device, candidate_every, fee, horizon, config, initial_champion,
                            model_dir, settings_dir)
    restart_server_requested = threading.Event()

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
            if route == "/api/health":
                return self._send({"ok": True, "service": "stockrl", "port": port})
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
            if route == "/api/server/restart":
                if restart_server_requested.is_set():
                    return self._send({"error": "Server restart is already in progress."}, 409)
                self._send({"ok": True, "message": "Server restart accepted; saving and stopping live workers."})
                threading.Timer(0.5, restart_server_requested.set).start()
                return
            if route == "/api/start":
                result = supervisor.start(payload.get("mode", "live"),payload.get("horizon"))
                return self._send(result, 200 if result.get("ok") else 400)
            if route == "/api/restart":
                result = supervisor.restart(payload.get("mode", "live"),payload.get("horizon"))
                return self._send(result, 200 if result.get("ok") else 400)
            if route == "/api/stop":
                return self._send(supervisor.stop())
            if route == "/api/paper-accounts/reset":
                result = supervisor.reset_paper_accounts()
                return self._send(result, 200 if result.get("ok") else 409)
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
        while thread.is_alive() and not restart_server_requested.is_set():
            restart_server_requested.wait(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        supervisor.stop()
        server.shutdown(); server.server_close()
        if restart_server_requested.is_set():
            # Wait for the existing worker processes to save and exit before
            # relaunching the server, avoiding duplicate feed/agent processes.
            while supervisor.stopping:
                time.sleep(0.25)
            original_argv = getattr(sys, "orig_argv", None)
            if original_argv and len(original_argv) > 1:
                os.execv(sys.executable, [sys.executable, *original_argv[1:]])
            os.execv(sys.executable, [sys.executable, "-m", "stockrl.launch_web", "--server-only"])


