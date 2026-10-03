"""Dashboard status and cached measurements."""
from __future__ import annotations
import csv
from collections import deque
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from ..account_diagnostics import summarize_account, input_availability
from .health import _json, _market_group, _market_overview

class _StatusMixin:
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

    def status(self) -> dict:
        with self.lock:
            profile=self.profile or self.runtime/self.mode
            data=profile/"market.csv"
            quotes=self._latest_by_symbol(data,"quotes")["latest"]
            settings=_json(self.config)
            instruments=settings.get("instruments",[])
            feed=_json(profile/("live_feed_metrics.json" if self.mode=="live" else "mock_feed_metrics.json"))
            fresh=set(feed.get("fresh_symbols_5m",[]))
            provider=__import__("stockrl.provider_credentials",fromlist=["public_status"]).public_status(self.runtime)
            feed_running=bool(self.children.get("feed") and self.children["feed"].poll() is None)
            runtime={}; accounts={}; ledgers={}; health={}; metrics={}; decisions=[]
            for role,requested in self.model_enabled.items():
                worker=self._moe_model_worker(role)
                native=worker.status()
                live=bool(native.get("alive")); compute=native.get("compute",{})
                decision=native.get("decision",{}); learning=native.get("learning",{})
                loaded=live and native.get("load_count",0)>0
                status="saving" if native.get("stop_requested") and live else native["status"]
                ledger=_json(worker.state/"paper_account.json")
                # Saved operating accounts survive model replacement and trial runs.
                account_path=worker.state/"paper_account.json"
                saved_path=profile/"agent"/("paper_account.json" if role=="champion" else "candidate_observer_account.json")
                saved_ledger=_json(saved_path)
                def has_trades(account):
                    return bool(account.get("fills") or account.get("pending") or any(
                        book.get("positions") or book.get("trade_count")
                        for book in account.get("books",{}).values()))
                saved_account=bool(saved_ledger and (not ledger or not has_trades(ledger)))
                if saved_account:
                    ledger=saved_ledger
                    account_path=saved_path
                trial=worker.runner_script=="run_assembly_trial.py"
                summary=summarize_account(ledger)
                books=summary["books"]
                runtime[role]={"status":status,"requested":requested,"loaded":loaded,"pid":native.get("pid"),
                    "family":"trading_moe","checkpoint":str(worker.checkpoint),"error":native.get("error"),
                    "device":compute.get("learning_device") if loaded else None,
                    "compute_device":compute.get("inference_device") if live else None,
                    "ram_weight_bytes":native.get("worker_ram_bytes",0),
                    "gpu_weight_bytes":compute.get("allocated_bytes",0) if live else 0,
                    "last_decision":decision.get("as_of"),"decision_seconds":decision.get("seconds"),
                    "updated_at":native.get("updated_at"),"source_kind":"saved_operating_account" if saved_account else "historical_paper",
                    "source":"TradingMoE · 공식 ETHUSDT 과거 가상매매" if not trial else "TradingMoE · 조립 Candidate 시험",
                    "memory_scope":"worker","account_scope":"long_term","account_path":str(account_path),
                    "learning_active":live and bool(native.get("learning_active")),"learning":learning,
                    "optimizer_updates":native.get("optimizer_updates",0),"replay":native.get("replay",{}),"books":books}
                accounts[role]={**summary,"books":books,"training":runtime[role]["learning_active"],
                    "version":native.get("optimizer_updates",0),"last_inference_seconds":decision.get("seconds"),
                    "last_full_decision_timestamp":decision.get("as_of"),"account_scope":runtime[role]["account_scope"]}
                ledgers[role]=ledger
                health[role]={"status":"healthy" if status=="running" else status,
                    "reason":runtime[role]["source"],"lag_seconds":None,"lag_bars":None,
                    "agent_cursor_timestamp_utc":decision.get("as_of"),"latest_feed_timestamp_utc":None}
                metrics[role+"_training"]=runtime[role]["learning_active"]
                metrics[role+"_training_version"]=native.get("optimizer_updates",0)
                metrics[role+"_update_error"]=native.get("error")
                metrics[role+"_last_completed_round"]={"samples":learning.get("samples"),"loss":learning.get("loss"),
                    "optimizer_updates":native.get("optimizer_updates",0)}
                if decision:
                    decisions.append({"symbol":"ETHUSDT","date":decision.get("as_of"),"action":decision.get("action"),
                        "target_weight":decision.get("target_weight"),"value":decision.get("value"),"role":role})
            assembly=getattr(self,"assembly_orchestrator",None)
            experiment=assembly.status() if assembly else {}
            candidate=experiment.get("candidate",{}); scores=candidate.get("scores",{})
            paper=scores.get("paper",{}); replay_score=scores.get("replay",{})
            evaluation=paper or replay_score
            stage=candidate.get("evaluation_state","not_started")
            trial={"status":stage,"active":bool(experiment.get("worker",{}).get("alive")),
                "bars_current":evaluation.get("candidate",{}).get("decisions",0),
                "bars_required":evaluation.get("champion",{}).get("decisions",0),
                "snapshot_version":candidate.get("candidate_id"),
                "champion_snapshot_version":experiment.get("champion",{}).get("candidate_id"),
                "comparison_valid":bool(evaluation),"reason":candidate.get("reason"),
                "champion":{},"candidate":{},"scores":scores,"source":"assembly",
                "long_term_accounts_preserved":True}
            latest={row["symbol"]:row for row in decisions if row["role"]=="champion"}
            instrument_status=[]
            for instrument in instruments:
                symbol=instrument.get("symbol","")
                instrument_status.append({**instrument,"group":_market_group(instrument),"fresh":symbol in fresh,
                    "quote":quotes.get(symbol),"decision":latest.get(symbol)})
            cr=runtime["candidate"]; ch=runtime["champion"]
            remaining=sum(item["replay"].get("remaining_for_update",item["replay"].get("untrained",0)) for item in runtime.values())
            learning={"candidate_learning_enabled":self.learning_enabled,"champion_learning_enabled":self.learning_enabled,
                "dual_learning_enabled":True,"champion_training":ch["learning_active"],"candidate_training":cr["learning_active"],
                "candidate_stage":"training" if cr["learning_active"] else cr["status"],
                "champion_model_version":ch["optimizer_updates"],"candidate_optimizer_steps":cr["optimizer_updates"],
                "candidate_training_samples":cr["learning"].get("samples"),"candidate_update_seconds":cr["learning"].get("seconds"),
                "candidate_skip_reason":cr["learning"].get("reason"),"replay_current":remaining,"eligible_backlog":remaining,
                "replay_pending_count":sum(item["replay"].get("pending",0) for item in runtime.values()),
                "replay_quarantined_count":0,"replay_unsupported_count":0,
                "daily_learning":ch["replay"].get("daily",[]),"multiscale_input_status":{},
                "reward_credit":{"source":"TradingMoE paper account","definition":"cost-adjusted NAV change"},
                "gpu_device":ch.get("compute_device") or cr.get("compute_device"),
                "gpu_allocated_bytes":sum(item["gpu_weight_bytes"] for item in runtime.values())}
            books=accounts["champion"]["books"]
            positions={f"{currency}:{position['symbol']}":{**position,"currency":currency}
                for currency,book in books.items() for position in book.get("positions",[])}
            feed.update(broker_provider=provider["provider"],provider_environment=provider["environment"])
            if not feed_running:feed["broker_connected"]=False
            live=any(item.get("pid") for item in runtime.values())
            return {"status_updated_at":datetime.now().astimezone().isoformat(),"running":self.run_requested,
                "stopping":self.stopping,"restarting":self.restart_request is not None,"agent_reload_pending":False,
                "mode":self.mode,"horizon":self.horizon,"feed_running":feed_running,
                "agent_process_running":live,"agent_running":any(item["status"]=="running" for item in runtime.values()),
                "model_runtime":runtime,"agent_health":{**health["champion"],"candidate":health["candidate"]},
                "feed_rows":self._market_row_count(data),"configured_instruments":len(instruments),
                "markets":_market_overview(instruments,list(latest.values()),fresh),"instruments":instrument_status,
                "provider":provider,"feed_metrics":feed,"metrics":metrics,"backtest":{},
                "autonomy_enabled":self.autonomy_enabled,"paper_enabled":self.autonomy_enabled,
                "observe_enabled":self.observe_enabled,"learning_enabled":self.learning_enabled,"learning":learning,
                "decisions":decisions,"model_directions":{s:{"BUY":1,"HOLD":0,"SELL":-1}.get(d["action"],0) for s,d in latest.items()},
                "positions":positions,"paper_positions":positions,"paper_financials":books,"paper_account":ledgers["champion"],
                "candidate_live_account":{**accounts["candidate"],"available":bool(accounts["candidate"]["books"]),"status":cr["status"]},
                "account_observability":{**accounts,"fee_rate":self.fee,"real_orders_enabled":False},
                "live_account_comparison":{"score_is_promotion_gate":False,"reason_not_a_fair_score":"승급은 조립 시험의 같은 시장·비용 구간으로 비교합니다."},
                "validation_comparison":trial,"daily_cycle":{"automatic_account_reset":False,"history":experiment.get("history",[])},
                "universe_expansion":settings.get("universe_expansion",{}),"output_diagnostics":{},
                "input_availability":{"configured":len(instruments),"fresh":len(fresh),
                    "stored":input_availability(quotes.values()),"fresh_quotes":input_availability(q for s,q in quotes.items() if s in fresh)},
                "champion_version":ch["optimizer_updates"],"gpu":learning["gpu_device"] or "모델 정지",
                "physical_gpu":self._physical_gpu(),"real_orders_enabled":False,"logs":"\n".join(self.log_tail[-8:])}
