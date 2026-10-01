"""Dashboard status and cached measurements."""
from __future__ import annotations
import csv
from collections import deque
from contextlib import closing
import os
import sqlite3
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from ..paper_account import KR_SELL_TAX_ASSUMPTION, SEED_CASH
from ..account_diagnostics import summarize_account, valid_bid_ask_count, input_availability
from ..operating_rules import daily_boundary
from ..state_io import atomic_json
from .health import _agent_progress_health, _candidate_progress_health, _json, _market_group, _market_overview

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

    def _daily_cycle_status(self):
        profile=self.profile or self.runtime/self.mode
        path=profile/"agent"/"daily_account_summary.json"
        key,next_reset=daily_boundary(hour=int(self.operating_rules["account_reset_hour_kst"]))
        saved=_json(path)
        if not saved:
            # Enabling the schedule does not pretend old cumulative results
            # are a complete day, or reset the current account mid-session.
            saved={"session_key":key,"last_reset_utc":None,"history":[]}
            atomic_json(saved,path)
        return {**saved,"current_session_key":key,"next_reset_utc":next_reset,
            "hour_kst":int(self.operating_rules["account_reset_hour_kst"]),
            "in_progress":self.daily_cycle_pending,
            "live_accounts_preserved":not self.operating_rules.get("daily_reset_live_accounts",False),
            "order":"finish frozen daily competition; preserve long-term live accounts"}

    def status(self) -> dict:
        with self.lock:
            profile = self.profile or (self.runtime / self.mode)
            state, data = profile / "agent", profile / "market.csv"
            metrics = _json(state / "metrics.json")
            applied_rules=metrics.get("runtime_updates",{}).get("applied_rules")
            if applied_rules: self.operating_rules=dict(applied_rules)
            feed_metrics = _json(profile / ("live_feed_metrics.json" if self.mode == "live" else "mock_feed_metrics.json"))
            instrument_settings=_json(self.config)
            instruments=instrument_settings.get("instruments",[])
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
            # The Candidate worker publishes independently of Champion's
            # metrics writer. Read its committed cursor and the durable queue
            # directly so catch-up is visible while the main worker is busy.
            replay_path=state/"replay.sqlite3"
            if replay_path.exists():
                try:
                    with closing(sqlite3.connect(replay_path.resolve().as_uri()+"?mode=ro",uri=True,timeout=.2)) as db:
                        pending,oldest=db.execute("SELECT COUNT(*),MIN(stamp) FROM market_observations").fetchone()
                        totals=dict(db.execute("SELECT name,total FROM observation_counts"))
                        origins=dict(db.execute("SELECT role,created FROM origin_counts"))
                    metrics["shared_observation"]={"pending":pending,"oldest":oldest,
                        "common":totals.get("common",0),"candidate_completed":totals.get("candidate",0),
                        "experience_origins":origins}
                except sqlite3.Error:
                    pass  # Startup or schema migration: retain last known values.
            if candidate_observer_state.get("observation_profile"):
                metrics["candidate_live_observation_profile"]=candidate_observer_state["observation_profile"]
            observer_state_path=state/"candidate_observer_state.json"
            metrics_state_path=state/"metrics.json"
            observer_runtime_is_newer=(observer_state_path.exists() and
                (not metrics_state_path.exists() or
                 observer_state_path.stat().st_mtime_ns>metrics_state_path.stat().st_mtime_ns))
            if observer_runtime_is_newer and candidate_observer_state.get("learning_live_priority_enabled"):
                metrics["learning_priority"]=candidate_observer_state.get("gpu_scheduler",{}).get(
                    "policy","live_inference_first_then_complete_replay_coverage")
                for field in ("learning_wait_reason","candidate_training","champion_training","gpu_scheduler"):
                    if field in candidate_observer_state:
                        metrics[field]=candidate_observer_state[field]
            candidate_observer_books = summarize_account(candidate_observer_account)["books"]
            candidate_live_account = {
                "available": (bool(candidate_observer_books) and
                    candidate_observer_state.get("status") in
                    ("observing", "training_and_observing", "context_only", "judgment_paused")),
                "status": candidate_observer_state.get("status", "waiting_for_candidate_update"),
                "last_timestamp": candidate_observer_state.get("last_timestamp",
                    candidate_observer_account.get("last_timestamp")),
                "last_observation_timestamp": candidate_observer_state.get("last_timestamp",
                    candidate_observer_account.get("last_timestamp")),
                "last_full_decision_timestamp": candidate_observer_state.get("last_full_decision_timestamp"),
                "candidate_version": candidate_observer_state.get("candidate_version"),
                "candidate_training": bool(metrics.get("candidate_training")),
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
                "bars_required": int(self.operating_rules["validation_min_market_minutes"]),
                "snapshot_version": metrics.get("candidate_validation_snapshot_version",
                                                 validation_state.get("source_candidate_version")),
                "champion_snapshot_version":validation_state.get("source_champion_version"),
                "started_utc":validation_state.get("started_utc"),
                "promotion_schedule":self.operating_rules["promotion_schedule"],
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
                "shared_observation":metrics.get("shared_observation",{}),
                "last_bar_timestamps_equal": (
                    bool(paper_account.get("last_timestamp")) and
                    paper_account.get("last_timestamp") ==
                    candidate_observer_account.get("last_timestamp")),
                "reason_not_a_fair_score": (
                    "운영 계좌는 계속 학습하며 바뀐 여러 모델 버전의 결과를 누적합니다. "
                    "승급 점수는 별도 고정본의 하루 승급전만 사용합니다."),
            }
            champion_summary=summarize_account(paper_account)
            paper_financials=champion_summary["books"]
            paper_positions={f"{currency}:{position['symbol']}":{**position,"currency":currency}
                for currency,book in paper_financials.items() for position in book["positions"]}
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
            learning_min_holdout = int(self.operating_rules["validation_min_market_minutes"])
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
            agent_health["candidate"]=_candidate_progress_health(agent_health,metrics,agent_process_running,candidate_observer_state)
            agent_running=(agent_process_running and agent_health["status"]=="healthy")
            candidate_summary=summarize_account(candidate_observer_account)
            # Compatibility fields and the new view share exactly one ledger formula.
            paper_financials=champion_summary["books"]
            candidate_live_account["books"]=candidate_summary["books"]
            account_observability={
                "champion":{**champion_summary,
                    "training":bool(agent_process_running and metrics.get("champion_training")),
                    "version":metrics.get("champion_training_version",0),
                    "inference_count":metrics.get("champion_live_inference_count",0),
                    "last_inference_seconds":metrics.get("champion_live_last_inference_seconds"),
                    "skipped_observations":0,
                    "policy":metrics.get("champion_policy_diagnostics",{}),
                    "last_tradable_policy":metrics.get("champion_last_tradable_policy_diagnostics",{})},
                "candidate":{**candidate_summary,
                    "training":bool(agent_process_running and metrics.get("candidate_training")),
                    "version":candidate_observer_state.get("candidate_version"),
                    "inference_count":metrics.get("candidate_live_inference_count",0),
                    "last_inference_seconds":candidate_observer_state.get("last_inference_seconds"),
                    "inference_profile":candidate_observer_state.get("inference_profile",{}),
                    "skipped_observations":metrics.get("candidate_live_queue_drops",0),
                    "observer_status":candidate_observer_state.get("status"),
                    "observer_error":candidate_live_account["error"],
                    "last_full_decision_timestamp":candidate_observer_state.get("last_full_decision_timestamp"),
                    "policy":candidate_observer_state.get("policy_diagnostics",{}),
                    "last_tradable_policy":candidate_observer_state.get(
                        "last_tradable_policy_diagnostics",{})},
                "fee_rate":float(metrics.get("fee_rate",self.fee)),
                "slippage_bps":float(metrics.get("slippage_bps",1.0)),
                "krw_sell_tax_assumption":KR_SELL_TAX_ASSUMPTION,
                "usd_round_trip_cost_rate_before_spread":2*(
                    float(metrics.get("fee_rate",self.fee))+
                    float(metrics.get("slippage_bps",1.0))/10000),
                "quoted_bid_ask_symbols":valid_bid_ask_count(latest_quotes.values()),
                "quoted_bid_ask_scope":"latest stored quotes; not necessarily fresh",
                "minute_bar_input":True,"full_orderbook_available":False,
                "training_experience_source":"both_independent_paper_accounts",
                "shared_observation":metrics.get("shared_observation",{}),
                "live_policy_mode":"probability_sampling",
                "validation_policy_mode":"highest_probability_no_exploration",
                "reward_horizon":self.horizon,
                "reward_credit":metrics.get("reward_credit"),
                "reward_horizon_scope":"legacy_short_outcomes_only_when_future_credit_enabled",
                "real_orders_enabled":False}
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
                    "learning_enabled":self.learning_enabled,
                    "learning": {
                        "candidate_learning_enabled": metrics.get("candidate_learning_enabled", True),
                        "champion_learning_enabled":metrics.get("champion_learning_enabled",False),
                        "dual_learning_enabled":metrics.get("dual_learning_enabled",False),
                        "champion_training":metrics.get("champion_training",False),
                        "champion_model_version":metrics.get("champion_training_version",0),
                        "champion_remaining":metrics.get("champion_eligible_replay_count",0),
                        "eligible_backlog":metrics.get("replay_eligible_backlog",learning_replay),
                        "replay_pending_count":metrics.get("replay_pending_count",0),
                        "replay_passes_per_model":metrics.get("candidate_replay_passes"),
                        "reward_credit":metrics.get("reward_credit"),
                        "multiscale_input_status":metrics.get("multiscale_input_status",{}),
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
                    "daily_cycle":self._daily_cycle_status(),
                    "universe_expansion":instrument_settings.get("universe_expansion",{}),
                    "account_observability":account_observability,
                    "input_availability":{
                        "configured":len(instruments),"fresh":len(fresh_symbols),
                        "model_input":metrics.get("model_input_symbol_count"),
                        "configured_tradable":sum(item.get("asset_class") in ("equity","etf") for item in instruments),
                        "context_only":sum(item.get("asset_class") not in ("equity","etf") for item in instruments),
                        "stored":input_availability(row for symbol,row in latest_quotes.items()
                                                     if symbol in {str(item.get("symbol", "")) for item in instruments}),
                        "fresh_quotes":input_availability(row for symbol,row in latest_quotes.items() if symbol in fresh_symbols),
                        "second_resolution":{"1s":False,"15s":False,"30s":False}},
                    "real_orders_enabled": False,
                    "physical_gpu": self._physical_gpu(),
                    "logs": "\n".join(status_logs) or "Broker API is not connected; orders remain OFF."}
