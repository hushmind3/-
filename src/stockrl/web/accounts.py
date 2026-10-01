"""Explicit paper-account reset workflow."""
from __future__ import annotations
import json
import os
import time
from datetime import datetime, timezone
from ..account_diagnostics import summarize_account
from ..state_io import atomic_json
from .health import _json

class _AccountResetMixin:
    def reset_paper_accounts(self, daily=False, session_key=None) -> dict:
        with self.account_reset_lock:
            return self._reset_paper_accounts(daily,session_key)

    def _reset_paper_accounts(self, daily=False, session_key=None) -> dict:
        """Finish a daily competition or explicitly reset both live ledgers."""
        with self.lock:
            if self.stopping:
                return {"error": "System is stopping; wait before resetting accounts."}
            was_running = self.run_requested
            mode, horizon = self.mode, self.horizon
            profile = self.profile or (self.runtime / self.mode)

        if was_running:
            if daily:
                (profile/"agent"/"daily_cycle.completed.json").unlink(missing_ok=True)
                atomic_json({"session_key":session_key},profile/"agent"/"daily_cycle.request")
            self.stop()
            deadline = time.monotonic() + 120.0
            while time.monotonic() < deadline:
                with self.lock:
                    stopped = not self.stopping and not self.run_requested
                if stopped:
                    break
                time.sleep(0.1)
            else:
                return {"error": "Live workers did not stop; paper accounts were not reset."}
            if daily:
                completed=_json(profile/"agent"/"daily_cycle.completed.json")
                if completed.get("session_key")!=session_key:
                    return {"error":"Daily competition did not finish; paper accounts were not reset."}

        state = profile / "agent"
        state.mkdir(parents=True, exist_ok=True)
        from ..paper_account import PaperAccount

        if daily and not self.operating_rules.get("daily_reset_live_accounts",False):
            # The frozen competition ended above. Preserve live capital,
            # positions, goal progress and pending experiences across days.
            cycle=self._daily_cycle_status()
            completed_utc=datetime.now(timezone.utc).isoformat()
            accounts={role:_json(state/filename) for role,filename in (
                ("champion","paper_account.json"),("candidate","candidate_observer_account.json"))}
            record={"session":cycle["session_key"],"ended_utc":completed_utc,
                "reason":"scheduled_daily","live_accounts_preserved":True,
                "validation":{key:value for key,value in _json(state/"candidate_validation.json").items()
                    if key in ("status","reason","bars","candidate_score","champion_score","source_candidate_version","source_champion_version")},
                "accounts":{role:{currency:{key:book.get(key) for key in (
                    "initial_cash","equity","net_pnl","net_return_rate","costs","trade_count","position_count")}
                    for currency,book in summarize_account(account)["books"].items()}
                    for role,account in accounts.items()}}
            history=(cycle.get("history",[])+[record])[-int(self.operating_rules["daily_history_limit"]):]
            atomic_json({"session_key":session_key or cycle["current_session_key"],
                "last_reset_utc":cycle.get("last_reset_utc"),"last_competition_utc":completed_utc,
                "history":history,"live_accounts_preserved":True},state/"daily_account_summary.json")
            if was_running:
                result=self.start(mode,horizon)
                if not result.get("ok"):
                    return {"error":"Daily competition ended; live accounts preserved; restart failed: "+str(result.get("error"))}
            return {"ok":True,"reset":[],"live_accounts_preserved":True,"system_restarted":was_running,
                "message":"Daily frozen competition finished; long-term live accounts and experiences preserved."}

        old_accounts={role:_json(state/filename) for role,filename in (
            ("champion","paper_account.json"),("candidate","candidate_observer_account.json"))}
        if (state/"replay.sqlite3").is_file():
            from ..replay_store import GlobalReplayBuffer
            replay=GlobalReplayBuffer(journal_path=state/"replay.sqlite3",dual_learning=True)
            if replay.market_observation_stats()["pending"]:
                # A forced worker stop may leave compulsory observations awaiting
                # their original account episode. Do not change that episode.
                if was_running:
                    self.start(mode,horizon)
                return {"error":"Candidate observations are still pending; accounts were not reset. Retry after the observation queue drains."}
            regular=replay.load_pending("regular"); portfolio=replay.load_pending("portfolio")
            old_ledger=PaperAccount(state/"paper_account.json",self.fee,0.0)
            for experience in portfolio:
                if experience.get("reset_terminal"):
                    continue
                experience["reset_terminal"]=True
                experience["reset_equity"]=old_ledger.normalized_equity()
                experience["reset_goal_points"]=old_ledger.goal_points()
                experience["reset_symbol_net_pnl"]=old_ledger.symbol_net_pnl(experience.get("symbol",""))
                if not experience.get("fill_seen"):
                    experience["fill_expected"]=False; experience["trade_executed"]=False
            replay.save_pending(regular,portfolio)
            observer_pending=replay.load_pending("candidate_portfolio")
            old_observer=PaperAccount(state/"candidate_observer_account.json",self.fee,0.0)
            for experience in observer_pending:
                if experience.get("reset_terminal"):
                    continue
                experience["reset_terminal"]=True
                experience["reset_equity"]=old_observer.normalized_equity()
                experience["reset_goal_points"]=old_observer.goal_points()
                experience["reset_symbol_net_pnl"]=old_observer.symbol_net_pnl(experience.get("symbol",""))
                if not experience.get("fill_seen"):
                    experience["fill_expected"]=False;experience["trade_executed"]=False
            replay.save_pending_kind("candidate_portfolio",observer_pending)
        cycle=self._daily_cycle_status()
        completed_utc=datetime.now(timezone.utc).isoformat()
        record=cycle.get("pending_record") or {"session":cycle["session_key"],"ended_utc":completed_utc,
            "reason":"scheduled_daily" if daily else "manual",
            "validation":{key:value for key,value in _json(state/"candidate_validation.json").items()
                if key in ("status","reason","bars","candidate_score","champion_score","source_candidate_version","source_champion_version")},
            "accounts":{role:{currency:{key:book.get(key) for key in (
                "initial_cash","equity","net_pnl","net_return_rate","costs","fees","sell_tax","spread","slippage","trade_count","position_count")}
                for currency,book in summarize_account(account)["books"].items()}
                for role,account in old_accounts.items()}}
        history=(cycle.get("history",[])+[record])[-int(self.operating_rules["daily_history_limit"]):]
        atomic_json({"session_key":cycle["session_key"],"last_reset_utc":cycle.get("last_reset_utc"),
            "history":cycle.get("history",[]),"pending_record":record,
            "pending_session_key":session_key or cycle["current_session_key"]},state/"daily_account_summary.json")

        for filename in ("paper_account.json", "candidate_observer_account.json"):
            PaperAccount(state / filename, self.fee, 0.0).reset()
        if (state/"replay.sqlite3").is_file():
            replay.set_observer_account(PaperAccount(state/"candidate_observer_account.json",self.fee,0.0).state)

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
                         "last_inference_seconds": None, "error": None,
                         "policy_diagnostics":{},"last_tradable_policy_diagnostics":{}})
        temporary = observer_path.with_suffix(".json.reset.tmp")
        temporary.write_text(json.dumps(observer, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, observer_path)

        metrics_path = state / "metrics.json"
        metrics = _json(metrics_path)
        for role, filename in (("champion", "paper_account.json"),
                               ("candidate", "candidate_observer_account.json")):
            fresh_account = PaperAccount(state / filename, self.fee, 0.0)
            metrics[role + "_goal"] = fresh_account.goal_summary()
            metrics[role + "_reward_score"] = {
                "points": fresh_account.reward_points(), "change": None,
                "episode_id": fresh_account.state.get("episode_id"),
            }
        metrics.update({"paper_net_reward": 0.0, "paper_account_reward": 0.0,
                        "fee_total": 0.0, "slippage_total": 0.0,
                        "paper_account_reset_utc": datetime.now(timezone.utc).isoformat(),
                        "candidate_live_status": "waiting_for_candidate",
                        "candidate_live_last_timestamp": None,
                        "champion_policy_diagnostics":{},"champion_last_tradable_policy_diagnostics":{},
                        "candidate_last_tradable_policy_diagnostics":{}})
        temporary = metrics_path.with_suffix(".json.reset.tmp")
        temporary.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, metrics_path)
        atomic_json({"session_key":cycle.get("pending_session_key",session_key or cycle["current_session_key"]),
            "last_reset_utc":completed_utc,"history":history},state/"daily_account_summary.json")

        if was_running:
            result = self.start(mode, horizon)
            if not result.get("ok"):
                return {"error": "Accounts reset, but system restart failed: " +
                        str(result.get("error", "unknown error")), "reset": True}
        return {"ok": True, "reset": ["champion", "candidate_observer"],
                "system_restarted": was_running,
                "message": "Champion and Candidate observer paper accounts reset to seed cash."}
