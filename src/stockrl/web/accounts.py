"""Explicit long-term MoE ledger reset; assembly trial accounts stay separate."""
import time
from datetime import datetime, timezone
from ..state_io import atomic_json, read_json
from ..paper_account import PaperAccount
from ..replay_store import GlobalReplayBuffer

class _AccountResetMixin:
    def reset_paper_accounts(self) -> dict:
        with self.account_reset_lock:
            with self.lock:
                if self.stopping:return {"error":"System is stopping; wait before resetting accounts."}
                roles=[role for role in self.model_enabled if self.model_enabled[role]]
                workers=[self._moe_model_worker(role) for role in self.model_enabled]
                # A trial owns the slot, but its ledger is under assembly/<id>/.
                for worker in workers:worker.stop()
                self.model_enabled=dict.fromkeys(self.model_enabled,False)
                self._write_autonomy()
            deadline=time.monotonic()+120
            while any(worker.process() for worker in workers):
                if time.monotonic()>=deadline:return {"error":"Workers have not saved; accounts were not reset."}
                time.sleep(.1)
            reset=[]
            for role,worker in zip(self.model_enabled,workers):
                path=worker.state/"paper_account.json"
                if not path.exists():continue
                account=PaperAccount(path,self.fee,.0001)
                journal=worker.state/"replay.sqlite3"
                if journal.exists():
                    replay=GlobalReplayBuffer(journal_path=journal,dual_learning=False)
                    pending=replay.load_pending("candidate_portfolio")
                    for decision in pending:
                        decision.update(reset_terminal=True,reset_equity=account.normalized_equity(),
                            reset_goal_points=account.goal_points(),
                            reset_symbol_net_pnl=account.symbol_net_pnl(decision.get("symbol","")))
                        if not decision.get("fill_seen"):
                            decision.update(fill_expected=False,trade_executed=False)
                    replay.save_pending_kind("candidate_portfolio",pending)
                account.reset()
                saved=read_json(worker.state/"worker_status.json")
                # Preserve optimizer/replay counters, discard stale account snapshots.
                saved.update(books=account.snapshot()["books"],fills=[],pending_orders={},
                    reward_points=account.reward_points(),decision={},status="stopped",
                    account_reset_at=datetime.now(timezone.utc).isoformat())
                atomic_json(saved,worker.state/"worker_status.json")
                reset.append(role)
            for role in roles:
                worker=self._moe_model_worker(role)
                if worker.runner_script!="run_assembly_trial.py":self.set_model(role,True)
            return {"ok":True,"reset":reset,"trial_accounts_preserved":True,
                "message":"장기 운영계좌 초기화 완료 · 조립 시험계좌와 학습 state 유지"}
