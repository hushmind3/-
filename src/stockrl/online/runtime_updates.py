"""Apply operational settings and learner code between saved training rounds.

Models, accounts, replay and inference threads keep their current state.
Architecture and active episode goals require their own explicit migration.
"""
import ast
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from ..operating_rules import RULES_PATH, operating_rules

LIVE_RULES = frozenset(("training_batch_size", "training_optimizer_steps",
    "preopen_learning_priority", "reward_credit_observations", "reward_credit_seconds",
    "reward_credit_discount", "validation_min_market_minutes", "promotion_schedule",
    "account_reset_hour_kst", "daily_history_limit", "daily_reset_live_accounts"))


class RuntimeUpdates:
    def __init__(self, agent, rules_path=None, watch_code=True):
        self.path = Path(rules_path or RULES_PATH)
        self.stamp = self.path.stat().st_mtime_ns
        self.last_check = 0.0
        self.code = {}
        if watch_code:
            for name in ("stockrl.online.losses", "stockrl.online.learning", "stockrl.online.rewards", "stockrl.online.replay_updates"):
                module = sys.modules.get(name)
                if module is not None:
                    path = Path(module.__file__)
                    self.code[name] = (path, path.read_text(encoding="utf-8"),path.stat().st_mtime_ns)
        agent.metrics["runtime_updates"] = {"status":"ready", "restart_required":False,
            "applied_rules":dict(agent.operating_rules), "deferred_rules":{},
            "code_modules":list(self.code), "applied_code":[]}
        self.accepted_signature=self.signature()

    def signature(self):
        return (self.path.stat().st_mtime_ns,tuple((name,path.stat().st_mtime_ns) for name,(path,_,_) in self.code.items()))

    def pending_change(self):
        return self.signature()!=getattr(self,"accepted_signature",None)

    def poll(self, agent, force=False):
        if not force and time.monotonic() - self.last_check < 1.0:
            return
        self.last_check = time.monotonic()
        status = agent.metrics["runtime_updates"]
        changed_status=False
        try:
            # Adopt auxiliary monitoring when this controller is hot-applied.
            if "stockrl.online.learning" in self.code:
                for module_name in ("stockrl.online.rewards", "stockrl.online.replay_updates"):
                    if module_name not in self.code:
                        original=sys.modules[module_name]
                        self.code[module_name]=(Path(original.__file__),"",-1)
                        status["code_modules"]=list(self.code)
            stamp = self.path.stat().st_mtime_ns
            if stamp != self.stamp:
                desired = operating_rules(self.path)
                effective = dict(agent.operating_rules)
                deferred = {k:v for k,v in desired.items() if k not in LIVE_RULES and effective.get(k)!=v}
                effective.update({k:v for k,v in desired.items() if k in LIVE_RULES})
                agent.batch_size = int(effective["training_batch_size"])
                agent.updates_per_candidate = int(effective["training_optimizer_steps"])
                agent.reward_credit_observations = int(effective.get("reward_credit_observations",60))
                agent.reward_credit_seconds = int(effective.get("reward_credit_seconds",3600))
                # A trial already in progress keeps its original comparison rules.
                if not agent.validation_active:
                    agent.validation_window_bars = int(effective["validation_min_market_minutes"])
                    agent.daily_promotion = effective["promotion_schedule"]=="daily"
                else:
                    for key in ("validation_min_market_minutes", "promotion_schedule"):
                        if effective.get(key)!=agent.operating_rules.get(key):
                            deferred[key] = effective[key]
                            effective[key] = agent.operating_rules[key]
                scheduler = getattr(agent,"candidate_live_inference_lock",None)
                if scheduler is not None:
                    scheduler.preopen_learning = bool(effective.get("preopen_learning_priority",False))
                agent.operating_rules = effective
                # Retry deferred trial rules when the active trial finishes.
                self.stamp = None if any(key in deferred for key in ("validation_min_market_minutes","promotion_schedule")) else stamp
                status.update(applied_rules=dict(effective), deferred_rules=deferred,
                    restart_required=False, migration_required=bool(deferred), status="applied", error=None,
                    applied_utc=datetime.now(timezone.utc).isoformat())
                changed_status=True
            changed = {name:(path,path.read_text(encoding="utf-8"),path.stat().st_mtime_ns)
                       for name,(path,old,stamp) in self.code.items() if path.stat().st_mtime_ns!=stamp}
            if changed:
                if "stockrl.online.learning" in changed:
                    def loop_body(source):
                        tree=ast.parse(source)
                        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="_LearningMixin")
                        return ast.dump(next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=="_learner"))
                    if loop_body(changed["stockrl.online.learning"][1])!=loop_body(self.code["stockrl.online.learning"][1]):
                        raise ValueError("learner control-loop edits require a worker restart; calculation methods support live updates")
                self._apply_code(agent,changed)
                self.code.update(changed)
                status.update(applied_code=list(changed), status="applied", error=None,
                    applied_utc=datetime.now(timezone.utc).isoformat())
                changed_status=True
        except Exception as exc:
            # Keep the last valid settings/code. Never restart or discard replay.
            status.update(status="rejected", error=f"{type(exc).__name__}: {exc}"[:800])
            changed_status=True
        self.accepted_signature=self.signature()
        return changed_status

    @staticmethod
    def _apply_code(agent, changed):
        staged = {}
        for name,(path,source,stamp) in changed.items():
            ast.parse(source)
            original = sys.modules[name]
            module = types.ModuleType(name)
            module.__dict__.update(__package__=original.__package__, __file__=str(path))
            exec(compile(source,str(path),"exec"),module.__dict__)
            staged[name] = module
        reward_module=staged.get("stockrl.online.rewards")
        if reward_module is not None:
            old_rewards=sys.modules["stockrl.online.rewards"]._RewardMixin
            for name,value in vars(reward_module._RewardMixin).items():
                if callable(value) or isinstance(value,(staticmethod,classmethod)):
                    setattr(old_rewards,name,value)
        learning = sys.modules["stockrl.online.learning"]
        replay_updates=staged.get("stockrl.online.replay_updates")
        if replay_updates is not None:
            install=replay_updates.install_replay_updates
            sys.modules["stockrl.online.replay_updates"].install_replay_updates=install
            learning.install_replay_updates=install
            if agent is not None and getattr(agent,"replay",None) is not None:
                agent.replay._performance_code_stamp=None
                install(agent.replay)
            learning._LearningMixin._train_model.__globals__["install_replay_updates"]=install
        losses = staged.get("stockrl.online.losses",sys.modules["stockrl.online.losses"])
        updated = staged.get("stockrl.online.learning")
        if updated is not None:
            if replay_updates is not None:
                updated.install_replay_updates=replay_updates.install_replay_updates
            updated.shared_experience_losses = losses.shared_experience_losses
            # Rebind methods on the existing mixin; the current learner thread
            # and all inference/model objects retain their identity.
            old = learning._LearningMixin
            for name,value in vars(updated._LearningMixin).items():
                if callable(value) or isinstance(value,(staticmethod,classmethod)):
                    setattr(old,name,value)
        else:
            learning.shared_experience_losses = losses.shared_experience_losses
            learning._LearningMixin._train_model.__globals__["shared_experience_losses"] = losses.shared_experience_losses
        if "stockrl.online.losses" in staged:
            sys.modules["stockrl.online.losses"].shared_experience_losses = losses.shared_experience_losses
