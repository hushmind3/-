"""Shared, explicit operating settings and the KST daily boundary."""
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SEOUL = ZoneInfo("Asia/Seoul")
RULES_PATH = Path(__file__).resolve().parents[2]/"configs"/"online_learning.json"


def operating_rules(path=None):
    path=Path(path) if path is not None else RULES_PATH
    rules=json.loads(path.read_text(encoding="utf-8"))
    if not 0 <= int(rules["account_reset_hour_kst"]) <= 23:
        raise ValueError("account_reset_hour_kst must be 0..23")
    if int(rules["validation_min_market_minutes"]) < 1:
        raise ValueError("validation_min_market_minutes must be positive")
    for key in ("training_batch_size","training_optimizer_steps"):
        if int(rules[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if int(rules.get("reward_credit_observations",60)) < 1:
        raise ValueError("reward_credit_observations must be positive")
    if int(rules.get("reward_credit_seconds",3600)) < 1:
        raise ValueError("reward_credit_seconds must be positive")
    if float(rules.get("reward_credit_discount",1.0)) != 1.0:
        raise ValueError("net-equity reward uses undiscounted credit (1.0)")
    for key,default in (("goal_target_multiple",10.0),("goal_win_bonus_points",100.0)):
        value=float(rules.get(key,default))
        if not math.isfinite(value) or value <= (1.0 if key=="goal_target_multiple" else 0.0):
            raise ValueError(f"{key} must be finite and positive (target > 1)")
    return rules


def daily_boundary(now=None, hour=7):
    local=(now or datetime.now(timezone.utc)).astimezone(SEOUL)
    boundary=local.replace(hour=hour,minute=0,second=0,microsecond=0)
    if local < boundary:
        boundary-=timedelta(days=1)
    return boundary.date().isoformat(), (boundary+timedelta(days=1)).astimezone(timezone.utc).isoformat()
