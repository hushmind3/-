"""Executed outcome rewards and delayed action credit."""
from __future__ import annotations
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import Counter
import numpy as np
from .market_panel import ACTION_NAMES, GlobalMarketPanel
from .paper_account import _currency
from .experience import Experience, REWARD_VERSION

def net_action_reward(action:int, forward_return:float, previous_position:int,
                      fee:float, slippage:float) -> tuple[float,float,float]:
    """One equal-notional paper trade held to horizon; charge entry and exit costs."""
    position=int(action)-1
    # Each decision is evaluated as an isolated paper trade over its configured
    # horizon. BUY/SELL pay for opening and closing; HOLD stays in cash.
    turnover=2 if position else 0
    gross=float(position)*float(forward_return)
    fee_cost=float(fee)*turnover; slippage_cost=float(slippage)*turnover
    return gross-fee_cost-slippage_cost,fee_cost,slippage_cost


def closed_market_credit_ready(dec, panel, index, allow_unfilled=False):
    """Settle elapsed wall-clock credit at a real closing mark, never an outage.

    Korea includes the extended venue until 20:00 KST; US equities include the
    existing pre/after-market range until 20:00 New York time. A five-minute
    buffer avoids treating the final bars as a completed session. No synthetic
    quote or fabricated price movement is introduced.
    """
    seconds = int(dec.get("credit_seconds") or 0)
    if not seconds or (not allow_unfilled and dec.get("fill_expected") and not dec.get("fill_seen")):
        return False
    symbol = dec.get("symbol")
    if symbol not in panel.symbols:
        return False
    market, asset = panel.groups[symbol]
    if asset != "equity" or market not in ("KRX", "KOSDAQ", "US"):
        return False
    current = panel.dates[index]
    start = np.datetime64(dec.get("fill_timestamp", dec["timestamp"]))
    if current < start + np.timedelta64(seconds, "s"):
        return False
    local = datetime.fromtimestamp(float(current.astype("datetime64[s]").astype(int)), timezone.utc).astimezone(
        ZoneInfo("America/New_York" if market == "US" else "Asia/Seoul"))
    minute = local.hour * 60 + local.minute
    opening = 4 * 60 if market == "US" else 8 * 60
    if local.weekday() < 5 and opening <= minute < 20 * 60 + 5:
        return False
    symbol_index = panel.symbols.index(symbol)
    seen = np.flatnonzero(panel.observed[:index+1, symbol_index])
    return bool(len(seen) and panel.dates[seen[-1]] >= start
                and np.isfinite(panel.closes[index, symbol_index])
                and panel.closes[index, symbol_index] > 0)



def saved_closing_panel(pending, live_panel, index):
    """Recover a removed venue from its latest durable, actually observed input.

    The saved decision price is the real observed close, not a synthetic quote.
    Reuse the complete shared input for successor value; never mix today's US
    state into yesterday's Korean closing input or create a terminal episode.
    """
    from types import SimpleNamespace
    missing=[d for d in pending if d.get("symbol") not in live_panel.symbols
             and str(d.get("symbol","")).endswith((".KS",".KQ"))
             and "features" in d and d.get("input_symbols")]
    if not missing:
        return None
    usable=[d for d in missing if d.get("entry_price",0)>0
            and d["valid_mask"][-1,d["input_symbols"].index(d["symbol"])]]
    if not usable:
        return None
    sources=[d for d in usable if not (d.get("fill_expected") and not d.get("fill_seen"))]
    latest=max(sources or usable,key=lambda d:d["timestamp"])
    stamp=latest["timestamp"]; symbols=latest["input_symbols"]
    quotes={}
    for dec in sorted(usable,key=lambda d:d["timestamp"]):
        if dec["timestamp"]<=stamp:
            quotes[dec["symbol"]]=(float(dec["entry_price"]),dec["timestamp"])
    dates=np.asarray(sorted({v[1] for v in quotes.values()}|{str(live_panel.dates[index])}),dtype="datetime64[ns]")
    observed=np.zeros((len(dates),len(symbols)),dtype=bool)
    closes=np.full(observed.shape,np.nan)
    groups={}
    for j,symbol in enumerate(symbols):
        groups[symbol]=("KOSDAQ" if symbol.endswith(".KQ") else "KRX","equity")
        if symbol in quotes and latest["valid_mask"][:,j].any():
            price,quoted=quotes[symbol]
            row=int(np.searchsorted(dates,np.datetime64(quoted)))
            observed[row,j]=True; closes[row:,j]=price
    successor=Experience(latest["features"],latest["symbol_ids"],latest["market_ids"],
        latest["asset_ids"],latest["valid_mask"],0,1,0.,stamp,
        market_context=latest.get("market_context"),multiscale_state=latest.get("multiscale_state"),
        daily_history=latest.get("daily_history"),portfolio_state=latest.get("portfolio_state"),
        account_state=latest.get("account_state"),goal_state=latest.get("goal_state"))
    return SimpleNamespace(symbols=symbols,groups=groups,
        dates=dates,
        observed=observed,closes=closes,saved_reward_input=successor,
        reward_equity=float(latest.get("equity_before",0)))


class _RewardMixin:
    def _target_index(self,panel:GlobalMarketPanel,start:int,symbol:int)->int|None:
        future=np.flatnonzero(panel.observed[start+1:,symbol])+start+1
        if not len(future): return None
        if self.horizon_kind=="bars":
            return int(future[self.horizon_amount-1]) if len(future)>=self.horizon_amount else None
        target=panel.dates[start]+np.timedelta64(self.horizon_amount,"s")
        eligible=future[panel.dates[future]>=target]
        return int(eligible[0]) if len(eligible) else None

    def _mature(self,pending,panel,end_index):
        keep=[]
        for dec in pending:
            if dec.get("blocked_reason"):
                keep.append(dec); continue
            if "entry_price" in dec:
                # Keep the entry price with the decision so rolling the panel
                # cannot make the original row disappear before its outcome.
                stamp=np.datetime64(dec["timestamp"])
                now=panel.dates[end_index]
                symbol=dec.get("symbol")
                if symbol not in panel.symbols:
                    keep.append(dec); continue
                symbol_ix=panel.symbols.index(symbol)
                if now<=stamp or not panel.observed[end_index,symbol_ix]:
                    keep.append(dec); continue
                if self.horizon_kind=="bars":
                    dec["bars_elapsed"]=int(dec.get("bars_elapsed",0))+1
                    matured=dec["bars_elapsed"]>=self.horizon_amount
                else:
                    target=stamp+np.timedelta64(self.horizon_amount,"s")
                    matured=now>=target
                if not matured:
                    keep.append(dec); continue
                entry=float(dec["entry_price"]); exit_price=float(panel.closes[end_index,symbol_ix])
                if not np.isfinite(entry*exit_price) or entry<=0:
                    self.metrics["pending_expired_after_window"] = int(
                        self.metrics.get("pending_expired_after_window",0))+1
                    continue
                forward=exit_price/entry-1.0
                dec["index"]=end_index; dec["symbol_index"]=symbol_ix
            else:
                # Finite historical runs still refer to their fixed panel rows.
                if dec.get("timestamp"):
                    stamp=np.datetime64(dec["timestamp"])
                    start_ix=int(np.searchsorted(panel.dates,stamp,side="left"))
                    if start_ix>=len(panel.dates) or panel.dates[start_ix]!=stamp:
                        keep.append(dec); continue
                    symbol_ix=(panel.symbols.index(dec["symbol"])
                               if dec.get("symbol") in panel.symbols else dec["symbol_index"])
                    dec["index"]=start_ix; dec["symbol_index"]=symbol_ix
                symbol_ix=dec["symbol_index"]
                target=self._target_index(panel,dec["index"],symbol_ix)
                if target is None or end_index<target:
                    keep.append(dec); continue
                forward=panel.return_to(dec["index"],target,symbol_ix)
            i=symbol_ix
            self.metrics["matured"]+=1
            if dec.get("promotion_holdout",False):
                self.metrics["promotion_validation_outcomes"] = int(
                    self.metrics.get("promotion_validation_outcomes",0))+1
                self.replay.acknowledge_pending("regular",dec)
                continue
            if dec.get("is_validation",False):
                self._append_validation(self.validation,self.validation_dates,dec["timestamp"],
                    (dec["features"],dec["symbol_ids"],dec["market_ids"],dec["asset_ids"],
                    dec["valid_mask"],i,forward,dec["timestamp"],dec.get("previous_position",0),
                    dec.get("market_context")))
                self.replay.acknowledge_pending("regular",dec)
                continue
            action=dec["action"]
            reward,fee_cost,slippage_cost=net_action_reward(action,forward,dec.get("previous_position",0),
                                                            self.fee,self.slippage)
            self.metrics["paper_net_reward"]+=float(reward)
            action_name=ACTION_NAMES[action]
            rewards=self.metrics.setdefault("action_net_reward",{"SELL":0.0,"HOLD":0.0,"BUY":0.0})
            counts=self.metrics.setdefault("action_count",{"SELL":0,"HOLD":0,"BUY":0})
            rewards[action_name]=float(rewards.get(action_name,0.0))+float(reward)
            counts[action_name]=int(counts.get(action_name,0))+1
            self.metrics["fee_total"] = float(self.metrics.get("fee_total",0.0))+fee_cost
            self.metrics["slippage_total"] = float(self.metrics.get("slippage_total",0.0))+slippage_cost
            day=str(dec.get("timestamp","")[:10])
            daily=self.metrics.setdefault("daily_net_pnl",{})
            daily[day]=float(daily.get(day,0.0))+float(reward)
            # This isolated BUY/SELL proxy assumes a trade at the decision
            # price. It is useful as a diagnostic, but is not an executed
            # cash-only account transition, so never place it in training replay.
            self.replay.acknowledge_pending(
                "regular",dec)
        return keep

    def _mature_portfolio(self, pending, panel, end_index: int, filled_orders=(),account=None,origin_model="champion"):
        """Turn completed paper-account transitions into reward experiences.

        The reward uses the account's net-of-cost normalized equity change.
        Trade actions begin their horizon on the linked next-bar fill; HOLD
        observations use their decision time because they create no order.
        """
        account=account or self.paper_account
        account.observe_goal(str(panel.dates[end_index]))
        self.metrics[origin_model+"_goal"]=account.goal_summary()
        pending_kind="portfolio" if origin_model=="champion" else "candidate_portfolio"
        score=self.replay.record_account_score(origin_model,str(panel.dates[end_index]),
            account.state.get("episode_id","legacy"),account.reward_points())
        self.metrics[origin_model+"_reward_score"]=score
        if not pending:
            self.metrics[origin_model+"_pending_reward_status"] = {"total":0,"reasons":{},"oldest":None}
            return pending
        recovered=[]
        if not hasattr(panel,"saved_reward_input"):
            closing=saved_closing_panel(pending,panel,end_index)
            if closing is not None:
                repair=[d for d in pending if d.get("symbol") not in panel.symbols
                        and d.get("symbol") in closing.symbols
                        and closed_market_credit_ready(d,closing,len(closing.dates)-1,allow_unfilled=True)]
                if repair:
                    recovered=self._mature_portfolio(repair,closing,len(closing.dates)-1,filled_orders,account,origin_model)
                    repair_ids={id(d) for d in repair}
                    pending=[d for d in pending if id(d) not in repair_ids]
        now = float(getattr(panel,"reward_equity",account.normalized_equity()))
        fills_by_id={str(fill.get("decision_id")):fill for fill in filled_orders
                     if fill.get("decision_id")}
        fills_by_order={(str(fill.get("order_date")),str(fill.get("symbol"))):fill
                        for fill in filled_orders if fill.get("order_date")}
        keep=list(recovered); account_transition_added=set(); credit_successor=getattr(panel,"saved_reward_input",None)
        matured_experiences=[]; matured_acks=[]
        waiting = Counter()
        def retain(dec, reason):
            keep.append(dec)
            waiting[reason] += 1
        for dec in pending:
            if dec.get("blocked_reason"):
                retain(dec,"blocked"); continue
            if dec.get("reward_version")!=REWARD_VERSION:
                dec["blocked_reason"]="reward schema is incompatible"
                retain(dec,"blocked")
                continue
            stamp=np.datetime64(dec["timestamp"]); current=panel.dates[end_index]
            symbol=dec.get("symbol")
            if symbol not in panel.symbols:
                waiting_fill=dec.get("fill_expected") and not dec.get("fill_seen")
                retain(dec,"fill" if waiting_fill else "missing_market_input"); continue
            symbol_ix=panel.symbols.index(symbol)
            reset_terminal=bool(dec.get("reset_terminal"))
            goal_points=dec.get("reset_goal_points",account.goal_points()) if reset_terminal else account.goal_points()
            goal_before=dec.get("goal_points_before",{})
            same_goal_episode=reset_terminal or dec.get("goal_episode_id")==account.state.get("episode_id")
            goal_terminal=bool(goal_before and same_goal_episode and not dec.get("goal_complete_before")
                and all(goal_points.get(c,0)>0 for c in goal_points))
            terminal=reset_terminal or goal_terminal
            # A paper order with no next-bar fill expires with its session.
            # Record the real unexecuted result; never carry yesterday's order
            # into tomorrow or manufacture a fill to unblock learning.
            if dec.get("fill_expected") and not dec.get("fill_seen") and closed_market_credit_ready(dec,panel,end_index,allow_unfilled=True):
                order=account.state.get("pending",{}).get(symbol)
                if order and (order.get("decision_id")==dec.get("decision_id") or
                        (not order.get("decision_id") and order.get("date")==dec.get("timestamp"))):
                    account.state["pending"].pop(symbol,None)
                dec["fill_expected"]=False;dec["trade_executed"]=False
                dec["session_order_expired"]=True
                key=origin_model+"_expired_session_orders"
                self.metrics[key]=int(self.metrics.get(key,0))+1
            closed_mark = not terminal and closed_market_credit_ready(dec,panel,end_index)
            if not terminal and (current<=stamp or not panel.observed[end_index,symbol_ix]) and not closed_mark:
                retain(dec,"next_quote" if current>stamp else "reward_horizon"); continue
            input_symbols=dec.get("input_symbols")
            input_symbol_index=(input_symbols.index(symbol) if input_symbols and symbol in input_symbols
                                else int(dec["symbol_index"]))
            if not 0<=input_symbol_index<dec["features"].shape[1]:
                dec["blocked_reason"]="decision symbol is missing from its saved input"
                retain(dec,"blocked")
                continue
            if int(dec.get("action",1))!=1 and not dec.get("fill_expected"):
                # Still learn its zero executed outcome. Never credit unrelated
                # portfolio drift to an order that did not change this position.
                dec["trade_executed"]=False
            if dec.get("fill_expected") and not dec.get("fill_seen"):
                fill=(fills_by_id.get(str(dec.get("decision_id"))) or
                      fills_by_order.get((str(dec.get("timestamp")),str(symbol))))
                if fill is not None:
                    dec["fill_seen"]=True
                    dec["fill_timestamp"]=str(fill["date"])
                    dec["fill_price"]=float(fill["price"])
                    if not dec.get("credit_observations_target"):
                        dec["symbol_pnl_before"]=float(
                            fill.get("symbol_pnl_before_fill",dec.get("symbol_pnl_before",0.0)))
                    dec["bars_elapsed"]=0
                    retain(dec,"reward_horizon")
                    continue
                queued=account.state.get("pending",{}).get(symbol)
                still_queued=bool(queued and (
                    queued.get("decision_id")==dec.get("decision_id") or
                    (not queued.get("decision_id") and queued.get("date")==dec.get("timestamp"))))
                if still_queued:
                    retain(dec,"fill")
                else:
                    dec["fill_expected"]=False
                    dec["trade_executed"]=False
                    self.metrics["paper_unfilled_decisions"]=int(
                        self.metrics.get("paper_unfilled_decisions",0))+1
                if still_queued:
                    continue
            reward_start=np.datetime64(dec.get("fill_timestamp",dec["timestamp"]))
            if current<=reward_start and not terminal:
                retain(dec,"reward_horizon"); continue
            if dec.get("credit_observations_target"):
                if str(current)<=dec.get("credit_last_timestamp",str(stamp)) and not terminal:
                    retain(dec,"reward_horizon"); continue
                dec["credit_last_timestamp"]=str(current)
                dec["credit_observations_elapsed"]=int(dec.get("credit_observations_elapsed",0))+1
                matured=(current>=reward_start+np.timedelta64(int(dec["credit_seconds"]),"s")
                         if dec.get("credit_seconds") else
                         dec["credit_observations_elapsed"]>=int(dec["credit_observations_target"]))
            elif self.horizon_kind=="bars":
                dec["bars_elapsed"]=int(dec.get("bars_elapsed",0))+1
                matured=dec["bars_elapsed"]>=self.horizon_amount
            else:
                matured=current>=reward_start+np.timedelta64(self.horizon_amount,"s")
            if not matured and not terminal:
                retain(dec,"reward_horizon"); continue
            final_equity=float(dec.get("reset_equity",now)) if terminal else now
            account_reward = final_equity - float(dec.get("equity_before", final_equity))
            symbol_reward = ((float(dec["reset_symbol_net_pnl"]) if reset_terminal else account.symbol_net_pnl(symbol))
                             - float(dec.get("symbol_pnl_before", 0.0)))
            if dec.get("trade_executed") is False:
                symbol_reward=0.0
            if dec.get("promotion_holdout",False):
                self.metrics["promotion_validation_outcomes"] = int(
                    self.metrics.get("promotion_validation_outcomes",0))+1
                self.replay.acknowledge_pending(pending_kind,dec)
                continue
            entry=float(dec.get("fill_price",dec.get("entry_price",0.0)))
            exit_price=float(panel.closes[end_index,symbol_ix])
            forward_return=(exit_price/entry-1.0 if entry>0 and np.isfinite(entry*exit_price) else None)
            exp=Experience(dec["features"],dec["symbol_ids"],dec["market_ids"],dec["asset_ids"],
                    dec["valid_mask"],input_symbol_index,int(dec["action"]),float(symbol_reward),
                    dec["timestamp"],"paper_account_symbol",float(dec.get("regime",0.0)),
                    reward_version=REWARD_VERSION,
                    market_context=dec.get("market_context"),
                    multiscale_state=dec.get("multiscale_state"),
                    daily_history=dec.get("daily_history"),
                    portfolio_state=dec.get("portfolio_state"),
                    account_state=dec.get("account_state"),forward_return=forward_return,
                    behavior_log_prob=dec.get("behavior_log_prob"),
                    trade_executed=dec.get("trade_executed",True))
            if dec.get("is_validation",False):
                self._append_validation(self.portfolio_validation,self.portfolio_validation_dates,
                                        exp.timestamp,exp)
                self.replay.acknowledge_pending(pending_kind,dec)
            else:
                timestamp=dec["timestamp"]
                # Keep one allocation-credit row per symbol. It carries that
                # symbol's realized contribution separately from the shared
                # whole-account result; never attach the whole account return
                # to whichever symbol happened to mature first.
                first_account_transition=(timestamp not in account_transition_added and
                    not self.replay.has_portfolio_value(timestamp,origin_model))
                account_exp=Experience(dec["features"],dec["symbol_ids"],dec["market_ids"],dec["asset_ids"],
                    dec["valid_mask"],input_symbol_index,int(dec["action"]),float(symbol_reward),timestamp,
                    "paper_account_portfolio",float(dec.get("regime",0.0)),
                    reward_version=REWARD_VERSION,
                    market_context=dec.get("market_context"),
                    multiscale_state=dec.get("multiscale_state"),
                    daily_history=dec.get("daily_history"),
                    portfolio_state=dec.get("portfolio_state"),
                    account_state=dec.get("account_state"),portfolio_reward=float(account_reward),
                    portfolio_transition=True,portfolio_value_transition=first_account_transition,
                    forward_return=forward_return,
                    behavior_log_prob=dec.get("behavior_log_prob"),
                    trade_executed=dec.get("trade_executed",True))
                exp.origin_model=origin_model;account_exp.origin_model=origin_model
                if closed_mark:
                    last_quote = np.flatnonzero(panel.observed[:end_index+1,symbol_ix])[-1]
                    for experience in (exp,account_exp):
                        experience.reward_settlement="closed_market_last_real_mark"
                        experience.reward_end_timestamp=str(current)
                        experience.reward_quote_timestamp=str(panel.dates[last_quote])
                    key=origin_model+"_closed_market_rewards_settled"
                    self.metrics[key]=int(self.metrics.get(key,0))+1
                if goal_before and same_goal_episode:
                    currency=_currency(*panel.groups[symbol])
                    deltas={c:max(0.0,float(goal_points.get(c,0))-float(goal_before.get(c,0))) for c in goal_points}
                    contribution=deltas.get(currency,0.0)*min(1.0,max(0.0,float(dec.get("goal_weight_before",0.0))))
                    for experience in (exp,account_exp):
                        experience.goal_state=dec.get("goal_state")
                        experience.goal_reward_points=contribution if experience.trade_executed else 0.0
                        experience.goal_episode_id=dec.get("goal_episode_id")
                        experience.goal_terminal=goal_terminal
                    account_exp.portfolio_goal_reward_points=sum(deltas.values())
                if dec.get("credit_observations_target"):
                    for experience in (exp,account_exp):
                        experience.credit_observations=int(dec.get("credit_observations_elapsed",0))
                        if not terminal:
                            experience.bootstrap_discount=1.0
                            experience.bootstrap_symbol_index=symbol_ix
                    if not terminal:
                        # One shared successor input for all matured decisions at
                        # this market/account state. It is persisted in the same DB.
                        if credit_successor is None:
                            args=self._window(panel,end_index)
                            pstate,astate=account.model_inputs(panel,end_index)
                            credit_successor=Experience(args[0][0].numpy(),args[1][0].numpy(),
                                args[2][0].numpy(),args[3][0].numpy(),args[4][0].numpy(),0,1,0.0,
                                str(current),market_context=(args[5][0].numpy() if len(args)>5 else None),
                                portfolio_state=np.asarray(pstate,dtype=np.float16),
                                 account_state=np.asarray(astate,dtype=np.float16),
                                 goal_state=(np.asarray(account.goal_inputs(),dtype=np.float32) if account.goal_inputs() is not None else None),
                                multiscale_state=panel.multiscale_at(end_index).astype(np.float16),
                                daily_history=(panel.daily_history_at(end_index) if hasattr(panel,"daily_history_at") else None))
                        exp._bootstrap_experience=credit_successor
                        account_exp._bootstrap_experience=credit_successor
                matured_experiences.extend((exp,account_exp))
                matured_acks.append((pending_kind,
                    f"{dec.get('timestamp','')}|{dec.get('symbol',dec.get('symbol_index',''))}"))
                if first_account_transition:
                    account_transition_added.add(timestamp)
                    if not dec.get("credit_observations_target"):
                        self.metrics["paper_account_reward"] = float(
                            self.metrics.get("paper_account_reward",0.0))+account_reward
                    self.metrics["portfolio_experiences"] = int(
                        self.metrics.get("portfolio_experiences",0))+1
                self.metrics["paper_experiences_seen"]=int(
                    self.metrics.get("paper_experiences_seen",0))+1
                self.metrics["paper_experiences_since_candidate"]=int(
                    self.metrics.get("paper_experiences_since_candidate",0))+1
                self.replay.note_paper_outcome()
        if matured_experiences:
            # One atomic durable commit per observation. Neither an experience
            # nor its pending acknowledgement can be lost independently.
            self.replay.add_many(matured_experiences,pending_acks=matured_acks)
        self.metrics[origin_model+"_pending_reward_status"] = {
            "total":len(keep),"reasons":dict(waiting),
            "oldest":min((dec["timestamp"] for dec in keep),default=None)}
        return keep
