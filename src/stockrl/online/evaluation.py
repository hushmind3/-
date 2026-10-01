"""Offline account evaluation, backtests and inference benchmarks."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import torch
from ..global_transformer import ACTION_NAMES, GlobalMarketPanel
from ..paper_account import PaperAccount, _currency
from .data import Experience

def benchmark_model(model, panel, window=128, repeats=3, device=None):
    from ..core import device_for
    from ..global_online import OnlineGlobalAgent
    dev=device or device_for(); model.to(dev)
    contextual=getattr(model,"_stockrl_uses_market_context",False)
    if dev.type=="cuda":
        # The large backbone uses FP16 while context adapters consume FP32.
        (model.backbone if contextual else model).half()
    model.eval()
    end_index=min(len(panel.dates)-1,window)
    args=[x.to(dev) for x in panel.window(end_index,window,include_context=contextual)]
    args[0]=args[0].to(dtype=next(model.parameters()).dtype)
    extra=({"multiscale_state":torch.as_tensor(panel.multiscale_at(end_index)[None],device=dev)}
           if contextual else {})
    if contextual:
        extra.update(OnlineGlobalAgent._daily_history_kwargs(panel,end_index,dev))
    samples=[]
    with torch.inference_mode():
        model(*args,**extra)
        if dev.type=="cuda": torch.cuda.synchronize(dev)
        for _ in range(repeats):
            if dev.type=="cuda": torch.cuda.synchronize(dev)
            t=time.perf_counter(); model(*args,**extra); samples.append(time.perf_counter()-t)
            if dev.type=="cuda":
                torch.cuda.synchronize(dev); samples[-1]=time.perf_counter()-t
    return {"inference_seconds_p50":float(np.percentile(samples,50)),"inference_seconds_p95":float(np.percentile(samples,95)),
            "inference_repeats":repeats}


class _EvaluationMixin:
    def evaluate_panel(self,panel:GlobalMarketPanel,model=None,start_index:int|None=None,
                       end_index:int|None=None,stride:int=1,report_rows=None)->dict:
        """Evaluate real cash-only portfolio actions using the live paper ledger."""
        model=model or self.champion
        start=max(self.window-1,0 if start_index is None else start_index)
        end=len(panel.dates) if end_index is None else min(len(panel.dates),end_index)
        if stride<1:
            raise ValueError("stride must be positive")
        account=PaperAccount.in_memory(self.fee,self.slippage)
        rules=getattr(self,"operating_rules",{})
        account.configure_goal(rules.get("goal_target_multiple",10.0),rules.get("goal_win_bonus_points",100.0))
        initial=account.normalized_equity()
        peak=initial; max_dd=0.0; daily={}; action_counts={name:0 for name in ACTION_NAMES}
        decisions=0; observed_bars=0; previous_equity=initial
        was_training=model.training; model.eval()
        model_device=next(model.parameters()).device
        try:
            with torch.inference_mode():
                for ti in range(start,end):
                    fills=account.process_bar(panel,ti,True)
                    account.observe_goal(str(panel.dates[ti]))
                    if (ti-start)%stride==0:
                        contextual=getattr(model,"_stockrl_uses_market_context",False)
                        args=[x.to(model_device) for x in panel.window(ti,self.window,include_context=contextual)]
                        args[0]=args[0].to(dtype=next(model.parameters()).dtype)
                        pstate,astate=account.model_inputs(panel,ti)
                        if contextual:
                            logits,values,allocation=model(*args,
                                portfolio_state=torch.as_tensor(np.asarray(pstate)[None],device=model_device,dtype=torch.float32),
                                account_state=torch.as_tensor(np.asarray(astate)[None],device=model_device,dtype=torch.float32),
                                multiscale_state=torch.as_tensor(panel.multiscale_at(ti)[None],device=model_device),
                                **self._daily_history_kwargs(panel,ti,model_device),
                                **self._goal_kwargs(account,model_device),
                                return_allocation=True)
                            allocation=allocation[0].float().cpu().numpy()
                        else:
                            logits,values=model(*args); allocation=None
                        probabilities=self._account_action_probabilities(logits[0].float().cpu().numpy(),pstate)
                        actions=self._deterministic_actions(probabilities)
                        account.queue_decisions(panel,ti,probabilities,True,allocation=allocation,actions=actions)
                        for j,symbol in enumerate(panel.symbols):
                            if panel.observed[ti,j] and _currency(*panel.groups[symbol]) is not None:
                                action_counts[ACTION_NAMES[actions[j]]]+=1
                                decisions+=1
                    equity=account.normalized_equity()
                    peak=max(peak,equity)
                    max_dd=max(max_dd,(peak-equity)/max(peak,1e-9))
                    day=str(panel.dates[ti])[:10]
                    daily[day]=daily.get(day,0.0)+(equity-previous_equity)/initial
                    previous_equity=equity; observed_bars+=1
                    if report_rows is not None:
                        report_rows.append({"date":str(panel.dates[ti]),
                            "net_asset_return":equity/initial-1.0,"fills":len(fills),
                            "KRW_equity":account._equity("KRW"),"USD_equity":account._equity("USD")})
        finally:
            model.train(was_training)
        books=account.snapshot()["books"]
        costs=sum(sum(float(book[k]) for k in ("fees","sell_tax","spread","slippage"))
                  /float(book["initial_cash"]) for book in books.values())/initial
        net_return=account.normalized_equity()/initial-1.0
        return {"start_index":start,"end_index":end,"timestamps":observed_bars,"decisions":decisions,
            "evaluation_mode":"cash_only_sequential_paper_account",
            "score_definition":"mean of KRW and USD seed-normalized net asset returns; no raw currency sum",
            "net_return":net_return,"net_pnl_return_sum":net_return,"compounded_step_return":net_return,
            "gross_pnl_return_sum":net_return+costs,"cost_return_sum":costs,
            "max_drawdown":max_dd,"action_counts":action_counts,"daily_net_return":daily,
            "fee_rate":self.fee,"slippage_bps":self.slippage*10000,"books":books,
            "trade_count":sum(book["trade_count"] for book in books.values()),
            "unfilled_orders_at_end":len(account.state["pending"]),
            "fills_at":"next available observation for that symbol",
            "data_usage":"historical diagnostic; held-out status requires explicit training lineage"}

    def backtest_panel(self,panel:GlobalMarketPanel,stride:int=1)->dict:
        """Run the final 15% with the same ledger used in live and promotion trials."""
        import pandas as pd
        started=time.perf_counter()
        start=max(1,int(len(panel.dates)*.85))
        rows=[]
        out=self.evaluate_panel(panel,self.champion,start_index=start,stride=stride,report_rows=rows)
        pd.DataFrame(rows,columns=["date","net_asset_return","fills","KRW_equity","USD_equity"]).to_csv(
            self.state_dir/"backtest.csv",index=False)
        out["elapsed_seconds"]=float(time.perf_counter()-started)
        out["checkpoint_sha256"]=self._sha256_file(self.champion_path)
        out["completed_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
        self._atomic_json(out,self.state_dir/"backtest.json")
        self.metrics["backtest"]=out
        return out

    def import_teacher_csv(self, panel:GlobalMarketPanel, path:str|Path, symbol="MSFT", max_index:int|None=None,
                           source_name:str|None=None):
        """Import normalized SELL/HOLD/BUY labels as replayed imitation data."""
        import pandas as pd
        df=pd.read_csv(path)
        if not {"date","action"}.issubset(df.columns): raise ValueError("teacher CSV needs date,action columns")
        source_name=source_name or f"teacher:{Path(path).stem}"
        symbol_to_ix={s:i for i,s in enumerate(panel.symbols)}
        if symbol not in symbol_to_ix: raise ValueError(f"teacher symbol {symbol} is absent from panel")
        date_to_ix={pd.Timestamp(d):i for i,d in enumerate(panel.dates)}
        with self.replay.lock:
            existing={(e.source,e.timestamp,e.symbol_index,e.action) for e in self.replay.items}
        actions={"SELL":0,"HOLD":1,"WAIT":1,"BUY":2}
        added=0
        for row in df.itertuples(index=False):
            date=pd.Timestamp(row.date)
            if date.tzinfo is not None: date=date.tz_convert("UTC").tz_localize(None)
            ti=date_to_ix.get(date)
            if ti is None: ti=date_to_ix.get(date.normalize())
            row_symbol=str(getattr(row,"symbol",symbol)); j=symbol_to_ix.get(row_symbol)
            action=actions.get(str(row.action).upper())
            if ti is None or action is None or j is None or (max_index is not None and ti>=max_index): continue
            key=(source_name,str(date.normalize()),j,action)
            if key in existing: continue
            window=self._window(panel,ti); x,sid,mid,aid,mask=window[:5]
            self.replay.add(Experience(x[0].numpy().astype(np.float16),sid[0].numpy(),mid[0].numpy(),aid[0].numpy(),
                mask[0].numpy(),j,action,0.0,str(date),source_name,0.0,
                market_context=window[5][0].numpy().astype(np.float16) if len(window)>5 else None)); added+=1; existing.add(key)
        return added
