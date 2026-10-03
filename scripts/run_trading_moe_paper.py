"""Execute saved TradingMoE decisions against real quotes using existing paper code.

--continuous watches new completed quotes and new inference outputs. It never
replays a stale decision as a new inference or calls any live broker.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from copy import deepcopy
import pandas as pd
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from stockrl.moe_paper import TradingMoEPaper
from stockrl.market_panel import GlobalMarketPanel
from stockrl.expert_registry import atomic_json, read_fusion_output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result",type=Path)
    parser.add_argument("--registry",type=Path,default=Path("runtime/trading_moe/registry.json"))
    parser.add_argument("--market",type=Path,default=Path("data/global_market_daily.csv"))
    parser.add_argument("--state",type=Path,default=Path("runtime/trading_moe/paper"))
    parser.add_argument("--continuous",action="store_true")
    parser.add_argument("--bars",type=int,default=3,help="historical quotes after the decision, including fill/reward quotes")
    args=parser.parse_args()
    bridge=TradingMoEPaper(args.state)
    initial=deepcopy(bridge.paper_account.snapshot())
    consumed=set()
    rows=[]
    while True:
        started=time.perf_counter()
        result=json.loads(args.result.read_text(encoding="utf-8")) if args.result else read_fusion_output(args.registry)
        symbols=list(result["trading_output"]["target_weights"])
        frame=pd.read_csv(args.market)
        frame=frame[frame.symbol.isin(symbols)].copy()
        panel=GlobalMarketPanel(args.market,raw_frame=frame)
        stamp=np.datetime64(result["as_of"])
        positions=np.flatnonzero(panel.dates==stamp)
        if not len(positions): raise ValueError("no actual quotes for inference as-of")
        start=int(positions[0])
        last=bridge.paper_account.state["last_timestamp"]
        end=len(panel.dates) if args.continuous else min(len(panel.dates),start+args.bars+1)
        for index in range(start,end):
            current=str(panel.dates[index])
            if last and current<=last: continue
            fills=bridge.advance(panel,index)
            submitted=None
            run=result.get("run_id",result["as_of"])
            if index==start and run not in consumed:
                submitted=bridge.submit(result,panel,index,paper_executable=True)
                consumed.add(run)
            account=deepcopy(bridge.paper_account.snapshot())
            row={"timestamp":current,"orders":submitted,"fills":fills,
                "books":account["books"],"reward_points":bridge.paper_account.reward_points(),
                "reward_accumulated":100*bridge.metrics.get("paper_account_reward",0),
                "replay":bridge.replay.stats(),"pending_outcomes":len(bridge.pending)}
            rows.append(row)
            with (args.state/"paper_cycles.jsonl").open("a",encoding="utf-8") as handle:
                handle.write(json.dumps(row,ensure_ascii=False)+"\n")
            last=current
        report={"initial":initial["books"],"final":bridge.paper_account.snapshot()["books"],
            "fills":bridge.paper_account.state["fills"],"fill_count":sum(b["trade_count"] for b in bridge.paper_account.state["books"].values()),
            "reward_points":bridge.paper_account.reward_points(),"reward_accumulated":100*bridge.metrics.get("paper_account_reward",0),
            "replay":bridge.replay.stats(),"pending_outcomes":len(bridge.pending),
            "connection_seconds":time.perf_counter()-started,
            "saved_inference_seconds":result.get("pipeline_timings",{}).get("total_seconds"),
            "expert_workers_started":0,"live_executable":False,"training_performed":False}
        atomic_json(args.state/"paper_report.json",report)
        if not args.continuous:
            print(json.dumps(report,ensure_ascii=False));break
        if (args.state/"stop").exists(): break
        time.sleep(1)


if __name__=="__main__":main()
