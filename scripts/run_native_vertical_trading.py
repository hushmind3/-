"""Immediate native vertical MoE paper run. No repeat downloads or 14-model probe.

Uses official ETH 36+9 observations, real daily stock excess returns and real
archival AAPL ITCH. Every evidence timestamp is recorded; archival ITCH is not
claimed to be a contemporaneous ETH order book.
"""
import argparse
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from stockrl.trading_moe import TradingMoE,parameter_digest
from stockrl.moe_paper import TradingMoEPaper
from stockrl.moe_training import update_controller
from stockrl.global_transformer import GlobalMarketPanel
from stockrl.expert_registry import atomic_json
from stockrl.expert_system import registry_owner


def publish_paper_status(root,state,bridge,decision,row,model,completed=False):
    registry=Path("runtime/trading_moe/registry.json")
    if not registry.is_file():return
    document=json.loads(registry.read_text(encoding="utf-8"))
    pipeline=document.setdefault("pipeline",{})
    previous=pipeline.get("paper_trading",{})
    paper={**previous,"timestamp":row["timestamp"],"books":row["books"],"fills":bridge.paper_account.state["fills"][-20:],
        "pending_orders":dict(bridge.paper_account.state["pending"]),"reward_points":bridge.paper_account.reward_points(),
        "optimizer_updates":model.optimizer_updates,"cycle_seconds":row["seconds"] if decision else previous.get("cycle_seconds"),"checkpoint":str(root/"TradingMoE.pt"),
        "replay":bridge.replay.stats(),"status":"paper_complete" if completed else "paper_running","live_executable":False}
    pipeline["paper_trading"]=paper
    if decision:
        paper["actions"]=decision["trading_output"]["actions"];paper["target_weights"]=decision["trading_output"]["target_weights"]
        paper["tradable_symbols"]=decision.get("tradable_symbols")
        destination=root/"inference/runs/native_vertical";destination.mkdir(parents=True,exist_ok=True)
        for packet in decision["raw_outputs"]:
            raw=json.dumps(packet).encode();target=destination/(packet["expert"]+".json");target.write_bytes(raw)
            entry=next(e for e in document["experts"] if e["id"]==packet["expert"])
            entry.update(raw_output_path=str(target),raw_output_sha256=hashlib.sha256(raw).hexdigest(),raw_output_origin="runtime_inference",
                last_used_at=pd.Timestamp.now(tz="UTC").isoformat(),last_output_shape=packet["output_shape"],last_input_shapes=packet["input_shapes"],
                router_selected=True,last_timings={"cold_load_seconds":packet.get("cold_load_seconds"),"gpu_transfer_seconds":packet.get("gpu_transfer_seconds"),
                    "forward_seconds":packet.get("forward_seconds"),"round_trip_seconds":packet.get("worker_seconds")})
        atomic_json(destination/"TradingMoE.json",decision)
        pipeline.update(stage="complete",result_path=str(destination/"TradingMoE.json"),selected_experts=decision["used_experts"],completed_experts=len(decision["used_experts"]),
            fusion_head_status="trainable_vertical_controller",as_of=decision["as_of"],timings={"total_seconds":decision["decision_seconds"]})
    atomic_json(registry,document)


def market_inputs(root,frame,index):
    stamp=pd.Timestamp(frame.iloc[index].timestamp)
    bars=frame.iloc[max(0,index-127):index+1].copy()
    price={"symbols":["ETHUSDT"],"as_of":str(stamp),"series":[bars.close.astype(float).tolist()],
        "observation_timestamps":[bars.timestamp.astype(str).tolist()],"horizon":1,"sampling_seconds":60,
        "frequency_id":0,"units":"price","input_authenticity":"real_official_ETHUSDT_price"}
    excess=pd.read_csv(root/"sources/TSFM_Finance/data/two_stocks_excess_returns.csv")
    excess=excess[pd.to_datetime(excess.TradingDate)<stamp.normalize()].tail(128)
    rates={"symbols":["AAPL"],"as_of":str(pd.Timestamp(excess.TradingDate.iloc[-1])),"horizon":1,
        "sampling_seconds":86400,"units":"daily_excess_return","series":[excess.AAPL.astype(float).tolist()],
        "observation_timestamps":[excess.TradingDate.tolist()],"input_authenticity":"real_official_daily_excess_return"}
    candles=bars[["timestamp","open","high","low","close","volume"]].copy()
    candles["timestamp"]=candles.timestamp.astype(str)
    candles["amount"]=candles.close*candles.volume
    candles_input={"symbols":["ETHUSDT"],"as_of":str(stamp),"horizon":1,"sampling_seconds":60,
        "bars":[candles.to_dict("records")],"future_timestamps":[str(stamp+pd.Timedelta(minutes=1))],
        "amount_observed":False,"input_authenticity":"real_ETHUSDT_OHLCV_amount_price_volume_proxy"}
    itch=json.loads((root/"native_data/MarketGPT/AAPL-20191230-native.json").read_text())
    return {"timesfm":rates,"chronos":rates,"toto":price,"kronos":candles_input,
        "fincast":price,"exaone":price,"timemoe":price,"marketgpt":itch}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--steps",type=int,default=6)
    p.add_argument("--state",type=Path,default=Path("runtime/trading_moe/native_vertical"))
    p.add_argument("--device",default="cuda:0");p.add_argument("--resume",action="store_true")
    args=p.parse_args();torch.set_num_threads(4)
    checkpoint=args.root/"TradingMoE.pt";model,saved=TradingMoE.load_checkpoint(checkpoint)
    groups=model.parameter_groups()
    optimizer=torch.optim.AdamW(groups,lr=1e-4)
    if args.resume and saved:
        try:optimizer.load_state_dict(saved)
        except ValueError:pass
    bridge=TradingMoEPaper(args.state,credit_seconds=60)
    episode=bridge.paper_account.state["episode_id"]
    if model.config.get("replay_account_episode")!=episode:
        model.config["applied_replay_rows"]={}
        model.config["replay_account_episode"]=episode
    initial=deepcopy(bridge.paper_account.snapshot())
    native=pd.read_feather(args.root/"native_data/MacroHFT/df_val.feather").iloc[:256].copy()
    native.timestamp=pd.to_datetime(native.timestamp)
    market=native[["timestamp","open","high","low","close","volume"]].rename(columns={"timestamp":"date"})
    market["symbol"]="ETHUSDT";market["market"]="US";market["asset_class"]="crypto"
    stock=pd.read_csv(Path(__file__).resolve().parents[1]/"data/global_market_daily.csv")
    stock=stock[(stock.symbol=="AAPL")&(pd.to_datetime(stock.date)<native.timestamp.iloc[128])].tail(64)
    market=pd.concat([stock,market],ignore_index=True)
    panel=GlobalMarketPanel("native_ETHUSDT",raw_frame=market)
    panel.groups["ETHUSDT"]=("BINANCE_USDT","crypto")
    # Reuse the integer-unit paper engine with explicit crypto contract lots.
    # One simulated unit = 0.001 ETH; quote per unit = real ETH price / 1000.
    # Not a modified native model input: experts keep original ETH prices.
    eth_index=panel.symbols.index("ETHUSDT")
    panel.closes[:,eth_index]*=.001
    before_hash=parameter_digest(model.controller)
    initial_updates=model.optimizer_updates
    contexts={};rows=[];update_logs=[];last=bridge.paper_account.state["last_timestamp"]
    # Expensive market experts run once for this short window. Native policy Q
    # and account-aware controller run every minute using the frozen market state.
    first=128 if not args.resume or not last else max(128,int((pd.Timestamp(last)-native.timestamp.iloc[0]).total_seconds()/60)+1)
    inputs=market_inputs(args.root,native,first)
    market_packets=None
    for step in range(args.steps+2):
        index=first+step;stamp=str(native.iloc[index].timestamp)
        pi=int(np.flatnonzero(panel.dates==np.datetime64(stamp))[0])
        if last and panel.dates[pi]<=np.datetime64(last):continue
        started=time.perf_counter();fills=bridge.advance(panel,pi)
        decision=None;orders=None
        if step<args.steps:
            pstate,astate=bridge.paper_account.model_inputs(panel,pi)
            account=torch.tensor(np.column_stack([pstate,np.broadcast_to(astate,(len(pstate),len(astate)))]),dtype=torch.float32)[None]
            held=bool(bridge.paper_account.state["books"]["USD"]["positions"].get("ETHUSDT"))
            policy_data=model.macro_input_adapter(native,index,int(held))
            snapshot={"as_of":stamp,"symbols":panel.symbols,"currencies":{s:"USD" for s in panel.symbols},
                "tradable_symbols":[s for j,s in enumerate(panel.symbols) if panel.observed[pi,j]],
                "current_weights":{s:float(pstate[j][1]) for j,s in enumerate(panel.symbols)},"expert_inputs":{}}
            for key in model.controller.policy_ids:snapshot["expert_inputs"][key]={**policy_data,"variant":model.experts[key].entry["variant"]}
            if market_packets is None:
                snapshot["expert_inputs"].update(inputs)
                with torch.no_grad():decision,_=model(snapshot,account,device=args.device,explore=True)
                market_packets=[p for p in decision["raw_outputs"] if p["expert"] in model.controller.market_ids]
            else:
                packets=list(market_packets)
                for key in model.controller.policy_ids:
                    data=snapshot["expert_inputs"][key]
                    with registry_owner(model.gpu_lock),model.scheduler.work("champion_live"):
                        packet=model.experts[key](model.root,data,args.device)
                    packet["expert"]=key;packet["native_features_verified"]=True
                    packets.append(packet)
                    if args.device.startswith("cuda"):torch.cuda.empty_cache()
                with torch.no_grad():decision,_=model(snapshot,account,packets=packets,explore=True)
            orders=bridge.submit(decision,panel,pi,paper_executable=True)
            contexts[str(panel.dates[pi])] = (snapshot,decision)
        batch=bridge.replay.pending_batch(256,exclude_row_ids={int(k) for k in model.config.get("applied_replay_rows",{})});ack={}
        for exp in batch:
            if exp.source!="paper_account_portfolio" or not exp.portfolio_value_transition:continue
            context=contexts.get(exp.timestamp)
            if not context:continue
            snapshot,original=context
            change=update_controller(model,optimizer,exp,snapshot,original["raw_outputs"],original["trading_output"]["target_weights"],original["trading_output"]["cash_weights_by_currency"]["USD"],original["trading_output"]["actions"])
            update_logs.append({"timestamp":exp.timestamp,**change})
            ack.update({e._replay_row_id:1 for e in batch if e.timestamp==exp.timestamp})
        # Save once after this short continuous run, then acknowledge the rows.
        row={"timestamp":stamp,"decision":decision,"orders":orders,"fills":fills,
            "books":deepcopy(bridge.paper_account.snapshot()["books"]),"reward_points":bridge.paper_account.reward_points(),
            "seconds":time.perf_counter()-started,"replay_rows":bridge.replay.stats()["total"]}
        rows.append(row)
        publish_paper_status(args.root,args.state,bridge,decision,row,model)
        with (args.state/"cycles.jsonl").open("a",encoding="utf-8") as handle:handle.write(json.dumps(row)+"\n")
        # Remember updated IDs so the next step does not consume them again.
        for e in batch:
            if e._replay_row_id in ack:e._updated=True
        if ack:
            model.config.setdefault("applied_replay_rows",{}).update({str(k):1 for k in ack})
        last=str(panel.dates[pi])
        print(json.dumps({"step":step,"actions":decision["trading_output"]["actions"] if decision else None,
            "fills":fills,"NAV":row["books"]["USD"]["equity"],"updates":model.optimizer_updates,"seconds":row["seconds"]}),flush=True)
    model.save_checkpoint(checkpoint,optimizer)
    bridge.replay.acknowledge_training({int(k):v for k,v in model.config.get("applied_replay_rows",{}).items()})
    if rows:publish_paper_status(args.root,args.state,bridge,None,rows[-1],model,completed=True)
    report={"initial":initial["books"],"final":bridge.paper_account.snapshot()["books"],"fills":bridge.paper_account.state["fills"],
        "contract_units":{"ETHUSDT":{"quantity_per_unit":.001,"quote_currency":"USDT","USD_parity_assumption":True}},
        "steps":[{"timestamp":r["timestamp"],"seconds":r["seconds"],"actions":r["decision"]["trading_output"]["actions"] if r["decision"] else None,
                  "target_weights":r["decision"]["trading_output"]["target_weights"] if r["decision"] else None} for r in rows],
        "used_experts":rows[0]["decision"]["used_experts"] if rows and rows[0]["decision"] else [],
        "native_Q":[{p["expert"]:p["native_output"] for p in r["decision"]["raw_outputs"] if p["expert"].startswith("macrophft_")} for r in rows if r["decision"]],
        "updates":update_logs,"updated_parameter_groups":[g["name"] for g in groups],"expert_parameters_frozen":True,
        "optimizer_updates":model.optimizer_updates,"initial_optimizer_updates":initial_updates,"reward_points":bridge.paper_account.reward_points(),"replay":bridge.replay.stats(),
        "controller_before":before_hash,"controller_after":parameter_digest(model.controller),
        "parameters":sum(p.numel() for p in model.parameters()),"checkpoint_bytes":checkpoint.stat().st_size,
        "evidence_as_of":rows[0]["decision"]["evidence_as_of"] if rows and rows[0]["decision"] else {},
        "MarketGPT_context":"Real AAPL 2019 ITCH archival reference, not a contemporaneous 2023 ETH order book",
        "live_executable":False}
    atomic_json(args.state/"report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("initial","final","native_Q","fills","steps","updates")}),flush=True)


if __name__=="__main__":main()
