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
import os
from datetime import datetime,timezone
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
from stockrl.paths import TRADING_MOE_CHECKPOINT

worker_status_path=None


def release_offloaded_pages():
    """Let Windows reclaim inactive mmap checkpoint pages, without unloading modules."""
    if os.name=="nt":
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        trim=ctypes.WinDLL("psapi",use_last_error=True).EmptyWorkingSet
        trim.argtypes=[wintypes.HANDLE];trim.restype=wintypes.BOOL
        trim(kernel.GetCurrentProcess())


def publish_worker(state,model=None,bridge=None,row=None,decision=None,learning=None,**changes):
    """Small polling snapshot; native evidence stays in the diagnostic registry."""
    path=state/"worker_status.json"
    try:data=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError):data={}
    data.update(changes,pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat())
    if model is not None:
        data.update(optimizer_updates=model.optimizer_updates,
            applied_replay_rows=len(model.config.get("applied_replay_rows",{})))
        learner_device=next(model.controller.parameters()).device
        data["compute"]={"inference_device":"cuda:0" if torch.cuda.is_available() else "cpu",
            "learning_device":str(learner_device),"gpu_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "allocated_bytes":torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0,
            "reserved_bytes":torch.cuda.memory_reserved(0) if torch.cuda.is_available() else 0,
            "expert_gpu_limit":1,"expert_storage":"memory-mapped PT / sequential CUDA"}
    if bridge is not None:
        data.update(books=bridge.paper_account.snapshot()["books"],fills=bridge.paper_account.state["fills"][-20:],
            pending_orders=bridge.paper_account.state["pending"],reward_points=bridge.paper_account.reward_points(),replay=bridge.replay.stats())
        if model is not None:
            applied=bridge.replay.retained_row_count(model.config.get("applied_replay_rows",{}))
            data["replay"].update(awaiting_checkpoint=applied,
                remaining_for_update=max(0,data["replay"]["untrained"]-applied))
    if row:data.update(market_timestamp=row["timestamp"],cycle_seconds=row["seconds"])
    if decision:
        output=decision["trading_output"];symbols=list(output["actions"])
        eth=symbols.index("ETHUSDT")
        cash=output["cash_weights_by_currency"]["USD"]+sum(weight for symbol,weight in output["target_weights"].items() if symbol not in decision["tradable_symbols"])
        data["decision"]={"action":output["actions"]["ETHUSDT"],"target_weight":output["target_weights"]["ETHUSDT"],
            "current_weight":decision["current_weights"].get("ETHUSDT",0),"cash_weight":cash,"as_of":decision["as_of"],
            "value":decision["fusion_output"]["native_head_output"]["value"][0][eth],"seconds":decision.get("native_decision_seconds",decision["decision_seconds"])}
        data["stages"]={"market":len([x for x in decision["used_experts"] if not x.startswith("macrophft_")])==8,
            "state":True,"policy":len([x for x in decision["used_experts"] if x.startswith("macrophft_")])==6,"controller":True,"action":True}
    if learning:data["learning"]={"loss":learning["loss"],"reward_points":learning["reward_points"],"updated_at":data["updated_at"]}
    atomic_json(path,data)


def save_contexts(state,contexts):
    # Retain the evidence used by each pending action, including an action
    # immediately before market-cache refresh. Never relabel it as newer data.
    pending={}
    for stamp,(snapshot,decision) in contexts.items():
        pending[stamp]={"snapshot":snapshot,"trading_output":decision["trading_output"],"raw_outputs":decision["raw_outputs"]}
    atomic_json(state/"pending_contexts.json",pending)


def publish_paper_status(root,state,bridge,decision,row,model,completed=False):
    registry=Path("runtime/trading_moe/registry.json")
    if not registry.is_file():return
    document=json.loads(registry.read_text(encoding="utf-8"))
    pipeline=document.setdefault("pipeline",{})
    previous=pipeline.get("paper_trading",{})
    paper={**previous,"timestamp":row["timestamp"],"books":row["books"],"fills":bridge.paper_account.state["fills"][-20:],
        "pending_orders":dict(bridge.paper_account.state["pending"]),"reward_points":bridge.paper_account.reward_points(),
        "optimizer_updates":model.optimizer_updates,"cycle_seconds":row["seconds"] if decision else previous.get("cycle_seconds"),"checkpoint":str(getattr(model,"runtime_checkpoint",TRADING_MOE_CHECKPOINT)),
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


def runtime_modes(state):
    """Named models share the operator's flags; standalone MoE keeps its own lifecycle."""
    if state.name not in ("champion_moe", "candidate_moe"):
        return dict(observe_enabled=True, paper_enabled=True, learning_enabled=True)
    try:
        flags=json.loads((state.parent/"autonomy.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(observe_enabled=False, paper_enabled=False, learning_enabled=False)
    return {key:bool(flags.get(key,False)) for key in ("observe_enabled","paper_enabled","learning_enabled")}


def learn_saved_contexts(model,optimizer,bridge,contexts):
    batch=bridge.replay.pending_batch(256,
        exclude_row_ids={int(k) for k in model.config.get("applied_replay_rows",{})},timestamps=contexts)
    ack={};latest=None;logs=[]
    for exp in batch:
        if exp.source!="paper_account_portfolio" or not exp.portfolio_value_transition:continue
        context=contexts.get(exp.timestamp)
        if not context:continue
        snapshot,original=context
        latest=update_controller(model,optimizer,exp,snapshot,original["raw_outputs"],original["trading_output"]["target_weights"],original["trading_output"]["cash_weights_by_currency"]["USD"],original["trading_output"]["actions"])
        logs.append({"timestamp":exp.timestamp,**latest})
        ack.update({e._replay_row_id:1 for e in batch if e.timestamp==exp.timestamp})
    for exp in batch:
        if exp._replay_row_id in ack:exp._updated=True
    if ack:
        model.config.setdefault("applied_replay_rows",{}).update({str(k):1 for k in ack})
        for stamp in {exp.timestamp for exp in batch if exp._replay_row_id in ack}:contexts.pop(stamp,None)
    return latest,logs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--steps",type=int,default=6)
    p.add_argument("--state",type=Path,default=Path("runtime/trading_moe/native_vertical"))
    p.add_argument("--checkpoint",type=Path,help="use this named model file without renaming or copying it")
    p.add_argument("--recipe",type=Path,help="shared expert base with a separate small learned state")
    p.add_argument("--device",default="cuda:0");p.add_argument("--resume",action="store_true")
    p.add_argument("--continuous",action="store_true",help="keep one model resident and run until stop.request")
    p.add_argument("--interval",type=float,default=2,help="wall seconds between historical decisions")
    args=p.parse_args()
    args.state.mkdir(parents=True,exist_ok=True)
    global worker_status_path
    worker_status_path=args.state/"worker_status.json"
    with registry_owner(args.state/"worker-owner.lock"):
        run(args)


def run(args):
    torch.set_num_threads(4)
    checkpoint=args.checkpoint or TRADING_MOE_CHECKPOINT
    publish_worker(args.state,status="loading",load_count=0,error=None,stop_requested=False,gpu_waiting=False,gpu_wait_seconds=0,message=f"{checkpoint.name}를 한 번 적재하는 중입니다.")
    load_started=time.perf_counter()
    model,saved=TradingMoE.load_checkpoint(checkpoint)
    assembly_recipe=None
    if args.recipe:
        assembly_recipe=json.loads(args.recipe.read_text(encoding="utf-8"))
        model.apply_assembly_recipe(assembly_recipe)
        if assembly_recipe.get("trainable_state"):
            saved=model.load_assembly_state(assembly_recipe["trainable_state"])
    model.runtime_checkpoint=checkpoint
    if args.device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA is required for the requested GPU worker")
    model.set_learning_device(args.device)
    release_offloaded_pages()
    publish_worker(args.state,model,status="loading",load_count=1,load_seconds=time.perf_counter()-load_started,
        parameters=sum(p.numel() for p in model.parameters()),checkpoint_bytes=checkpoint.stat().st_size,message="계좌와 optimizer 상태를 복원하는 중입니다.")
    groups=model.parameter_groups()
    optimizer=torch.optim.AdamW(groups,lr=1e-4)
    if args.resume and saved:
        try:optimizer.load_state_dict(saved)
        except ValueError:pass
    bridge=TradingMoEPaper(args.state,credit_seconds=60)
    def gpu_wait(waiting,seconds):
        publish_worker(args.state,model,bridge,gpu_waiting=waiting,gpu_wait_seconds=seconds,
            message="다른 모델의 GPU 작업을 기다립니다 · 종료하지 않고 차례대로 실행" if waiting else "GPU 차례 확보 · 가상매매를 이어 실행합니다.")
    model.gpu_wait_callback=gpu_wait
    episode=bridge.paper_account.state["episode_id"]
    if model.config.get("replay_account_episode")!=episode:
        model.config["applied_replay_rows"]={}
        model.config["replay_account_episode"]=episode
    initial=deepcopy(bridge.paper_account.snapshot())
    native=pd.read_feather(args.root/"native_data/MacroHFT/df_val.feather")
    native.timestamp=pd.to_datetime(native.timestamp)
    last=bridge.paper_account.state["last_timestamp"]
    first=128 if not args.resume or not last else max(128,int((pd.Timestamp(last)-native.timestamp.iloc[0]).total_seconds()/60)+1)
    native=native.iloc[:min(len(native),first+(4096 if args.continuous else args.steps+2))].copy()
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
    contexts={};rows=[];update_logs=[]
    # Expensive market experts run once for this short window. Native policy Q
    # and account-aware controller run every minute using the frozen market state.
    if first>=len(native):
        publish_worker(args.state,model,bridge,status="stopped",message="확보된 과거 시세 구간을 모두 처리했습니다.")
        return
    if args.resume and (args.state/"pending_contexts.json").exists():
        for stamp,context in json.loads((args.state/"pending_contexts.json").read_text()).items():
            contexts[stamp]=(context["snapshot"],{"trading_output":context["trading_output"],"raw_outputs":context["raw_outputs"]})
    inputs=market_inputs(args.root,native,first)
    market_packets=None
    step_count=len(native)-first if args.continuous else args.steps+2
    publish_worker(args.state,model,bridge,status="running",message="공식 ETHUSDT 과거 구간을 이어 실행합니다.")
    for step in range(step_count):
        modes=runtime_modes(args.state)
        while args.continuous and not modes["observe_enabled"] and not (args.state/"stop.request").exists():
            latest=None
            if modes["learning_enabled"]:
                latest,logs=learn_saved_contexts(model,optimizer,bridge,contexts)
                update_logs.extend(logs);update_logs=update_logs[-1:]
                save_contexts(args.state,contexts)
            publish_worker(args.state,model,bridge,learning=latest,status="running",modes=modes,
                learning_active=latest is not None,message="새 판단 중지 · 저장 경험 학습 허용" if modes["learning_enabled"] else "새 판단·학습 중지 · 모델 메모리 유지")
            time.sleep(.25)
            modes=runtime_modes(args.state)
        if args.continuous and (args.state/"stop.request").exists():break
        index=first+step;stamp=str(native.iloc[index].timestamp)
        pi=int(np.flatnonzero(panel.dates==np.datetime64(stamp))[0])
        if last and panel.dates[pi]<=np.datetime64(last):continue
        started=time.perf_counter();fills=bridge.advance(panel,pi,enabled=modes["paper_enabled"])
        decision=None;orders=None
        if args.continuous or step<args.steps:
            pstate,astate=bridge.paper_account.model_inputs(panel,pi)
            account=torch.tensor(np.column_stack([pstate,np.broadcast_to(astate,(len(pstate),len(astate)))]),dtype=torch.float32)[None]
            held=bool(bridge.paper_account.state["books"]["USD"]["positions"].get("ETHUSDT"))
            policy_data=model.macro_input_adapter(native,index,int(held))
            snapshot={"as_of":stamp,"symbols":panel.symbols,"currencies":{s:"USD" for s in panel.symbols},
                "tradable_symbols":[s for j,s in enumerate(panel.symbols) if panel.observed[pi,j]],
                "current_weights":{s:float(pstate[j][1]) for j,s in enumerate(panel.symbols)},"expert_inputs":{}}
            for key in model.controller.macro_policy_ids:snapshot["expert_inputs"][key]={**policy_data,"variant":model.experts[key].entry["variant"]}
            refresh_steps=max(1,min(assembly_recipe["refresh_seconds"][k] for k in model.controller.market_ids if k in assembly_recipe["enabled_experts"])//60) if assembly_recipe else 120
            if market_packets is None or (args.continuous and step%refresh_steps==0):
                inputs=market_inputs(args.root,native,index)
                snapshot["expert_inputs"].update(inputs)
                with torch.no_grad():decision,_=model(snapshot,account,device=args.device,explore=True)
                market_packets=[p for p in decision["raw_outputs"] if p["expert"] in model.controller.market_ids]
                atomic_json(args.state/"market_evidence.json",market_packets)
                release_offloaded_pages()
            else:
                packets=list(market_packets)
                for key in model.controller.macro_policy_ids:
                    if assembly_recipe and key not in assembly_recipe["enabled_experts"]:continue
                    data=snapshot["expert_inputs"][key]
                    with registry_owner(model.gpu_lock,wait=True,on_wait=gpu_wait),model.scheduler.work("champion_live"):
                        packet=model.experts[key](model.root,data,args.device)
                    packet["expert"]=key;packet["native_features_verified"]=True
                    packets.append(packet)
                    if args.device.startswith("cuda"):torch.cuda.empty_cache()
                with torch.no_grad():decision,_=model(snapshot,account,packets=packets,explore=True)
            decision["native_decision_seconds"]=time.perf_counter()-started
            decision["current_weights"]=snapshot["current_weights"]
            orders=bridge.submit(decision,panel,pi,paper_executable=modes["paper_enabled"])
            contexts[str(panel.dates[pi])] = (snapshot,decision)
        latest_learning=None
        if modes["learning_enabled"]:
            latest_learning,logs=learn_saved_contexts(model,optimizer,bridge,contexts)
            update_logs.extend(logs)
        # Save once after this short continuous run, then acknowledge the rows.
        row={"timestamp":stamp,"decision":decision,"orders":orders,"fills":fills,
            "books":deepcopy(bridge.paper_account.snapshot()["books"]),"reward_points":bridge.paper_account.reward_points(),
            "seconds":time.perf_counter()-started,"replay_rows":bridge.replay.stats()["total"]}
        rows.append(row)
        if args.continuous:rows=rows[-2:]
        publish_paper_status(args.root,args.state,bridge,decision,row,model)
        with (args.state/"cycles.jsonl").open("a",encoding="utf-8") as handle:handle.write(json.dumps(row)+"\n")
        save_contexts(args.state,contexts)
        publish_worker(args.state,model,bridge,row,decision,latest_learning,status="running",modes=modes,learning_active=latest_learning is not None)
        last=str(panel.dates[pi])
        print(json.dumps({"step":step,"actions":decision["trading_output"]["actions"] if decision else None,
            "fills":fills,"NAV":row["books"]["USD"]["equity"],"updates":model.optimizer_updates,"seconds":row["seconds"]}),flush=True)
        if args.continuous:
            update_logs=update_logs[-1:]
            deadline=time.monotonic()+max(0,args.interval)
            while time.monotonic()<deadline and not (args.state/"stop.request").exists():time.sleep(min(.1,max(0,deadline-time.monotonic())))
    publish_worker(args.state,model,bridge,status="saving",stop_requested=True,message="계좌·replay·optimizer·TradingMoE.pt를 저장하는 중입니다.")
    bridge.paper_account.save();save_contexts(args.state,contexts)
    if assembly_recipe:
        small_path=args.recipe.parent/"trainable"/(assembly_recipe["candidate_id"]+".pt")
        model.save_assembly_state(small_path,optimizer)
        assembly_recipe["trainable_state"]=str(small_path)
        atomic_json(args.recipe,assembly_recipe)
    else:model.save_checkpoint(checkpoint,optimizer)
    bridge.replay.acknowledge_training({int(k):v for k,v in model.config.get("applied_replay_rows",{}).items()})
    if rows:publish_paper_status(args.root,args.state,bridge,None,rows[-1],model,completed=True)
    publish_worker(args.state,model,bridge,status="stopped",stop_requested=False,message="저장 완료 · 다음 시작은 같은 계좌와 학습 상태에서 이어집니다.")
    if args.continuous:return
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


if __name__=="__main__":
    try:main()
    except Exception as exc:
        if worker_status_path:publish_worker(worker_status_path.parent,status="error",error=f"{type(exc).__name__}: {exc}")
        raise
