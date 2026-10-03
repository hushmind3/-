"""One existing Candidate worker, one shared frozen base, paired paper trials.

Actual stored native market evidence is replayed without relabeling its as-of.
MacroHFT Q inputs are recomputed from official ETH features and EACH book's action.
Only learned head/adapters are saved; no full checkpoint save or expert training.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
import torch
from stockrl.assembly_orchestrator import read
from stockrl.expert_registry import atomic_json
from stockrl.expert_system import registry_owner
from stockrl.global_transformer import GlobalMarketPanel
from stockrl.moe_paper import TradingMoEPaper
from stockrl.trading_moe import TradingMoE
from stockrl.paths import PROJECT_ROOT
from run_native_vertical_trading import publish_worker, release_offloaded_pages


def cached_rows():
    """Bounded tail, not a full runtime journal scan."""
    files=list((PROJECT_ROOT/"runtime").glob("**/cycles.jsonl"))
    files=sorted((p for p in files if "assembly" not in p.parts),key=lambda p:p.stat().st_mtime,reverse=True)
    for path in files:
        rows=[]
        with path.open("rb") as handle:
            handle.seek(max(0,path.stat().st_size-16*1024**2))
            for line in handle.read().splitlines():
                try:row=json.loads(line)
                except ValueError:continue
                decision=row.get("decision") or {}
                if decision.get("raw_outputs") and decision.get("trading_output"):
                    rows.append(row)
        rows=sorted({r["timestamp"]:r for r in rows}.values(),key=lambda r:r["timestamp"])
        if len(rows)>=8:return rows[-24:],str(path)
    raise ValueError("실제 native 출력 cache가 8시점 이상 필요합니다. 합성 출력으로 시험하지 않습니다.")


def evaluate(args,model,recipe,phase,rows,native,panel,state_file):
    model.load_assembly_state(state_file)
    model.apply_assembly_recipe(recipe)
    # Every pair starts from the same fresh capital. Reusing a Champion book
    # from an earlier candidate would skip bars and lose its measured drawdown.
    side="candidate" if recipe["candidate_id"]==args.candidate_id else "champion"
    directory=args.state/"assembly"/args.candidate_id/side/phase
    # Restart interrupted trials from their real saved books; no operational reset.
    bridge=TradingMoEPaper(directory,credit_seconds=60)
    initial=bridge.paper_account.state["books"]["USD"]["initial_cash"]
    progress_file=directory/"progress.json"
    progress=read(progress_file)
    peak=progress.get("peak",initial);drawdown=progress.get("drawdown",0)
    seconds=progress.get("seconds",0);decisions=progress.get("decisions",0)
    used=set(progress.get("used_experts",[]))
    def save_progress():
        atomic_json(progress_file,{"peak":peak,"drawdown":drawdown,"seconds":seconds,
            "decisions":decisions,"used_experts":sorted(used)})
    for row in rows:
        if (args.state/"stop.request").exists():raise InterruptedError("사용자가 시험을 정지했습니다.")
        stamp=row["timestamp"]
        index=int(np.flatnonzero(panel.dates==np.datetime64(stamp))[0])
        if bridge.paper_account.state.get("last_timestamp") and np.datetime64(stamp)<=np.datetime64(bridge.paper_account.state["last_timestamp"]):continue
        started=time.perf_counter()
        bridge.advance(panel,index)
        pstate,astate=bridge.paper_account.model_inputs(panel,index)
        account=torch.tensor(np.column_stack([pstate,np.broadcast_to(astate,(len(pstate),len(astate)))]),dtype=torch.float32)[None]
        held=bool(bridge.paper_account.state["books"]["USD"]["positions"].get("ETHUSDT"))
        native_index=int(np.flatnonzero(native.timestamp==pd.Timestamp(stamp))[0])
        policy_data=model.macro_input_adapter(native,native_index,int(held))
        packets=[]
        for packet in row["decision"]["raw_outputs"]:
            if packet["expert"].startswith("macrophft_"):continue
            if packet["expert"] not in recipe["enabled_experts"]:continue
            age=(pd.Timestamp(stamp)-pd.Timestamp(packet["as_of"])).total_seconds()
            if age<0:raise ValueError("cache에 미래 시점 출력이 있습니다.")
            if age>recipe["refresh_seconds"].get(packet["expert"],7200):continue
            packets.append(packet)
        for key in model.controller.macro_policy_ids:
            if key not in recipe["enabled_experts"]:continue
            data={**policy_data,"variant":model.experts[key].entry["variant"]}
            with registry_owner(model.gpu_lock),model.scheduler.work("candidate_live"):
                packet=model.experts[key](model.root,data,args.device)
            packet.update(expert=key,native_features_verified=True)
            packets.append(packet)
        snapshot={"as_of":stamp,"symbols":panel.symbols,"currencies":{s:"USD" for s in panel.symbols},
            "tradable_symbols":[s for j,s in enumerate(panel.symbols) if panel.observed[index,j]],
            "current_weights":{s:float(pstate[j][1]) for j,s in enumerate(panel.symbols)},"expert_inputs":{}}
        with torch.no_grad():decision,_=model(snapshot,account,packets=packets,device=args.device,explore=False)
        decision["current_weights"]=snapshot["current_weights"]
        bridge.submit(decision,panel,index,paper_executable=True)
        book=bridge.paper_account.snapshot()["books"]["USD"]
        peak=max(peak,book["equity"]);drawdown=max(drawdown,1-book["equity"]/peak)
        decisions+=1;seconds+=time.perf_counter()-started
        used.update(p["expert"] for p in packets)
        save_progress()
        publish_worker(args.state,model,bridge,decision=decision,status="running",evaluation_stage=phase,
            assembly_candidate_id=args.candidate_id,message=phase+" · "+recipe["candidate_id"],learning_active=False,
            selected_experts=decision["used_experts"])
        with (directory/"decisions.jsonl").open("a",encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp":stamp,"trading_output":decision["trading_output"],"NAV":book["equity"],
                "used_experts":sorted(used),"reward":bridge.paper_account.reward_points()})+"\n")
    # One actual later bar matures orders/reward under the same fee engine.
    last=rows[-1]["timestamp"]
    after=np.flatnonzero(panel.dates>np.datetime64(last))
    if len(after) and (not bridge.paper_account.state.get("last_timestamp") or
            panel.dates[int(after[0])]>np.datetime64(bridge.paper_account.state["last_timestamp"])):
        bridge.advance(panel,int(after[0]))
    book=bridge.paper_account.snapshot()["books"]["USD"]
    drawdown=max(drawdown,1-book["equity"]/max(peak,book["equity"]))
    save_progress()
    return {"initial_NAV":initial,"final_NAV":book["equity"],"net_return":book["equity"]/initial-1,
        "pnl":book["net_pnl"],"trades":book["trade_count"],"fees":book["fees"],
        "slippage":book.get("slippage",0),"max_drawdown":drawdown,"seconds":seconds,
        "decisions":decisions,"reward":bridge.paper_account.reward_points(),"replay":bridge.replay.stats(),
        "used_experts":sorted(used),"first_as_of":rows[0]["timestamp"],"last_as_of":last}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--state",type=Path,required=True);parser.add_argument("--assembly-root",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0");parser.add_argument("--interval",type=float,default=.1)
    parser.add_argument("--resume",action="store_true");parser.add_argument("--continuous",action="store_true")
    args=parser.parse_args();args.state.mkdir(parents=True,exist_ok=True)
    recipe=read(args.assembly_root/"current_recipe.json");champion=read(args.assembly_root/"champion_recipe.json")
    args.candidate_id=recipe["candidate_id"]
    result_file=args.assembly_root/"results"/(args.candidate_id+".json");result_file.parent.mkdir(exist_ok=True)
    result={"candidate_id":args.candidate_id,"state":"blocked","scores":{}}
    try:
        torch.set_num_threads(4)
        publish_worker(args.state,status="loading",evaluation_stage="replay",assembly_candidate_id=args.candidate_id,error=None)
        rows,source=cached_rows()
        # Decode only the small trained modules and six real native policy bodies.
        # All eight large market expert bodies stay in their shared file/cache.
        model,_=TradingMoE.load_checkpoint(args.checkpoint,cached_market=True)
        if args.device.startswith("cuda") and not torch.cuda.is_available():raise ValueError("CUDA를 사용할 수 없습니다.")
        model.set_learning_device(args.device);release_offloaded_pages()
        states=args.assembly_root/"trainable";states.mkdir(exist_ok=True)
        champion_state=champion.get("trainable_state")
        if not champion_state:
            champion_state=str(states/"champion-base.pt");model.save_assembly_state(champion_state)
        else:model.load_assembly_state(champion_state)
        candidate_state=states/(args.candidate_id+".pt")
        if not candidate_state.exists():model.save_assembly_state(candidate_state)
        result["trainable_state"]=str(candidate_state)
        result["small_state_bytes"]=candidate_state.stat().st_size
        native=pd.read_feather(args.root/"native_data/MacroHFT/df_val.feather")
        native.timestamp=pd.to_datetime(native.timestamp)
        market=native[["timestamp","open","high","low","close","volume"]].rename(columns={"timestamp":"date"})
        market["symbol"]="ETHUSDT";market["market"]="US";market["asset_class"]="crypto"
        stocks=pd.read_csv(PROJECT_ROOT/"data/global_market_daily.csv")
        stocks=stocks[(stocks.symbol=="AAPL")&(pd.to_datetime(stocks.date)<pd.Timestamp(rows[0]["timestamp"]))].tail(64)
        panel=GlobalMarketPanel("assembly_ETHUSDT",raw_frame=pd.concat([stocks,market],ignore_index=True))
        panel.groups["ETHUSDT"]=("BINANCE_USDT","crypto");panel.closes[:,panel.symbols.index("ETHUSDT")]*=.001
        split=max(4,len(rows)//3)
        stages=[("replay",rows[:split]),("paper",rows[split:])]
        for phase,interval in stages:
            champion_score=evaluate(args,model,champion,phase,interval,native,panel,champion_state)
            candidate_score=evaluate(args,model,recipe,phase,interval,native,panel,candidate_state)
            delta=candidate_score["net_return"]-champion_score["net_return"]
            result["scores"][phase]={"champion":champion_score,"candidate":candidate_score,"delta":delta,"source":source}
            if not np.isfinite(delta):raise ValueError("평가값이 유한하지 않습니다.")
            if phase=="replay" and delta<-.001:
                result.update(state="rejected",reason="replay 예선: Champion 대비 -0.10%p 미만")
                break
        else:
            compared=result["scores"]["paper"]
            passes=compared["delta"]>0 and compared["candidate"]["net_return"]>0 and compared["candidate"]["max_drawdown"]<=compared["champion"]["max_drawdown"]+.01
            result.update(state="qualified" if passes else "rejected",
                reason="분리된 paper 구간 비용 차감 비교 통과" if passes else "paper 비교: 양수 수익·Champion 초과·손실폭 조건 미충족")
        result["evaluation_note"]="저장된 실제 시장 expert 출력 + 해당 계좌 previous_action으로 재계산한 native ETH policy · 탐험 OFF · 평가 중 학습 없음"
    except InterruptedError as exc:result.update(state="paused",reason=str(exc))
    except Exception as exc:result.update(state="blocked",reason=f"{type(exc).__name__}: {exc}")
    result["completed_at"]=datetime.now(timezone.utc).isoformat()
    atomic_json(result_file,result)
    publish_worker(args.state,status="stopped",evaluation_stage=result["state"],message=result.get("reason"),error=None,
        assembly_candidate_id=args.candidate_id,small_state_bytes=result.get("small_state_bytes",0),checkpoint_copies=0)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
