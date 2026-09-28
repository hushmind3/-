"""Build audited, position-aware warm-start decisions and a reusable replay."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from stockrl.global_online import Experience, GlobalReplayBuffer
from stockrl.global_transformer import GlobalMarketPanel
from stockrl.research_ingest import (fi2010_book_to_features, fi2010_file_to_arrays,
                                    trademaster_csv_to_global)

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/external_sources"
OUT=ROOT/"runtime-global-research-pretrain/teacher_replay.pt"
ACTION={"SELL":0,"HOLD":1,"BUY":2}


def build_finrl():
    path=DATA/"finrl_imitation/trade_data.csv"
    df=pd.read_csv(path).sort_values(["tic","date"]).copy()
    df["previous_weight"]=df.groupby("tic")["greedy"].shift(1).fillna(0.0)
    delta=df["greedy"]-df["previous_weight"]
    df["action"]=np.select([delta>1e-6,delta < -1e-6],["BUY","SELL"],default="HOLD")
    df["symbol"]=df.tic.astype(str); df["market"]="US"; df["asset_class"]="EQUITY"
    target=DATA/"finrl_imitation/teacher_records.csv"
    df.to_csv(target,index=False)
    return {"rows":len(df),"action_counts":df.action.value_counts().to_dict(),"file":str(target),
            "label":"change in the source's greedy target allocation; not verified fills"}


def build_fi2010():
    base=DATA/"fi2010/selected/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_1.txt"
    raw,labels=fi2010_file_to_arrays(base,sample_stride=100)
    features=np.stack([fi2010_book_to_features(r) for r in raw])
    mid=(raw[:,0]+raw[:,20])*.5
    features[:,0]=np.r_[0,np.diff(mid)]
    features[:,1]=mid-np.r_[np.zeros(5),mid[:-5]]
    features[:,2]=mid-np.r_[np.zeros(20),mid[:-20]]
    label=labels[:,0]
    action=np.where(label==1,"SELL",np.where(label==2,"HOLD","BUY"))
    frame=pd.DataFrame(features,columns=[f"f{i}" for i in range(features.shape[1])])
    frame["event_index"]=np.arange(0,len(features)*100,100)
    frame["action"]=action; frame["horizon_events"]=10
    frame["source"]="FI-2010 label warmstart"
    target=DATA/"fi2010/teacher_records_train_cf1.csv"
    frame.to_csv(target,index=False)
    return {"rows":len(frame),"actions":pd.Series(action).value_counts().to_dict(),"file":str(target),
            "heldout_test_labels_used_for_training":False}


def build_macro_replay():
    source=DATA/"macrophft"
    data=pd.read_feather(source/"data/df_test.feather").sort_values("timestamp").drop_duplicates("timestamp")
    # Exactly the every-10th held-out observations used to generate teacher
    # predictions, with native book/flow fields retained for the 17-slot panel.
    data=data.iloc[::10].copy()
    panel_csv=source/"global_test_stride10.csv"
    panel_data=pd.DataFrame({
        "date":pd.to_datetime(data.timestamp,utc=True).astype(str),"symbol":"ETHUSDT",
        "market":"CRYPTO","asset_class":"CRYPTO_PERP","open":data.open,
        "high":data.high,"low":data.low,"close":data.close,"volume":data.volume,
        "bid":data.bid1_price,"ask":data.ask1_price,"bid_size":data.bid1_size,
        "ask_size":data.ask1_size,"buy_volume":data.buy_volume,"sell_volume":data.sell_volume,
        "trade_count":np.zeros(len(data)),
    })
    panel_data.to_csv(panel_csv,index=False)
    panel=GlobalMarketPanel(panel_csv)
    date_index={pd.Timestamp(d):i for i,d in enumerate(panel.dates)}
    teacher=pd.read_csv(source/"teacher_outputs.csv")
    # Keep a balanced 10% temporal sample from each of six checkpoints; all
    # complete teacher decisions remain archived in teacher_outputs.csv.
    buffer=GlobalReplayBuffer(capacity=20_000,seed=7)
    counts={}
    for model_name, group in teacher.groupby("model",sort=True):
        group=group.iloc[::10]
        added=0
        for row in group.itertuples(index=False):
            date=pd.Timestamp(row.timestamp)
            if date.tzinfo is not None: date=date.tz_convert("UTC").tz_localize(None)
            ti=date_index.get(date)
            action=ACTION.get(str(row.action).upper())
            if ti is None or action is None: continue
            x,sid,mid,aid,mask=panel.window(ti,32)
            buffer.add(Experience(x[0].numpy().astype(np.float16),sid[0].numpy(),mid[0].numpy(),
                                  aid[0].numpy(),mask[0].numpy(),0,action,0.0,str(date),
                                  f"teacher:MacroHFT:{model_name}",0.0,"teacher_warmstart_v1"))
            added+=1
        counts[model_name]=added
    OUT.parent.mkdir(parents=True,exist_ok=True); buffer.save(OUT)
    return {"rows":len(buffer),"per_teacher":counts,"file":str(OUT),"replay_bytes":OUT.stat().st_size,
            "sampled_share":0.1,"source_output_rows":len(teacher)}


def build_trademaster_panels():
    source=DATA/"trademaster/order_execution_BTC"
    results={}
    for split in ("train","valid","test"):
        path=source/f"{split}.csv"
        target=source/f"global_{split}.csv"
        results[split]=trademaster_csv_to_global(path,target)
    return results


def main():
    summary={"finrl_imitation":build_finrl(),"fi2010":build_fi2010(),
             "trademaster_panels":build_trademaster_panels(),"macrophft_replay":build_macro_replay()}
    path=ROOT/"data/external_sources/ingest_summary.json"
    path.write_text(json.dumps(summary,indent=2),encoding="utf8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__": main()
