"""Run the unchanged 0.5B champion on real parsed public-source panels."""
from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch

from stockrl.global_online import load_model
from stockrl.global_transformer import ACTION_NAMES, GlobalMarketPanel, parameter_count

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runtime-global-cuda-final/champion.pt"
OUT=ROOT/"data/external_sources/global_transformer_predictions.csv"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if DEVICE.type!="cuda": raise RuntimeError("Expected CUDA RTX 3070 inference")
    model,cfg=load_model(BASE,DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    sources={
      "NASDAQ_ITCH_prefix":ROOT/"data/external_sources/nasdaq/itch_snapshots.csv",
      "TradeMaster_BTC_test":ROOT/"data/external_sources/trademaster/order_execution_BTC/global_test.csv",
      "MacroHFT_ETH_test":ROOT/"data/external_sources/macrophft/global_test_stride10.csv",
    }
    rows=[]; metrics={}
    for source,path in sources.items():
        raw=pd.read_csv(path)
        if source.startswith("NASDAQ"):
            symbols=raw.symbol.value_counts().head(3).index.tolist()
            raw=raw[raw.symbol.isin(symbols)].copy()
        temp=ROOT/"data/external_sources"/f"_{source}_panel.csv"
        raw.to_csv(temp,index=False)
        panel=GlobalMarketPanel(temp)
        if not len(panel.dates): continue
        ends=np.unique(np.linspace(min(15,len(panel.dates)-1),len(panel.dates)-1,min(12,len(panel.dates)),dtype=int))
        source_rows=[]; latency=[]
        for ti in ends:
            x,sid,mid,aid,mask=panel.window(int(ti),min(32,cfg.max_seq_len))
            x=x.to(DEVICE,dtype=torch.float16)
            sid=sid.to(DEVICE); mid=mid.to(DEVICE); aid=aid.to(DEVICE); mask=mask.to(DEVICE)
            if DEVICE.type=="cuda": torch.cuda.synchronize(DEVICE)
            started=time.perf_counter()
            with torch.inference_mode(): logits,values=model(x,sid,mid,aid,mask)
            torch.cuda.synchronize(DEVICE); latency.append(time.perf_counter()-started)
            probs=torch.softmax(logits[0].float(),-1).cpu().numpy(); vals=values[0].float().cpu().numpy()
            stamp=str(panel.dates[ti])
            for j,symbol in enumerate(panel.symbols):
                if not panel.observed[ti,j]: continue
                act=int(probs[j].argmax())
                source_rows.append({"source":source,"date":stamp,"symbol":symbol,
                  "action":ACTION_NAMES[act],"confidence":float(probs[j,act]),"value":float(vals[j]),
                  "p_sell":float(probs[j,0]),"p_hold":float(probs[j,1]),"p_buy":float(probs[j,2])})
        rows.extend(source_rows)
        metrics[source]={"rows_in_panel":len(panel.frame),"dates":len(panel.dates),"symbols":len(panel.symbols),
          "prediction_rows":len(source_rows),"action_counts":pd.Series([r['action'] for r in source_rows]).value_counts().to_dict(),
          "inference_p50_ms":round(float(np.median(latency)*1000),3),"timestamps_sampled":len(ends)}
        temp.unlink(missing_ok=True)
    pd.DataFrame(rows).to_csv(OUT,index=False)
    result={"checkpoint":str(BASE),"parameter_count":parameter_count(model),
      "device":torch.cuda.get_device_name(DEVICE),"prediction_rows":len(rows),
      "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(DEVICE),"sources":metrics,"output":str(OUT)}
    (OUT.with_suffix(".json")).write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__": main()
