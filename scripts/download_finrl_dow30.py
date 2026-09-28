"""Fetch a real chronological Dow-30 + VIX OHLCV warm-up set."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import date,datetime,timedelta,timezone
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from download_global_data import fetch

TICKERS=("AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
"GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM","MRK","MSFT",
"NKE","PG","SHW","TRV","UNH","V","VZ","WMT","^VIX")

def main():
    start=int(datetime(2018,1,1,tzinfo=timezone.utc).timestamp())
    end=int((datetime.now(timezone.utc)+timedelta(days=1)).timestamp())
    items=[(symbol,("US","volatility" if symbol=="^VIX" else "equity")) for symbol in TICKERS]
    rows=[]; errors={}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(fetch,item,start,end):item[0] for item in items}
        for f in as_completed(futures):
            data,error=f.result(); rows.extend(data)
            if error: errors[futures[f]]=error
            print(f"{futures[f]}: {len(data)}"+(f" ERROR {error}" if error else ""),flush=True)
    if not rows: raise RuntimeError("Yahoo returned no Dow30/VIX observations")
    import pandas as pd
    frame=pd.DataFrame(rows).sort_values(["date","symbol"]).drop_duplicates(["date","symbol"],keep="last")
    out=Path("data/external_sources/finrl_base/dow30_vix_daily.csv"); out.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out,index=False)
    report={"source":"Yahoo Finance chart API","reference_pipeline":"FinRL Dow30 tutorial universe + VIX",
      "rows":len(frame),"symbols":frame.symbol.nunique(),"date_min":frame.date.min(),"date_max":frame.date.max(),"errors":errors,"file":str(out)}
    out.with_suffix(".json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
