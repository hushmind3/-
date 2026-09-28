"""Download daily global OHLCV proxies from Yahoo Finance chart endpoint."""
from __future__ import annotations
import argparse, datetime as dt, json, time, urllib.parse, urllib.request
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

UNIVERSE = {
    # Korean equities and indices
    "005930.KS":("KR","equity"), "000660.KS":("KR","equity"), "035420.KS":("KR","equity"),
    "005380.KS":("KR","equity"), "051910.KS":("KR","equity"), "^KS11":("KR","index"), "^KQ11":("KR","index"),
    # US equities, index futures, indices and sectors
    "AAPL":("US","equity"), "MSFT":("US","equity"), "NVDA":("US","equity"), "AMZN":("US","equity"),
    "GOOGL":("US","equity"), "JPM":("US","equity"), "XOM":("US","equity"),
    "^GSPC":("US","index"), "^IXIC":("US","index"), "^DJI":("US","index"), "ES=F":("US","index_future"), "NQ=F":("US","index_future"),
    "XLK":("US","sector_etf"), "XLF":("US","sector_etf"), "XLE":("US","sector_etf"), "SPY":("US","index_etf"), "QQQ":("US","index_etf"),
    # Major international equity indices
    "^N225":("JP","index"), "^FTSE":("UK","index"), "^GDAXI":("DE","index"), "^FCHI":("FR","index"),
    "^HSI":("HK","index"), "^STOXX50E":("EU","index"), "^AXJO":("AU","index"), "^BSESN":("IN","index"),
    # Rates, currencies, commodities and volatility
    "^IRX":("US","treasury_yield"), "^FVX":("US","treasury_yield"), "^TNX":("US","treasury_yield"), "^TYX":("US","treasury_yield"),
    "TLT":("US","bond_etf"), "EURUSD=X":("FX","currency"), "JPY=X":("FX","currency"), "KRW=X":("FX","currency"),
    "GBPUSD=X":("FX","currency"), "^VIX":("US","volatility"), "CL=F":("US","commodity_future"),
    "GC=F":("US","commodity_future"), "SI=F":("US","commodity_future"), "HG=F":("US","commodity_future"),
    "BZ=F":("US","commodity_future"), "NG=F":("US","commodity_future"), "ZC=F":("US","commodity_future"),
    "GLD":("US","commodity_etf"), "SLV":("US","commodity_etf"), "CPER":("US","commodity_etf"),
}

def fetch(item, start, end):
    symbol,(market,asset)=item
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol,safe='')}?period1={start}&period2={end}&interval=1d&events=history"
    req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 global-market-research/1.0"})
    last=None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req,timeout=25) as r: data=json.load(r)
            result=data["chart"]["result"]
            if not result: raise ValueError(data["chart"].get("error"))
            q=result[0]; quote=q["indicators"]["quote"][0]
            rows=[]
            for i,stamp in enumerate(q.get("timestamp",[])):
                values=[quote[k][i] if quote[k] and i<len(quote[k]) else None for k in ("open","high","low","close","volume")]
                if any(v is None for v in values): continue
                tz=("America/New_York" if symbol.endswith("=X") else
                    q.get("meta",{}).get("exchangeTimezoneName","UTC"))
                rows.append({"date":dt.datetime.fromtimestamp(stamp,ZoneInfo(tz)).date().isoformat(),"symbol":symbol,
                             "market":market,"asset_class":asset,"open":values[0],"high":values[1],"low":values[2],
                             "close":values[3],"volume":values[4]})
            return rows,None
        except Exception as e:
            last=f"{type(e).__name__}: {e}"; time.sleep(.5*(attempt+1))
    return [],last

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",default="data/global_market_daily.csv")
    p.add_argument("--start",default="2023-01-01"); p.add_argument("--end",default=dt.date.today().isoformat())
    p.add_argument("--workers",type=int,default=4); a=p.parse_args()
    start=int(dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.timezone.utc).timestamp())
    end=int((dt.datetime.fromisoformat(a.end)+dt.timedelta(days=1)).replace(tzinfo=dt.timezone.utc).timestamp())
    results=[]; errors={}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures={ex.submit(fetch,item,start,end):item[0] for item in UNIVERSE.items()}
        for f in as_completed(futures):
            rows,error=f.result(); results.extend(rows)
            if error: errors[futures[f]]=error
            print(f"{futures[f]}: {len(rows)} rows"+(f" ({error})" if error else ""))
    if not results: raise SystemExit("No market data downloaded")
    df=pd.DataFrame(results)
    df=df[pd.to_datetime(df.date).dt.weekday<5].sort_values(["date","symbol"])
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False)
    report={"file":str(out),"rows":len(df),"symbols":int(df.symbol.nunique()),"date_min":df.date.min(),"date_max":df.date.max(),"errors":errors,
            "source":"Yahoo Finance chart API; daily OHLCV; no order-book/options chain history"}
    out.with_suffix(".download.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
