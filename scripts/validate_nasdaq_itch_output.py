from pathlib import Path
import json, time
import pandas as pd
p=Path('data/external_sources/nasdaq/itch_snapshots_full_corrected.csv')
summary=json.loads(Path(str(p)+'.json').read_text())
started=time.monotonic(); rows=0; bad_ts=0; dupes=0; nonpositive=0; crossed=0; volume=0.0; symbols=set()
first=last=None; prev_ns=None; prev_sym=None
for chunk in pd.read_csv(p,chunksize=500_000):
    rows += len(chunk)
    dt=pd.to_datetime(chunk['date'],utc=True,format='ISO8601',errors='coerce')
    bad_ts += int(dt.isna().sum())
    ns=dt.astype('int64').to_numpy()
    syms=chunk['symbol'].astype(str).to_numpy()
    if len(ns):
        if first is None: first=dt.iloc[0].isoformat()
        last=dt.iloc[-1].isoformat()
        if prev_ns is not None and (ns[0] < prev_ns or (ns[0] == prev_ns and syms[0] < prev_sym)):
            raise RuntimeError(f'cross-chunk order violation at row {rows-len(chunk)+1}')
        if len(ns)>1:
            regress=(ns[1:]<ns[:-1]) | ((ns[1:]==ns[:-1]) & (syms[1:]<syms[:-1]))
            if regress.any(): raise RuntimeError(f'ordering violation within chunk ending row {rows}')
            dupes += int(((ns[1:]==ns[:-1]) & (syms[1:]==syms[:-1])).sum())
        if prev_ns is not None and ns[0]==prev_ns and syms[0]==prev_sym: dupes += 1
        prev_ns=int(ns[-1]); prev_sym=syms[-1]
    symbols.update(chunk.symbol.dropna().astype(str).unique())
    nonpositive += int(((chunk.bid<=0)|(chunk.ask<=0)|(chunk.close<=0)).sum())
    crossed += int((chunk.bid>chunk.ask).sum())
    volume += float(chunk.volume.sum())
    print(f'validated={rows:,}/{summary["snapshot_rows"]:,} unique_symbols={len(symbols):,}',flush=True)
result={'path':str(p),'bytes':p.stat().st_size,'rows':rows,'summary_rows':summary['snapshot_rows'],'rows_match_summary':rows==summary['snapshot_rows'],'messages':summary['messages'],'message_types':summary['message_types'],'symbols':len(symbols),'first_timestamp_utc':first,'last_timestamp_utc':last,'invalid_timestamps':bad_ts,'out_of_order_rows':0,'duplicate_symbol_timestamps':dupes,'nonpositive_price_rows':nonpositive,'crossed_bbo_rows':crossed,'sum_snapshot_volume':volume,'elapsed_seconds':round(time.monotonic()-started,1)}
Path('data/external_sources/nasdaq/itch_snapshots_full.validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
