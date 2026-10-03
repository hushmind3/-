"""Market sessions and persisted progress health."""
from __future__ import annotations
from datetime import datetime, time as day_time, timezone
from ..state_io import read_json as _json

def _market_group(item: dict) -> str:
    market=item.get("market",""); asset=item.get("asset_class","")
    if market in ("KRX","KOSDAQ") and asset=="index": return "korea_index"
    if market in ("KRX","KOSDAQ"): return "korea"
    if asset in ("index_future","commodity_future"): return "futures"
    if asset=="yield": return "bonds"
    if asset=="currency": return "fx"
    if asset=="crypto": return "crypto"
    if market=="US": return "us"
    return f"global:{market}"

def _session(group: str) -> str:
    # Session labels are a clock-based guide only; exchange holidays and halts
    # are not available from the current free data providers.
    from dateutil import tz
    if group=="crypto": return "가상자산 · 24시간 · 연중"
    if group.startswith("global:"):
        venue=group.split(":",1)[1]
        specs={"Japan":("Asia/Tokyo","일본 현물 09:00–11:30 / 12:30–15:30"),
               "HongKong":("Asia/Hong_Kong","홍콩 현물 09:30–12:00 / 13:00–16:00"),
               "Germany":("Europe/Berlin","독일 현물 09:00–17:30"),
               "UK":("Europe/London","영국 현물 08:00–16:30")}
        if venue in specs:
            zone,hours=specs[venue]; now=datetime.now(tz.gettz(zone)); open_now=now.weekday()<5
            t=now.time()
            if venue=="Japan": open_now &= day_time(9)<=t<day_time(11,30) or day_time(12,30)<=t<day_time(15,30)
            elif venue=="HongKong": open_now &= day_time(9,30)<=t<day_time(12) or day_time(13)<=t<day_time(16)
            elif venue=="Germany": open_now &= day_time(9)<=t<day_time(17,30)
            else: open_now &= day_time(8)<=t<day_time(16,30)
            return f"{hours} · 현지 {now:%H:%M} · {'장중' if open_now else '장외'} · 휴장일 미반영"
        return "현지 거래소별 시간"
    if group=="fx":
        now=datetime.now(tz.gettz("America/New_York")); wd=now.weekday(); t=now.time()
        opened=(wd==6 and t>=day_time(17)) or (0<=wd<4) or (wd==4 and t<day_time(17))
        return f"FX · 주중 24/5 · 뉴욕 {now:%H:%M} · {'거래 시간대' if opened else '주말 휴장 시간대'}"
    if group in ("futures","bonds","us"):
        now=datetime.now(tz.gettz("America/New_York")); wd=now.weekday(); t=now.time()
        if group=="us": opened=wd<5 and day_time(9,30)<=t<day_time(16)
        elif group=="bonds": opened=wd<5 and day_time(8)<=t<day_time(17)
        else: opened=((wd==6 and t>=day_time(18)) or (0<=wd<4) or (wd==4 and t<day_time(17))) and not day_time(17)<=t<day_time(18)
        hours={"us":"미국 현물 09:30–16:00 ET","bonds":"미 국채지표 08:00–17:00 ET","futures":"선물 주중 23시간(정비시간 제외)"}[group]
        return f"{hours} · 뉴욕 {now:%H:%M} · {'장중' if opened else '장외/정비'} · 휴장일 미반영"
    if group=="korea_index":
        now=datetime.now(tz.gettz("Asia/Seoul")); t=now.time()
        opened=now.weekday()<5 and day_time(9)<=t<day_time(15,30)
        return f"한국 지수 09:00–15:30 KST · 서울 {now:%H:%M} · {'장중' if opened else '장외'} · 휴장일 미반영"
    if group=="korea":
        now=datetime.now(tz.gettz("Asia/Seoul")); t=now.time()
        if now.weekday()>=5: phase="주말 장외"
        elif day_time(8)<=t<day_time(8,50): phase="NXT 프리마켓"
        elif day_time(8,50)<=t<day_time(9): phase="NXT 종료 · KRX 개장 대기"
        elif day_time(9,0,30)<=t<day_time(15,20): phase="KRX 정규장 · NXT 메인마켓"
        elif day_time(9)<=t<day_time(15,30): phase="KRX 정규장"
        elif day_time(15,30)<=t<day_time(15,40): phase="NXT 호가접수 · 체결 대기"
        elif day_time(15,40)<=t<day_time(20): phase="NXT 애프터마켓"
        else: phase="장외"
        return ("KRX 09:00–15:30 · NXT 프리 08:00–08:50 / 메인 09:00:30–15:20 / "
                f"애프터 15:40–20:00 KST · 서울 {now:%H:%M} · {phase} · 휴장일 미반영")
    return "현지 거래소별 시간"

def _market_overview(instruments: list[dict], decisions: list[dict],
                     fresh_symbols: set[str] | None = None) -> list[dict]:
    fresh_symbols = fresh_symbols or set()
    titles={"korea":"한국 주식","korea_index":"한국 지수","us":"미국 주식·NASDAQ·지수",
            "futures":"지수·원자재 선물","bonds":"국채 금리","fx":"환율","crypto":"가상자산"}
    grouped={}
    for item in instruments:
        key=_market_group(item); label=titles.get(key,key.split(":",1)[1]+" 지수" if key.startswith("global:") else key)
        grouped.setdefault(key,{"key":key,"label":label,"symbols":[]})["symbols"].append(item.get("symbol",""))
    for value in grouped.values():
        symbols=set(value["symbols"]); rows=[d for d in decisions if d.get("symbol") in symbols]
        value["count"]=len(symbols); value["session"]=_session(value["key"])
        value["fresh_count"]=len(symbols & fresh_symbols)
        value["latest_decision"]=max((row.get("date", "") for row in rows), default=None)
        value["actions"]={name:sum(1 for row in rows if row.get("action")==name) for name in ("BUY","HOLD","SELL")}
    order=("korea","korea_index","us","global:Japan","global:HongKong","global:Germany","global:UK","futures","bonds","fx","crypto")
    return [grouped[key] for key in order if key in grouped]
