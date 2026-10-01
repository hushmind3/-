"""Kiwoom production WebSocket adapter for normalized one-minute bars."""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import requests
import websockets

from .provider_credentials import get_credentials, read_settings

WS_URLS = {
    "real": "wss://api.kiwoom.com:10000/api/dostk/websocket",
    "paper": "wss://mockapi.kiwoom.com:10000/api/dostk/websocket",
}
SEOUL = ZoneInfo("Asia/Seoul")


def _number(value, *, absolute=False):
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        out = float(text)
        return abs(out) if absolute else out
    except ValueError:
        return None


class KiwoomRealtimeStream:
    """Connect, login, register 0B trade ticks, aggregate and enqueue bars."""
    def __init__(self, runtime: Path, instruments: list[dict], stop: asyncio.Event,
                 output: queue.Queue, status: Callable[[dict], None] | None = None):
        self.runtime = Path(runtime)
        self.environment = read_settings(self.runtime).get("environment", "paper")
        self.instruments = instruments
        self.stop = stop
        self.output = output
        self.status = status or (lambda _state: None)
        self.by_code = {}
        self.krx_codes = []
        self.nxt_codes = []
        self.subscription_codes=[]
        self.nxt_seen_codes = set()
        self.krx_seen_codes = set()
        for instrument in instruments:
            code = str(instrument.get("provider_symbol", instrument["symbol"])).split(".")[0]
            self.by_code[code] = instrument
            self.by_code[f"{code}_NX"] = instrument
            self.by_code[f"{code}_AL"] = instrument
            self.krx_codes.append(code)
            self.nxt_codes.append(f"{code}_NX")
            self.subscription_codes.append(f"{code}_AL" if self.environment=="real" else code)
        self.subscription_codes=list(dict.fromkeys(self.subscription_codes))
        if len(self.subscription_codes)>200:
            raise ValueError("Kiwoom session permits 200 subscribed identities; configured instruments were not silently omitted")
        self.bars = {}
        self.retry = 2
        self.stats = {"connected": False, "symbols": len(instruments),
                      "last_message_utc": None, "last_error": None, "source": "kiwoom_ws",
                      "subscription_mode":"SOR_combined" if self.environment=="real" else "KRX_mock",
                      "subscription_count":len(self.subscription_codes),"subscription_limit":200,
                      "nxt_subscribed": len(self.nxt_codes), "nxt_active_symbols": 0,
                      "nxt_last_message_utc": None, "nxt_ticks": 0,
                      "krx_subscribed": len(self.krx_codes), "krx_active_symbols": 0,
                      "krx_last_message_utc": None, "krx_ticks": 0}

    def _set(self, **values):
        self.stats.update(values)
        self.status(dict(self.stats))

    def _access_token(self):
        cfg = read_settings(self.runtime)
        if cfg.get("provider") != "kiwoom":
            raise RuntimeError("Kiwoom is not the selected market-data provider")
        if cfg.get("environment") not in WS_URLS:
            raise RuntimeError("Unsupported Kiwoom environment")
        saved = get_credentials(self.runtime, cfg["environment"])
        app_key = saved["app_key"]
        secret = saved["secret"]
        if not app_key or not secret:
            raise RuntimeError("Kiwoom App Key and Secret Key are not saved")
        host = "https://api.kiwoom.com" if cfg["environment"] == "real" else "https://mockapi.kiwoom.com"
        response = requests.post(
            host + "/oauth2/token",
            json={"grant_type": "client_credentials", "appkey": app_key, "secretkey": secret},
            headers={"Content-Type": "application/json;charset=UTF-8"}, timeout=15)
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Kiwoom token endpoint returned HTTP {response.status_code} non-JSON") from exc
        token = data.get("token")
        if not response.ok or not token:
            code = data.get("return_code", "")
            detail = data.get("return_msg", "token missing")
            raise RuntimeError(f"Kiwoom token rejected ({code}): {detail}")
        return token

    @staticmethod
    def _decode(raw):
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return json.loads(raw)

    def _on_message(self, message):
        if not isinstance(message, dict):
            return
        if message.get("trnm") == "PING":
            return "PING"
        if message.get("trnm") != "REAL":
            return None
        data = message.get("data") or []
        if isinstance(data, dict):
            data = [data]
        accepted = 0
        nxt_accepted = 0
        krx_accepted = 0
        for event in data:
            if not isinstance(event, dict):
                continue
            kind = str(event.get("type", ""))
            if kind and kind != "0B":
                continue
            code = str(event.get("item") or event.get("name") or "").split(".")[0]
            instrument = self.by_code.get(code)
            values = event.get("values") or {}
            if instrument is None or not isinstance(values, dict):
                continue
            price = _number(values.get("10"), absolute=True)
            if not price or price <= 0:
                continue
            stamp = str(values.get("20") or "")
            if len(stamp) < 6 or not stamp[:6].isdigit():
                stamp = datetime.now(SEOUL).strftime("%H%M%S")
            now = datetime.now(SEOUL)
            local = now.replace(hour=int(stamp[0:2]), minute=int(stamp[2:4]),
                                second=int(stamp[4:6]), microsecond=0)
            bucket = local.replace(second=0, microsecond=0)
            symbol = instrument["symbol"]
            old = self.bars.get(symbol)
            if old and bucket < old["_bucket"]:
                continue
            if old and bucket > old["_bucket"]:
                self._flush(symbol, old)
                old = None
            signed_volume=_number(values.get("15"))
            volume=abs(signed_volume or 0.0)
            buy_volume=volume if signed_volume is not None and signed_volume>0 else 0.0
            sell_volume=volume if signed_volume is not None and signed_volume<0 else 0.0
            bid = _number(values.get("28"), absolute=True)
            ask = _number(values.get("27"), absolute=True)
            if old is None:
                self.bars[symbol] = {
                    "_bucket": bucket, "date": bucket.astimezone(timezone.utc).isoformat(),
                    "symbol": symbol, "market": instrument.get("market", "KRX"),
                    "asset_class": instrument.get("asset_class", "equity"),
                    "open": price, "high": price, "low": price, "close": price,
                    "volume": volume, "bid": bid, "ask": ask,
                    "trade_count": 1,
                    "buy_volume":buy_volume,"sell_volume":sell_volume,
                }
            else:
                old["high"] = max(old["high"], price)
                old["low"] = min(old["low"], price)
                old["close"] = price
                old["volume"] += volume
                old["trade_count"] += 1
                old["buy_volume"]+=buy_volume;old["sell_volume"]+=sell_volume
                old["bid"] = bid if bid is not None else old["bid"]
                old["ask"] = ask if ask is not None else old["ask"]
            accepted += 1
            if code.endswith("_NX") or str(values.get("9081", "")).upper() in ("NXT","2"):
                nxt_accepted += 1
                self.nxt_seen_codes.add(code.removesuffix("_NX").removesuffix("_AL"))
            else:
                krx_accepted += 1
                self.krx_seen_codes.add(code.removesuffix("_AL"))
        if accepted:
            now_utc = datetime.now(timezone.utc).isoformat()
            state = {"last_message_utc": now_utc, "last_error": None}
            if nxt_accepted:
                state.update(nxt_last_message_utc=now_utc,
                             nxt_ticks=self.stats["nxt_ticks"] + nxt_accepted,
                             nxt_active_symbols=len(self.nxt_seen_codes))
            if krx_accepted:
                state.update(krx_last_message_utc=now_utc,
                             krx_ticks=self.stats["krx_ticks"] + krx_accepted,
                             krx_active_symbols=len(self.krx_seen_codes))
            self._set(**state)
        return None

    def _flush(self, symbol, bar):
        row = {key: value for key, value in bar.items() if not key.startswith("_")}
        self.output.put(row)

    async def _session(self):
        token = await asyncio.to_thread(self._access_token)
        async with websockets.connect(WS_URLS[self.environment], open_timeout=15, close_timeout=3,
                                      ping_interval=None, max_size=4 * 1024 * 1024) as ws:
            await ws.send(json.dumps({"trnm": "LOGIN", "token": token}, separators=(",", ":")))
            deadline = time.monotonic() + 15
            logged_in = False
            while time.monotonic() < deadline:
                response = self._decode(await asyncio.wait_for(ws.recv(), timeout=15))
                if response.get("trnm") == "PING":
                    await ws.send(json.dumps(response, separators=(",", ":")))
                    continue
                if response.get("trnm") == "LOGIN":
                    if int(response.get("return_code", -1)) != 0:
                        raise RuntimeError(f"Kiwoom WebSocket login rejected: {response.get('return_msg', response)}")
                    logged_in = True
                    break
            if not logged_in:
                raise RuntimeError("Kiwoom WebSocket login response timed out")
            self._set(connected=False, last_error=None)
            # Official 0B item supports SOR (_AL), one identity for both venues.
            groups=[self.subscription_codes[i:i+50] for i in range(0,len(self.subscription_codes),50)]
            registrations_left=len(groups)
            for index,codes in enumerate(groups,1):
                await ws.send(json.dumps({"trnm":"REG","grp_no":str(index),"refresh":"1",
                    "data":[{"item":codes,"type":["0B"]}]},separators=(",",":")))
            while not self.stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                message = self._decode(raw)
                if message.get("trnm") == "PING":
                    await ws.send(json.dumps(message, separators=(",", ":")))
                elif message.get("trnm") == "REG" and int(message.get("return_code", 0)) != 0:
                    raise RuntimeError(f"Kiwoom WebSocket registration rejected: {message.get('return_msg', message)}")
                elif message.get("trnm")=="REG":
                    registrations_left=max(0,registrations_left-1)
                    if registrations_left==0:
                        self._set(connected=True,last_error=None)
                else:
                    self._on_message(message)

    async def run_async(self):
        while not self.stop.is_set():
            try:
                await self._session()
                self.retry = 2
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._set(connected=False, last_error=f"{type(exc).__name__}: {exc}")
                logging.error("Kiwoom WebSocket disconnected: %s; retrying in %ss", exc, self.retry)
                for _ in range(self.retry):
                    if self.stop.is_set():
                        break
                    await asyncio.sleep(1)
                self.retry = min(60, self.retry * 2)
        for symbol, bar in list(self.bars.items()):
            self._flush(symbol, bar)
        self.bars.clear()
        self._set(connected=False)

    def run(self):
        asyncio.run(self.run_async())
