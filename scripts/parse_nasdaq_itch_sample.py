"""Replay a bounded prefix of the official Nasdaq ITCH 5.0 sample.

Reads original length-prefixed binary messages, maintains displayed orders,
and writes chronological per-symbol book/flow snapshots for the existing
17-feature input adapter. The default bound is intentionally small for a
repeatable parser smoke test; use --max-messages 0 for a full day.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import gzip
from pathlib import Path
import json
import os
import time
from zoneinfo import ZoneInfo

import pandas as pd


def u(b: bytes) -> int:
    return int.from_bytes(b, "big")


def decode(path: Path, output: Path, max_messages: int, bucket: int,
           progress_every: int = 1_000_000, row_batch: int = 100_000,
           resume: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    partial = output.with_suffix(output.suffix + ".part")
    if partial.exists() and not resume:
        raise FileExistsError(f"partial output already exists; inspect or move it first: {partial}")
    if resume and not partial.exists():
        raise FileNotFoundError(f"no partial output to resume: {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)
    symbols: dict[int, str] = {}
    # order id -> (stock locate, side, price in 1/10000 dollars, shares)
    orders: dict[int, tuple[int, str, int, int]] = {}
    book: dict[int, tuple[dict[int, int], dict[int, int]]] = {}
    counts: dict[int, Counter] = defaultdict(Counter)
    rows = []
    types = Counter()
    parsed = 0
    written_rows = 0
    skip_through_event = 0
    if resume:
        with partial.open("rb") as stream:
            stream.seek(max(0, partial.stat().st_size - 131072))
            tail = stream.read().rstrip(b"\r\n").splitlines()
        if len(tail) < 2:
            raise ValueError(f"partial output has no data rows: {partial}")
        skip_through_event = int(tail[-1].rsplit(b",", 1)[-1])
        print(f"resume_replay_until_event={skip_through_event:,}; reconstructing ITCH book from source start", flush=True)
        with partial.open("rb") as stream:
            written_rows = max(0, sum(1 for _ in stream) - 1)
        print(f"resume_existing_snapshot_rows={written_rows:,}", flush=True)
    started = time.monotonic()
    base = datetime(2019, 1, 30, tzinfo=ZoneInfo("America/New_York"))

    def apply(symbol_id: int, side: str, price: int, delta: int) -> None:
        bids, asks = book.setdefault(symbol_id, ({}, {}))
        sidebook = bids if side == "B" else asks
        new = sidebook.get(price, 0) + delta
        if new <= 0:
            sidebook.pop(price, None)
        else:
            sidebook[price] = new

    def flush(event_no: int) -> None:
        nonlocal written_rows
        for sid, c in list(counts.items()):
            sym = symbols.get(sid)
            bids, asks = book.get(sid, ({}, {}))
            if not sym or not bids or not asks:
                continue
            bp, ap = max(bids), min(asks)
            mid = (bp + ap) / 20000.0
            if mid <= 0:
                continue
            ts_ns = int(c.get("last_ns", 0))
            dt = base + timedelta(microseconds=ts_ns / 1000)
            dt_utc = dt.astimezone(timezone.utc).isoformat()
            if event_no <= skip_through_event:
                continue
            row = {
                "date": dt_utc, "symbol": sym, "market": "NASDAQ",
                "asset_class": "EQUITY", "open": mid, "high": mid,
                "low": mid, "close": mid, "volume": float(c["exec_shares"]),
                "bid": bp/10000.0, "ask": ap/10000.0,
                "bid_size": bids[bp], "ask_size": asks[ap],
                "buy_volume": float(c["buy_exec"]), "sell_volume": float(c["sell_exec"]),
                "trade_count": int(c["executions"]), "add_count": int(c["adds"]),
                "cancel_count": int(c["cancels"]), "event_index": event_no,
            }
            rows.append(row)
        counts.clear()
        if len(rows) >= row_batch:
            write_rows()

    def write_rows() -> None:
        nonlocal written_rows
        if not rows:
            return
        frame = pd.DataFrame(rows).sort_values(["date", "symbol"])
        frame.to_csv(partial, mode="a", index=False, header=not partial.exists())
        written_rows += len(frame)
        rows.clear()

    with gzip.open(path, "rb") as f:
        while max_messages <= 0 or parsed < max_messages:
            length_bytes = f.read(2)
            if not length_bytes:
                break
            if len(length_bytes) != 2:
                raise EOFError("truncated ITCH message length")
            length = u(length_bytes)
            msg = f.read(length)
            if len(msg) != length:
                raise EOFError("truncated ITCH message body")
            parsed += 1
            typ = chr(msg[0]); types[typ] += 1
            if typ == "R" and length >= 19:
                symbols[u(msg[1:3])] = msg[11:19].decode("ascii").strip()
            elif (typ == "A" and length >= 36) or (typ == "F" and length >= 40):
                sid, oid = u(msg[1:3]), u(msg[11:19])
                ts, shares = u(msg[5:11]), u(msg[20:24])
                side = chr(msg[19])
                price = u(msg[36:40]) if typ == "F" else u(msg[32:36])
                if side not in ("B", "S"):
                    continue
                orders[oid] = (sid, side, price, shares)
                apply(sid, side, price, shares)
                c = counts[sid]; c["adds"] += 1; c["last_ns"] = ts
            elif typ in ("E", "C") and length >= 31:
                oid, ts, executed = u(msg[11:19]), u(msg[5:11]), u(msg[19:23])
                prior = orders.get(oid)
                if prior:
                    sid, side, price, remaining = prior
                    take = min(executed, remaining)
                    apply(sid, side, price, -take)
                    if remaining <= take: orders.pop(oid, None)
                    else: orders[oid] = (sid, side, price, remaining-take)
                    c = counts[sid]; c["executions"] += 1; c["exec_shares"] += take
                    c["buy_exec" if side == "B" else "sell_exec"] += take
                    c["last_ns"] = ts
            elif typ == "X" and length >= 23:
                oid, ts, cancelled = u(msg[11:19]), u(msg[5:11]), u(msg[19:23])
                prior = orders.get(oid)
                if prior:
                    sid, side, price, remaining = prior
                    cancel = min(cancelled, remaining)
                    apply(sid, side, price, -cancel)
                    if remaining <= cancel: orders.pop(oid, None)
                    else: orders[oid] = (sid, side, price, remaining-cancel)
                    c = counts[sid]; c["cancels"] += 1; c["last_ns"] = ts
            elif typ == "D" and length >= 19:
                oid, ts = u(msg[11:19]), u(msg[5:11])
                prior = orders.pop(oid, None)
                if prior:
                    sid, side, price, remaining = prior
                    apply(sid, side, price, -remaining)
                    c = counts[sid]; c["cancels"] += 1; c["last_ns"] = ts
            elif typ == "U" and length >= 35:
                old_id, new_id = u(msg[11:19]), u(msg[19:27])
                ts, shares = u(msg[5:11]), u(msg[27:31])
                # U retains stock and side from the original order. Price is [31:35].
                price = u(msg[31:35]); prior = orders.pop(old_id, None)
                if prior:
                    sid, side, old_price, remaining = prior
                    apply(sid, side, old_price, -remaining)
                    orders[new_id] = (sid, side, price, shares)
                    apply(sid, side, price, shares)
                    c = counts[sid]; c["adds"] += 1; c["cancels"] += 1; c["last_ns"] = ts
            if parsed % bucket == 0:
                flush(parsed)
            if progress_every > 0 and parsed % progress_every == 0:
                now = time.monotonic()
                print(f"parsed={parsed:,} messages snapshots_written={written_rows:,} "
                      f"open_orders={len(orders):,} elapsed={now-started:.1f}s "
                      f"rate={(parsed/max(now-started,1e-9)):.0f} msg/s", flush=True)
    if counts:
        flush(parsed)
    write_rows()
    if not partial.exists():
        pd.DataFrame(columns=["date", "symbol", "market", "asset_class", "open", "high", "low", "close", "volume"]).to_csv(partial,index=False)
    os.replace(partial, output)
    result={"messages": parsed, "message_types": dict(types), "stock_directories": len(symbols),
            "open_orders": len(orders), "snapshot_rows": written_rows, "output": str(output),
            "elapsed_seconds":round(time.monotonic()-started,3),
            "message_rate_per_second":round(parsed/max(time.monotonic()-started,1e-9),1)}
    output.with_suffix(output.suffix+".json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/external_sources/nasdaq/01302019.NASDAQ_ITCH50.gz"))
    p.add_argument("--output", type=Path, default=Path("data/external_sources/nasdaq/itch_snapshots.csv"))
    p.add_argument("--max-messages", type=int, default=500_000)
    p.add_argument("--bucket", type=int, default=10_000)
    p.add_argument("--progress-every", type=int, default=1_000_000)
    p.add_argument("--row-batch", type=int, default=100_000)
    p.add_argument("--resume", action="store_true", help="resume an existing .part by replaying the source to rebuild book state")
    a = p.parse_args()
    print(decode(a.input, a.output, a.max_messages, a.bucket,a.progress_every,a.row_batch,a.resume))


if __name__ == "__main__":
    main()
