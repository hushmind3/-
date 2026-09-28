"""Replay genuine historical market observations into the live append-only feed."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .live_feed import AppendOnlyMarketCSV


def replay_market_csv(source: str | Path, output: str | Path, bars: int = 24,
                      interval_seconds: float = 0.5, start_offset: int = 0,
                      max_cycles: int | None = None, stop_file: str | Path | None = None) -> dict:
    frame = pd.read_csv(source)
    required = {"date", "symbol", "market", "asset_class", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"mock source lacks columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    stamps = sorted(frame.date.unique())[-max(1, int(bars)):]
    start = max(0, int(start_offset))
    appender = AppendOnlyMarketCSV(output)
    cycles = added = count = 0
    try:
        for stamp in stamps[start:]:
            if stop_file and Path(stop_file).exists(): break
            group = frame.loc[frame.date == stamp]
            records = group.where(pd.notna(group), None).to_dict(orient="records")
            added += appender.append(records)
            cycles += 1; count += len(records)
            if max_cycles is not None and cycles >= max_cycles:
                break
            if interval_seconds > 0 and cycles < len(stamps[start:]):
                remaining=interval_seconds
                while remaining>0 and not (stop_file and Path(stop_file).exists()):
                    delay=min(.1,remaining); time.sleep(delay); remaining-=delay
    finally:
        appender.close()
    result = {"source": str(source), "output": str(output), "bars_requested": bars,
              "bars_written_or_seen": cycles, "source_rows": count, "appended_unique_rows": added,
              "finished_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    Path(output).with_name("mock_feed_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
