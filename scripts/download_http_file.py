"""Resumable streaming downloader used for public market research datasets."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "stockrl-research-data-fetch/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=90) as response:
        status = getattr(response, "status", 200)
        if offset and status != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        content_length = response.headers.get("Content-Length")
        expected = offset + int(content_length) if content_length else None
        received = offset
        last_report = time.monotonic()
        with partial.open(mode) as output:
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
                now = time.monotonic()
                if now - last_report >= 2:
                    if expected:
                        print(f"{received:,}/{expected:,} bytes ({received / expected:.1%})", flush=True)
                    else:
                        print(f"{received:,} bytes", flush=True)
                    last_report = now
            output.flush()
            os.fsync(output.fileno())
    if expected is not None and received != expected:
        raise RuntimeError(f"incomplete transfer: got {received}, expected {expected}; retry to resume")
    partial.replace(destination)
    print(f"saved {destination} ({received:,} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    download(args.url, args.destination)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted; partial download can be resumed", file=sys.stderr)
        raise
