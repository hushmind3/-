"""Fetch the CC-BY FI-2010 package through Fairdata's public download API."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import requests

from download_http_file import download

DATASET_ID = "73eb48d7-4dbc-4a10-a52a-da745b47a649"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "external_sources" / "fi2010" / "BenchmarkDatasets.zip"


def main() -> None:
    # Fairdata metadata is the authoritative dataset record; its automated
    # download-authorization endpoint currently reports downloads disabled.
    # Kaggle's public API redirects this same dataset to a time-limited file URL.
    session = requests.Session()
    response = session.get(
        "https://www.kaggle.com/api/v1/datasets/download/ulfricirons/fi-2010",
        stream=True,
        timeout=(30, 90),
        headers={"User-Agent": "stockrl-research-data-fetch/1.0"},
    )
    response.raise_for_status()
    if "googleapis.com" not in response.url and "kaggle" not in response.url:
        raise RuntimeError("Kaggle did not redirect to the expected public dataset storage")
    file_url = response.url
    expected_size = int(response.headers.get("Content-Length", "0"))
    response.close()
    print(f"Downloading public FI-2010 bundle ({expected_size:,} bytes)", flush=True)
    # Keep the expiring signed URL in memory and out of logs.
    download(file_url, OUTPUT)
    digest = hashlib.sha256()
    with OUTPUT.open("rb") as downloaded:
        for chunk in iter(lambda: downloaded.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    print(f"sha256={digest.hexdigest()}")


if __name__ == "__main__":
    main()
