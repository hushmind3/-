"""Install a completed market candidate byte-for-byte as the serving champion."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    report = json.loads((ROOT / "runtime-global-market-training/one-epoch-training-final.json")
                        .read_text(encoding="utf-8"))
    candidate = Path(report["candidate"])
    champion = Path(report["champion"])
    if report.get("status") != "completed_one_epoch":
        raise RuntimeError("candidate training is not complete")
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    if int(payload.get("next_offset", 0)) != 34455 or not payload.get("training_split_only"):
        raise RuntimeError("candidate is incomplete or lacks training-split metadata")
    if len(payload.get("symbol_map", {})) != int(payload["config"]["max_symbols"]):
        raise RuntimeError("candidate deterministic symbol map is incomplete")
    if not {"backbone.input_proj.weight", "context_policy.weight", "context_value.weight"}.issubset(
            payload["state_dict"]):
        raise RuntimeError("candidate is not the context-conditioned training checkpoint")

    before_sha = sha256(champion)
    backup = champion.with_name("champion.pre-context-path.pt")
    if backup.exists() and sha256(backup) != before_sha:
        backup = champion.with_name(f"champion.pre-context-path-{before_sha[:12]}.pt")
    if not backup.exists():
        shutil.copy2(champion, backup)
    elif sha256(backup) != before_sha:
        raise RuntimeError("existing backup does not match current champion; refusing overwrite")

    champion.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="champion-raw-candidate-", suffix=".pt", dir=champion.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copy2(candidate, tmp)
        if sha256(tmp) != sha256(candidate):
            raise RuntimeError("temporary champion does not byte-match the candidate")
        # Check the exact candidate state_dict before atomic replacement.
        copied = torch.load(tmp, map_location="cpu", weights_only=False)
        if copied["state_dict"].keys() != payload["state_dict"].keys():
            raise RuntimeError("copied state_dict keys differ from candidate")
        for key in payload["state_dict"]:
            if tuple(copied["state_dict"][key].shape) != tuple(payload["state_dict"][key].shape):
                raise RuntimeError(f"copied tensor shape differs at {key}")
        tmp.replace(champion)
    finally:
        tmp.unlink(missing_ok=True)

    metadata = {
        "promotion_mode": "byte-for-byte candidate copy; no weight conversion",
        "candidate": str(candidate), "candidate_sha256": sha256(candidate),
        "champion": str(champion), "champion_sha256": sha256(champion),
        "previous_champion_backup": str(backup), "previous_champion_sha256": before_sha,
        "state_dict_keys": len(payload["state_dict"]),
        "parameter_count": sum(t.numel() for t in payload["state_dict"].values()),
        "symbol_map_entries": len(payload["symbol_map"]),
        "context_features": payload.get("context_features"),
        "candidate_validation_mean_net_return": report["candidate_validation"]["mean_net_return"],
        "baseline_validation_mean_net_return": report["baseline_validation"]["mean_net_return"],
        "test_valid_decisions": report["test_result"]["candidate"]["valid_decisions"],
    }
    (champion.with_name("champion.itch-promotion.json")).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
