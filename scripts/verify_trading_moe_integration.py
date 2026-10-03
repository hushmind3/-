"""Read saved inference evidence, verify raw preservation, and write timing report.

Does not load a neural network, start GPU workers, train, or modify accounts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stockrl.expert_registry import read_registry, read_fusion_output, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("runtime/trading_moe/registry.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/reports/trading-moe-inference-measurements.json"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/reports/trading-moe-inference-measurements.md"))
    args = parser.parse_args()
    root = args.root.resolve()
    registry = read_registry(args.registry)
    document = json.loads(args.registry.read_text(encoding="utf-8"))
    result = read_fusion_output(args.registry)
    assert registry["pipeline"]["stage"] == "complete"
    assert result["training_performed"] is False and result["live_integration"] is False
    assert result["trading_output"]["executable"] is False
    assert result["fusion_output"]["trained"] is False
    assert result["adapter_status"] == "connected"
    selected = result["selected_experts"]
    assert len(selected) == 14 and len(set(selected)) == 14
    entries = {e["id"]:e for e in document["experts"]}
    evidence = {e["expert"]:e for e in result["evidence_tokens"]}
    adapted = {e["expert"]:e for e in result["shared_representation"]}
    rows = []
    for profile in result["profiles"]:
        entry = entries[profile["expert"]]
        original_path = Path(entry["raw_output_path"])
        assert original_path.resolve().is_relative_to(root)
        raw = original_path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        assert checksum == entry["raw_output_sha256"]
        packet = json.loads(raw)
        assert packet["native_output"] == evidence[entry["id"]]["native_output"]
        assert packet["frozen"] is True
        def finite_tree(value):
            return all(finite_tree(v) for v in value) if isinstance(value,list) else math.isfinite(value)
        assert finite_tree(packet["native_output"])
        assert packet["parameters"] == entry["parameters"]
        assert packet["as_of"] == result["as_of"]
        features = adapted[entry["id"]]
        assert features["source_symbols"] == packet["symbols"]
        assert features["units"] == packet["units"]
        for row in features["features"]:
            assert all(math.isfinite(v) for v in row)
        timing = profile["timings"]
        keys = ("cold_load_seconds", "gpu_transfer_seconds", "forward_seconds", "round_trip_seconds")
        assert all(math.isfinite(timing[k]) and timing[k] >= 0 for k in keys)
        assert timing["round_trip_seconds"] >= sum(timing[k] for k in keys[:3])
        assert timing == entry["last_timings"]
        rows.append({"id":entry["id"], "name":entry["name"], "parameters":entry["parameters"],
            "dtype":entry["dtype"], "input_authenticity":packet.get("input_authenticity"),
            "raw_shape":packet["output_shape"], "adapter_shape":features["shape"],
            "raw_sha256":checksum, **{k:timing[k] for k in keys}})
    trace = json.loads((root / "verification/residency_trace.json").read_text(encoding="utf-8"))
    maximum = max(len(s["active"]) for s in trace)
    assert maximum == 1
    assert any("GPU" in e["location"] for s in trace for e in s["active"])
    assert all(not e["loaded"] and not e["active"] for e in registry["experts"])
    checkpoint = registry["pipeline"]["fusion_checkpoint"]
    assert checkpoint["frozen"] and not checkpoint["trained"] and checkpoint["device"] == "cpu"
    assert hashlib.sha256((root / checkpoint["path"]).read_bytes()).hexdigest() == checkpoint["sha256"]
    report = {"run_id":result["run_id"], "as_of":result["as_of"],
        "verification":"inference_path_only_not_trading_quality",
        "input_provenance":"real daily stock data; synthetic ITCH and ETH policy fixtures",
        "experts":rows, "pipeline_timings":result["pipeline_timings"],
        "fusion_shapes":result["fusion_output"]["shapes"], "fusion_status":result["fusion_output"]["status"],
        "fusion_checkpoint":checkpoint, "max_concurrent_gpu_experts":maximum, "residency_samples":len(trace),
        "all_unloaded_after_inference":True, "raw_output_hashes_verified":14,
        "training_performed":False, "account_execution":False}
    atomic_json(args.report,report)
    lines = ["# 14개 expert · 통합 추론 실제 측정", "", f"Run: `{result['run_id']}` · 입력 기준: {result['as_of']}", "",
        "단위: 초. 매번 새 worker/모델 적재. OS 파일 cache는 따뜻할 수 있습니다.", "",
        "| Expert | 첫 적재 | GPU 전송 | Forward | 전체 왕복 |",
        "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append("| " + r["name"] + " | " + " | ".join(f"{r[k]:.3f}" for k in keys) + " |")
    lines += ["", f"전체 pipeline: **{report['pipeline_timings']['total_seconds']:.3f}초** · "
        f"adapter: {report['pipeline_timings']['adapter_seconds']:.3f}초 · CPU fusion: {report['pipeline_timings']['fusion_seconds']:.3f}초", "",
        f"GPU 동시 expert 최대 **1개** · residency {len(trace)}회 기록 · 완료 후 모든 expert 미적재.", "",
        "원본 JSON SHA256 및 내용 일치 14개 확인. Fusion/head는 미학습 frozen 진단용, 학습/optimizer/계좌 실행 없음.", "",
        "실제 일봉 주식 입력과 합성 ITCH/ETH schema fixture를 함께 사용한 경로 검증입니다. 거래 성과 검증이 아닙니다.", "",
        "[측정 범위·구현 설명](expert-wrapper-inference.md) · [기계 판독 원본](trading-moe-inference-measurements.json)", ""]
    args.markdown.parent.mkdir(parents=True,exist_ok=True)
    args.markdown.write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"verified_experts":14, "raw_hashes":14, "timing_fields":56,
        "max_concurrent_gpu_experts":maximum, "seconds":report["pipeline_timings"]["total_seconds"],
        "report":str(args.report)},ensure_ascii=False))


if __name__ == "__main__":
    main()
