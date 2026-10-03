"""Build an immutable artifact catalog and report from independent native probes.

No model imports or training. Every registered expert must have a finite frozen
native output from verify_frozen_experts.py. Hashes identify original artifacts.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stockrl.paths import expert_weight_path
import argparse
import hashlib
import json
from datetime import datetime, timezone


SPECS = [
    ("fincast", "FinCast 1.0B", "수치 시계열 · 분위수 예측", "FinCast", "FinCast-fts", "Apache-2.0"),
    ("exaone", "EXAONE Finance", "수치 시계열 · 장기 분위수 예측", "EXAONE-Forecast-for-Finance-1.0", "EXAONE-Forecast", "EXAONE research license"),
    ("kronos", "Kronos Base + Tokenizer", "OHLCV · 다음 봉 생성", "Kronos-base", "Kronos", "MIT"),
    ("marketgpt", "MarketGPT", "ITCH 주문 메시지 토큰 예측", "MarketGPT-100m", "MarketGPT", "MIT"),
    ("chronos", "FinText Chronos Global", "일별 초과수익률 · 확률 예측", "Chronos_Small_2023_Global", "TSFM_Finance", "Apache-2.0"),
    ("timesfm", "FinText TimesFM Global", "일별 초과수익률 · 분위수 예측", "TimesFM_20M_2023_Global", "TSFM_Finance", "Apache-2.0"),
    ("timemoe", "Time-MoE Large", "수치 시계열 · sparse MoE 예측", "TimeMoE-200M", "Time-MoE", "Apache-2.0"),
    ("toto", "Toto 2.0", "다변량 시계열 · 분위수 예측", "Toto-2.0-313m", "toto", "Apache-2.0"),
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def build(root):
    root = root.resolve()
    entries = []
    specs = SPECS + [(f"macrophft_{r}_{n}", f"MacroHFT {r}/{n}",
        "ETHUSDT · 현금/롱 Q 정책", None, "MacroHFT", "No LICENSE found")
        for r in ("slope", "vol") for n in (1, 2, 3)]
    for key, name, role, folder, source, license_name in specs:
        profile = json.loads((root / "verification" / (key + ".json")).read_text(encoding="utf-8"))
        if not profile["frozen"] or not profile.get("native_output"):
            raise ValueError(f"independent probe missing: {key}")
        files = []
        pinned = []
        folders = [folder] if folder else []
        if key == "kronos":
            folders.append("Kronos-Tokenizer-base")
        for directory in folders:
            metadata = json.loads((root / "checkpoints" / directory / "download.json").read_text(encoding="utf-8"))
            pinned.append({"model": metadata["model"], "revision": metadata["revision"]})
            for item in metadata["files"]:
                path = expert_weight_path(Path(item["path"]))
                actual = digest(path)
                if path.stat().st_size != item["bytes"] or actual != item["sha256"]:
                    raise ValueError(f"original artifact changed: {path}")
                if path.suffix in (".pth", ".pt", ".safetensors", ".bin", ".zip"):
                    files.append({"path":str(path.resolve()), "bytes":path.stat().st_size,
                        "sha256":actual, "archive":path.suffix == ".zip"})
        if key == "marketgpt":
            path = expert_weight_path(root / "checkpoints/MarketGPT-100m/ckpt_finetune_AAPL_v3.pt")
            files.append({"path":str(path.resolve()), "bytes":path.stat().st_size,
                "sha256":digest(path), "archive":False})
        if not folder:
            regime, label = key.removeprefix("macrophft_").split("_")
            path = expert_weight_path(root / f"sources/MacroHFT/result/low_level/ETHUSDT/best_model/{regime}/{label}/best_model.pkl")
            files.append({"path":str(path.resolve()), "bytes":path.stat().st_size,
                "sha256":digest(path), "archive":False})
        source_meta = root / "sources" / source / "SOURCE_REVISION.json"
        if not source_meta.exists():
            matching = [json.loads(p.read_text(encoding="utf-8")) for p in root.glob("*.tree.json")]
            saved = next(d for d in matching if d["repo"]["name"] == source)
            provenance = {"repo":saved["repo"]["full_name"], "revision":saved["tree"]["sha"]}
            source_meta.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        provenance = json.loads(source_meta.read_text(encoding="utf-8"))
        auxiliary = []
        if key == "timesfm":
            auxiliary.append(json.loads((root / "sources/TimesFM-legacy/SOURCE_REVISION.json").read_text(encoding="utf-8")))
        entries.append({"id":key, "backend":"macrophft" if not folder else key,
            "variant":f"{regime}/{label}" if not folder else None, "name":name, "role":role,
            "parameters":profile["parameters"], "weight_bytes":profile["parameter_bytes"],
            "dtype":"BF16" if profile["parameter_dtypes"] == ["torch.bfloat16"] else "FP32",
            "checkpoint_bytes":sum(f["bytes"] for f in files if not f["archive"]),
            "original_archive_bytes":sum(f["bytes"] for f in files if f["archive"]),
            "files":files, "pinned_models":pinned,
            "source":{"repo":provenance.get("repo"), "revision":provenance.get("revision")},
            "auxiliary_sources":auxiliary,
            "license":license_name, "verified":True, "frozen":True,
            "probe":{k:v for k,v in profile.items() if k != "native_output"}})
    return {"schema":"trading_moe_registry_v1", "generated_at":datetime.now(timezone.utc).isoformat(),
        "artifact_root":str(root), "experts":entries,
        "unavailable":[{"name":n, "reason":"No trained checkpoint found in audited official repository/readme/releases"}
            for n in ("EarnHFT", "EarnMore", "DeepScalper", "EIIE")],
        "training_performed":False, "weights_merged":False,
        "artifact_directory_bytes":sum(p.stat().st_size for p in root.rglob("*") if p.is_file())}


def report(catalog):
    rows = ["# Frozen expert 独立 추론 측정", "",
        "원본 가중치 고정, optimizer 없음. 배치 1~2종목, 과거 128시점, 미래 1시점의 CUDA 검사입니다.",
        "RAM은 각 독립 프로세스의 Windows peak working set(패키지/CPU 적재 포함), VRAM은 PyTorch peak allocated입니다. 동시에 적재한 합계가 아닙니다.", "",
        "| Expert | 실제 parameters | dtype | 원본 checkpoint MiB | 실질 가중치 MiB | RAM peak MiB | VRAM peak MiB | 입력 → 출력 | 추론 s |",
        "|---|---:|---|---:|---:|---:|---:|---|---:|"]
    for e in catalog["experts"]:
        p = e["probe"]
        rows.append(f'| {e["name"]} | {e["parameters"]:,} | {e["dtype"]} | {e["checkpoint_bytes"]/2**20:,.2f} | {e["weight_bytes"]/2**20:,.2f} | {p["peak_ram_bytes"]/2**20:,.2f} | {p["peak_allocated_bytes"]/2**20:,.2f} | {json.dumps(p["input_shapes"])} → {p["output_shape"]} | {p["forward_seconds"]:.3f} |')
    rows += ["", f'합계: {sum(e["parameters"] for e in catalog["experts"]):,} parameters; 실질 가중치 {sum(e["weight_bytes"] for e in catalog["experts"])/2**30:.3f} GiB; 펼친 원본 checkpoint {sum(e["checkpoint_bytes"] for e in catalog["experts"])/2**30:.3f} GiB.',
        f'모델·zip·공식 source·격리 Python 환경 등을 포함한 artifact 디렉터리 파일 크기 합계: {catalog["artifact_directory_bytes"]/2**30:.3f} GiB (filesystem 압축/공유 블록 사용량과는 다릅니다).',
        "", "## 검사 범위와 제한", "",
        "- MarketGPT는 실제 ITCH 체결 데이터가 없는 상태의 native vocabulary 합성 토큰 검사, MacroHFT는 합성 36+9 feature 검사입니다. 수익성 검증이 아닙니다.",
        "- 나머지는 원본 예제의 일별 초과수익률 또는 프로젝트의 실제 일봉 입력으로 형상·유한 출력·고정 가중치를 확인했습니다. 정확도 backtest가 아닙니다.",
        "- Kronos의 amount가 없는 입력은 missing 표시와 원본 허용 방식의 0값을 사용했습니다. 실제 매수금액이 관측되었다는 뜻이 아닙니다.",
        "- Kronos Base 실제 parameters 102,310,592 + tokenizer 3,958,042 = 106,268,634. Buffer/공유 tensor를 parameter에 중복 합산하지 않습니다.",
        "- MarketGPT 원본 pt는 6,291,444,790 bytes. 5,159,780,352 bytes의 결정적 causal mask와 optimizer 등이 포함됩니다. 원본 파일은 보존하고 inference에서 mask만 재생성하여 learned parameter 94,292,736개를 사용합니다. 원본 zip 1,054,913,634 bytes는 별도 보존합니다.",
        "- Toto는 native network AST와 strict checkpoint load를 유지하고 Lightning을 불필요하게 가져오는 GluonTS bridge import/class만 실행에서 제외합니다. 원본 source/checkpoint 파일은 수정하지 않습니다.",
        "- Time-MoE의 이름 200M은 활성 parameters 규모입니다. 실제 총 parameters는 453,196,800개입니다. BF16 원본을 그대로 사용합니다.",
        "- Toto의 runtime buffer도 존재합니다. 실질 가중치 memory와 실제 VRAM peak를 구분합니다. RAM/VRAM peak는 입력 크기와 package 환경에 따라 달라집니다.",
        "- MacroHFT는 공개된 ETHUSDT 하위 정책 6개만 확인되었습니다. 학습된 상위 hyper-agent checkpoint는 제공되지 않았습니다.",
        "- EarnHFT/EarnMore/DeepScalper/EIIE: 코드만 확보, 확인한 공식 public 경로에 trained checkpoint 없음. random 초기화를 pretrained policy로 표시하지 않습니다.",
        "- EXAONE은 연구 license, MacroHFT/EarnHFT는 확인한 저장소에 LICENSE 없음. EIIE 원본은 GPL-3.0의 legacy TensorFlow 구현입니다.",
        "", "## 통합 설계", "",
        "모든 expert를 independent frozen 모듈로 유지합니다. capability와 memory 예산으로 top-k를 선택하고 단일 GPU에서 순차 실행 후 worker를 종료하여 VRAM을 반환합니다.",
        "원본 출력 단위·분위수·시간축·symbols·as_of를 보존한 typed evidence → modality projection → cross attention → BUY/HOLD/SELL·비중·현금·가치 head 구조입니다. 예측 결과를 평균하거나 서로 다른 가중치를 합치지 않습니다.",
        "이번에는 학습하지 않으므로 공통 trading head/router는 학습된 정책이 아닙니다. head의 실행 가능한 매매 출력은 차단합니다. 실시간 Champion/Candidate·replay·paper account와 연결하지 않습니다.",
        "단일 상위 checkpoint는 pinned original artifacts + router/adapter/fusion 상태 + hash manifest로 구성할 수 있습니다. 현재는 manifest가 원본 파일을 참조하며 하나의 self-contained giant weight 파일로 복사하지 않습니다.",
        "능력 중복: Chronos/TimesFM은 동일 초과수익률 영역, FinCast/EXAONE/Time-MoE/Toto는 시계열 예측 영역이 겹칩니다. 현재 제거하지 않습니다. 향후 같은 입력의 사용 기록/출력 상관관계를 측정한 뒤에만 후보로 판정합니다.",
        "worker 시작마다 적재 비용이 발생하므로 표의 추론 시간은 cold-load 전체 시간이 아닙니다. serving 최적화 시 원본 능력을 검증한 뒤 dependency별 persistent worker를 고려합니다.", "", "## 공식 원본과 고정 revision", ""]
    for e in catalog["experts"]:
        sources = [f'[{p["model"]}](https://huggingface.co/{p["model"]}/tree/{p["revision"]})' for p in e["pinned_models"]]
        s = e["source"]
        sources.append(f'[{s["repo"]}](https://github.com/{s["repo"]}/tree/{s["revision"]})')
        for aux in e["auxiliary_sources"]:
            sources.append(f'[{aux["repo"]}](https://github.com/{aux["repo"]}/tree/{aux["revision"]})')
        rows.append(f'- {e["name"]}: ' + ", ".join(sources))
    rows.append("")
    return "\n".join(rows).replace("独立", "독립").replace("現금", "현금")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    catalog = build(args.root)
    (args.root / "expert_catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.root / "verification/independent_summary.json").write_text(json.dumps([
        {**e["probe"], "name":e["id"], "status":"ok"} for e in catalog["experts"]], indent=2), encoding="utf-8")
    args.report.write_text(report(catalog), encoding="utf-8")
    print(json.dumps({"experts":len(catalog["experts"]), "parameters":sum(e["parameters"] for e in catalog["experts"]),
        "weight_bytes":sum(e["weight_bytes"] for e in catalog["experts"]), "report":str(args.report)}, ensure_ascii=False))
