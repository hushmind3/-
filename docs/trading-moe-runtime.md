# TradingMoE · 원본 보존 통합 추론

## 현재 구성

원본 모델 가중치는 `Desktop/모델/experts`에 보존합니다. 프로젝트에는 가중치를 두지 않습니다. 현재 저장 구조는 [project-storage.md](project-storage.md)를 따릅니다.

14개 등록: forecast/event expert 8개(Kronos tokenizer는 Kronos에 포함)와 MacroHFT 원본 subagent 6개입니다. EarnHFT/EarnMore/DeepScalper/EIIE는 공식 source를 확보했지만, 확인한 공식 public 경로에 trained checkpoint가 없어 미등록입니다.

- 소스·설정·데이터·실행 환경 위치: `C:\Users\hushm\OneDrive\문서\ChatGPT\금융매매모델\artifacts\experts`
- 기존 `champion.pt`/`candidate.pt`, replay, paper account는 이 시스템에 연결하지 않습니다.
- 실제 측정: [frozen-expert-measurements.md](frozen-expert-measurements.md)
- 설계: [heterogeneous-experts-design.md](heterogeneous-experts-design.md)
- pinned 파일 hashes/parameters/dtype/source revision: [trading-moe-artifacts.json](trading-moe-artifacts.json)
- 원본 파일은 분리 보존합니다. 모델 합치기·증류·가지치기·학습을 수행하지 않습니다.

## 실행 순서

현재 설치된 격리 환경에서 아래 순서로 실행합니다. 독립 검증용 입력은 artifact root의 `verification/`에 있습니다. 실행 시 GPU를 점유하는 기존 agent와 겹치지 않게 운영합니다. 각 worker가 끝나고 GPU를 반환한 뒤 다음 expert가 시작됩니다.

```powershell
$expertRoot = 'C:\Users\hushm\OneDrive\문서\ChatGPT\금융매매모델\artifacts\experts'
$expertPython = "$expertRoot\venv\Scripts\python.exe"
& $expertPython scripts/verify_frozen_experts.py --root $expertRoot --device cuda:0
& $expertPython scripts/audit_frozen_experts.py --root $expertRoot --report docs/frozen-expert-measurements.md
& $expertPython scripts/run_trading_moe.py --root $expertRoot --device cuda:0
# 선택된 2개 expert → adapter → CPU fusion → 진단용 매매 출력:
& $expertPython scripts/run_trading_moe.py --root $expertRoot --device cuda:0 --probe
# 14개 모두 순차 계산하는 통합 경로 검증 (평소 top-k=2와 구별):
& $expertPython scripts/run_trading_moe.py --root $expertRoot --device cuda:0 --probe-all
# 저장된 결과만 확인하여 측정 보고서 생성. GPU 연산 없음:
& $expertPython scripts/verify_trading_moe_integration.py --root $expertRoot
```

검증 단계는 원본 network에 strict load(결정적 MarketGPT mask 제외)를 적용하고, 모든 parameters를 `requires_grad=False`로 고정합니다. 추론 전후 parameter mutation version과 유한 출력을 검사합니다. 등록만 하는 명령은 torch를 import하거나 model을 적재하지 않습니다.

`TradingMoE.infer(snapshot)`는 worker의 원본 JSON 바이트를 실행별 파일에 먼저 보존합니다. Adapter가 종목별 feature와 유효 mask를 만들고, expert별 projection → 공통 64차원 표현 → cross-attention → actor/value/allocation/cash head가 실제 계산됩니다. Adapter는 서로 다른 단위의 출력을 평균하지 않습니다. 누락된 종목의 현재 비중은 유지하고 KRW/USD 계좌별 비중+현금 합계를 각각 1로 유지합니다.

현재 공통 fusion/head에는 학습된 checkpoint가 없습니다. 고정 seed로 만든 **미학습 진단용 가중치**를 CPU에서 frozen 상태로 실행합니다. `untrained_fusion_diagnostic`, `executable=false`이며 실제/가상계좌에 실행하지 않습니다. 고정 HOLD 결과를 통합 추론 결과로 가장하지 않습니다. 원본 evidence가 달라지면 공통 표현도 달라지는 것을 unit test로 확인합니다. Optimizer 생성, backward, 가중치 업데이트는 없습니다.

Router는 사용 가능한 입력과 비용 한도에 따른 deterministic top-k입니다. 학습된 선택기가 아닙니다. 기본값은 서로 다른 modality의 2개 expert이며 `--probe-all`은 검증 목적으로 14개를 순차 실행합니다. Artifact root의 OS file lock이 별도 wrapper 프로세스/registry 간 동시 실행도 차단합니다. 기존 live agent나 다른 GPU 프로그램까지 이 lock으로 제어하는 것은 아니므로 단독 실행 시 기존 agent와 GPU가 겹치지 않아야 합니다.

`save_manifest`는 원본 파일/hash, router 설정, 별도 fusion checkpoint를 참조합니다. `from_manifest`는 원본 목록을 확인하고 router 설정을 복원합니다. 각 실제 추론 전에 checkpoint 크기와 SHA256을 확인합니다. Fusion 가중치는 `fusion/<schema hash>.untrained.pt`에 원본과 별도로 보존합니다. 같은 schema는 동일 checkpoint를 strict-load합니다. 완전히 자체 포함된 묶음 파일로 복사하는 작업은 수행하지 않았습니다.

## Python 호환 환경

현재 Windows Python 3.13, system-site-packages의 Torch 2.14.0+cu132를 사용합니다. global 환경의 transformers/huggingface_hub/torchvision 불일치는 기존 runtime을 건드리지 않고 격리 환경에서 처리했습니다.

- `venv`: transformers 4.47.1, huggingface_hub 0.36.2, tokenizers 0.21.4, chronos-forecasting 2.2.2.
- `venv-toto`: native Toto 코드와 최신 global hub, dd-unit-scaling 0.1.0, unit-scaling 0.3.5, jaxtyping 0.3.11.
- FinText TimesFM은 새 TimesFM 2.5 loader를 사용하지 않습니다. pinned Google v1.2.6 decoder를 직접 로드하므로 불필요한 500M 모델을 먼저 불러오지 않습니다.
- Toto는 native model을 변경하지 않고 optional GluonTS/Lightning bridge의 import/class만 AST 실행에서 제외합니다. 원본 source 파일은 보존합니다.
- 이 버전 정보는 실제 설치 환경 기록이며 신규 컴퓨터의 모든 package 조합을 검증한 설치 lockfile은 아닙니다.

## 대시보드 연결

`http://127.0.0.1:8766/#experts`

프론트 `experts.js`는 `/api/experts`를 읽습니다. 모델 이름이나 parameter 수를 프론트에 하드코딩하지 않습니다. 현재 expert 화면에서는 전체 `/api/status`를 주기적으로 조립하지 않고 작은 registry API만 읽습니다. Raw output은 상세 버튼을 눌렀을 때만 별도 조회합니다.

- runtime registry: `runtime/trading_moe/registry.json`(generated, git 제외)
- 원본 native output: `verification/{expert}.json`, wrapper 실행별 원본은 `inference/runs/<run_id>/{expert}.json`; 과거 출력도 보존합니다.
- 전체 통합 결과: 같은 run 폴더의 `TradingMoE.json`, `/api/experts/fusion`에서 최근 결과 조회. 상세를 열고 버튼을 누를 때만 본문을 읽습니다.
- raw API: `/api/experts/output?id=<registry ID>`; 임의의 파일 경로는 받지 않습니다.
- 카드는 처음 등록/제거될 때만 생성/삭제합니다. 반복 polling은 바뀐 text/state만 갱신합니다.
- loaded는 원본 전체 적재, active는 적재/추론 중, router 선택은 최근 선택 목록입니다. 선택되었지만 미적재인 것은 정상입니다.
- 실제 tensor device로 CPU/GPU residency를 기록합니다. CUDA current allocated와 reserved를 별도로 수집합니다. 화면의 VRAM은 해당 worker의 PyTorch tensor allocated입니다. CUDA driver/context·Windows·다른 앱의 물리 VRAM 전체 사용량은 아닙니다.
- RAM은 실행 중인 expert Python worker의 현재 RSS입니다. 과거 peak, 웹 프로세스나 종료된 worker를 현재 사용량으로 표시하지 않습니다.
- Windows venv launcher와 실제 Python child PID가 다른 경우 부모 관계를 확인한 뒤 실제 child의 메모리를 집계합니다. 종료·PID 재사용 시 stale loaded 상태를 해제합니다.
- Windows 파일 공유 충돌 처리는 기존 `state_io.atomic_json`의 직렬화/retry를 재사용합니다.
- 일별 초과수익률, 가격, ITCH logits, ETHUSDT Q값은 서로 단위가 다릅니다. 원본과 shape를 보존하고 평균하지 않습니다.
- 각 카드에 첫 적재 / 가중치 GPU 전송 / Forward / 전체 왕복을 각각 표시합니다. 아직 측정하지 않은 값은 0초로 만들지 않고 미측정으로 표시합니다.
- 통합 카드에 진행 단계, 완료 expert 수, 전체 소요 시간, CPU fusion 시간, 입력 기준 시각과 합성 입력 여부를 표시합니다. 상태 요청은 추론이나 학습을 시작하지 않습니다.

중복 능력 후보 판정은 아직 수행하지 않았습니다. 사용 기록과 동일 입력에 대한 출력 상관관계가 충분히 쌓인 이후의 작업입니다.

## 검증

- 독립 CUDA inference: 14개 통과, optimizer/학습 0회.
- 14개 native worker → 원본 파일 → adapter → shared representation → CPU fusion → 매매 출력까지 실제 실행.
- 실제 최신 측정과 네 가지 시간의 정의: [trading-moe-inference.md](trading-moe-inference.md), [JSON 측정값](trading-moe-inference-measurements.json).
- 자동 residency trace: maximum concurrent expert 1, GPU residency 실제 관측, 완료 후 expert RAM/VRAM 0.
- CPU contract unit tests: 24개 통과. 원본 보존, 변경된 evidence 반영, missing mask, 통화별 비중, checkpoint 재현, owner lock, 경로 제한 포함.
- 기존 UI regression: 2,210 field 검사, 24개 independent mode action 검사, 67개 behavior 검사 통과. 13개 기존 action/filter 버튼 유지.
- 브라우저: 7개 화면, 14개 registry 카드, 56개 시간 값 일치, 원본 14개 조회, 통합 JSON 조회, stable card node, desktop/mobile overflow 검사. 콘솔 오류 0, 실제 조작 POST 0.
- 기존 학습·보상·계좌 로직은 변경하지 않았습니다. 웹 controller만 API 반영을 위해 재시작했고 기존 agent/feed를 시작하지 않았습니다.
