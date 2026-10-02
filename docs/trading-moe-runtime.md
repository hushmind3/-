# TradingMoE · 원본 expert 확인 단계

## 현재 구성

14개 등록: forecast/event expert 8개(Kronos tokenizer는 Kronos에 포함)와 MacroHFT 원본 subagent 6개입니다. EarnHFT/EarnMore/DeepScalper/EIIE는 공식 source를 확보했지만, 확인한 공식 public 경로에 trained checkpoint가 없어 미등록입니다.

- 원본 위치: `C:\Users\hushm\Desktop\모델\heterogeneous-experts`
- 기존 `champion.pt`/`candidate.pt`, replay, paper account는 이 시스템에 연결하지 않습니다.
- 실제 측정: [frozen-expert-measurements.md](frozen-expert-measurements.md)
- 설계: [heterogeneous-experts-design.md](heterogeneous-experts-design.md)
- pinned 파일 hashes/parameters/dtype/source revision: [trading-moe-artifacts.json](trading-moe-artifacts.json)
- 원본 파일은 분리 보존합니다. 모델 합치기·증류·가지치기·학습을 수행하지 않습니다.

## 실행 순서

현재 설치된 격리 환경에서 아래 순서로 실행합니다. 독립 검증용 입력은 artifact root의 `verification/`에 있습니다. 실행 시 GPU를 점유하는 기존 agent와 겹치지 않게 운영합니다. 각 worker가 끝나고 GPU를 반환한 뒤 다음 expert가 시작됩니다.

```powershell
$expertRoot = 'C:\Users\hushm\Desktop\모델\heterogeneous-experts'
$expertPython = "$expertRoot\venv\Scripts\python.exe"
& $expertPython scripts/verify_frozen_experts.py --root $expertRoot --device cuda:0
& $expertPython scripts/audit_frozen_experts.py --root $expertRoot --report docs/frozen-expert-measurements.md
& $expertPython scripts/run_trading_moe.py --root $expertRoot --device cuda:0
# 선택된 2개 expert의 raw-only 추론과 residency 측정:
& $expertPython scripts/run_trading_moe.py --root $expertRoot --device cuda:0 --probe
```

검증 단계는 원본 network에 strict load(결정적 MarketGPT mask 제외)를 적용하고, 모든 parameters를 `requires_grad=False`로 고정합니다. 추론 전후 parameter mutation version과 유한 출력을 검사합니다. 등록만 하는 명령은 torch를 import하거나 model을 적재하지 않습니다.

`TradingMoE.infer(snapshot)`는 선택된 원본 출력을 각 expert 파일에 먼저 보존합니다. Raw-only가 기본값입니다. `adapters_enabled=True`는 구조적 feature 정렬만 활성화합니다. 공통 learned fusion/trading head에는 trained checkpoint가 없으므로 매매 가능한 출력은 차단합니다. HOLD/current weights는 readiness sentinel이며 학습된 판단이 아닙니다.

`save_manifest`는 각 independent 파일과 hash를 참조하는 상위 checkpoint manifest를 저장합니다. `from_manifest`는 원본 목록을 확인합니다. 각 실제 추론 전에 checkpoint 크기와 SHA256을 확인합니다. 완전히 자체 포함된 묶음 파일로 복사하는 작업은 수행하지 않았습니다.

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
- 원본 native output: `verification/{expert}.json`, 최근 wrapper 추론은 `inference/{expert}.json`
- raw API: `/api/experts/output?id=<registry ID>`; 임의의 파일 경로는 받지 않습니다.
- 카드는 처음 등록/제거될 때만 생성/삭제합니다. 반복 polling은 바뀐 text/state만 갱신합니다.
- loaded는 원본 전체 적재, active는 적재/추론 중, router 선택은 최근 선택 목록입니다. 선택되었지만 미적재인 것은 정상입니다.
- 실제 tensor device로 CPU/GPU residency를 기록합니다. CUDA current allocated와 reserved를 별도로 수집합니다. 화면의 VRAM은 해당 worker의 PyTorch tensor allocated입니다. CUDA driver/context·Windows·다른 앱의 물리 VRAM 전체 사용량은 아닙니다.
- RAM은 실행 중인 expert Python worker의 현재 RSS입니다. 과거 peak, 웹 프로세스나 종료된 worker를 현재 사용량으로 표시하지 않습니다.
- Windows venv launcher와 실제 Python child PID가 다른 경우 부모 관계를 확인한 뒤 실제 child의 메모리를 집계합니다. 종료·PID 재사용 시 stale loaded 상태를 해제합니다.
- Windows 파일 공유 충돌 처리는 기존 `state_io.atomic_json`의 직렬화/retry를 재사용합니다.
- 일별 초과수익률, 가격, ITCH logits, ETHUSDT Q값은 서로 단위가 다릅니다. 원본과 shape를 보존하고 평균하지 않습니다.

중복 능력 후보 판정은 아직 수행하지 않았습니다. 사용 기록과 동일 입력에 대한 출력 상관관계가 충분히 쌓인 이후의 작업입니다.

## 완료 검증

- 독립 CUDA inference: 14개 통과, optimizer/학습 0회.
- wrapper raw-only: TimesFM→Toto 순차 수행; raw shape `[2,1,10]`, `[9,1,2,1]`.
- 자동 residency trace: maximum concurrent expert 1, GPU residency 실제 관측, 완료 후 expert RAM/VRAM 0.
- CPU contract unit tests: 13개 통과.
- 기존 UI regression: 2,210 field 검사, 24개 independent mode action 검사, 67개 behavior 검사 통과. 13개 기존 action/filter 버튼 유지.
- 브라우저: 7개 화면, 14개 registry 카드, raw 조회, stable card node, desktop/mobile overflow 검사. 콘솔 오류 0, 실제 조작 POST 0.
- 기존 학습·보상·계좌 로직은 변경하지 않았습니다. 웹 controller만 API 반영을 위해 재시작했고 기존 agent/feed를 시작하지 않았습니다.
