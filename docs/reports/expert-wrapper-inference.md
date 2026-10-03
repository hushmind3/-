# 초기 expert wrapper · 학습 없는 추론 검증

초기 14개 expert 진단 당시의 기록입니다. 현재 TradingMoE.pt 가상매매·학습 운용은 [실행 안내](../trading-moe-learning.md)를 따릅니다.

## 완료 범위

14개 원본 expert를 동결한 상태에서 실제 native inference → 원본 JSON 보관 → output adapter → 종목별 shared representation → CPU cross-attention fusion → 매매 출력까지 연결했습니다. 입력 adapter는 각 native schema의 시각·종목·단위를 확인하고 backend가 원본 모델의 preprocessing을 적용합니다. Router는 호환 입력과 가중치 예산을 보고 필요한 expert를 먼저 선택합니다. 보통 top-k=2이며 전체 검증에서는 14개를 **차례로** 실행합니다.

원본 출력은 변환 전에 worker가 쓴 JSON 바이트 그대로 `inference/runs/<run_id>/<expert>.json`에 보관합니다. 이전 실행 파일을 덮어쓰지 않습니다. 전체 evidence에도 같은 원본 숫자가 남으며 SHA256과 내용 일치를 자동 검사합니다. 기존 원본 checkpoint는 크기·SHA256을 검증한 뒤 읽기만 합니다.

Fusion/head는 기존 expert와 별도의 **미학습 진단용 checkpoint**입니다. 고정 seed의 per-expert projection, 64차원 공통 표현, cross-attention, actor/value/asset/cash head가 계산됩니다. 모든 parameter는 frozen입니다. 계산된 target weights, BUY/HOLD/SELL, 현금 비중, 종목 목록과 포지션 교체 필드가 나오지만 `executable=false`입니다. 아직 공동 매매 정책이 학습되었다는 뜻은 아닙니다. Optimizer·backward·학습·기존 계좌 실행은 연결하지 않았습니다.

입력에 없는 종목은 mask를 적용하고 기존 비중을 유지합니다. KRW와 USD의 비중은 독립적으로 계산합니다. 원본 MacroHFT의 Q값은 ETHUSDT에만 정렬하며 주식의 Q값으로 바꾸지 않습니다. 원본 logits/가격/초과수익률/Q값을 단순 평균하지 않습니다.

## 시간 측정

모든 단위는 초입니다. [14개 모델 측정 표](trading-moe-inference-measurements.md)와 [측정 JSON](trading-moe-inference-measurements.json)에 실제 수치가 있습니다. 대시보드는 같은 registry의 `last_timings`를 읽습니다.

| 항목 | 측정 범위 |
|---|---|
| 첫 적재 | worker 내부 패키지 import, CPU 모델 생성, checkpoint 읽기와 native 준비, CUDA context 준비. 가중치 GPU 전송 시간 제외 |
| GPU 전송 | 각 원본 모델의 CPU→GPU `to(device)`와 CUDA 완료 동기화. Kronos tokenizer 포함 |
| Forward | `inference_mode`에서 native API 계산, API 내부 입력 변환·출력 CPU 회수, 마지막 CUDA 동기화. registry 상태 쓰기 제외 |
| 전체 왕복 | 원본 파일 크기/hash 검사부터 worker 시작/종료, IPC, 원본 출력 파일 보관까지. 앞 세 항목 외의 overhead 포함 |

현재 worker는 환경 호환성과 메모리 반환을 위해 일회성 프로세스입니다. 이 표는 모두 새 프로세스/모델 적재 조건이며, Windows 파일 캐시는 이미 따뜻할 수 있습니다. Disk cache를 비운 물리적 첫 읽기나 GPU 상주 모델의 반복 추론 속도를 측정한 것은 아닙니다. Forward가 빠르더라도 전체 왕복이 더 길 수 있습니다.

공통 fusion은 CPU에서 계산하고 별도로 시간을 기록합니다. 처음 사용할 때 torch import 및 새 fusion 생성/보관도 이 구간에 포함됩니다. 전체 pipeline 시간은 expert 왕복 합계, adapter, CPU fusion, registry 쓰기를 포함하며 마지막 통합 JSON 직렬화/보관 직전까지 측정합니다.

## 검증 입력과 한계

2025-10-31 기준 실제 AAPL/MSFT 일별 초과수익률, 실제 AAPL 가격/OHLCV를 사용했습니다. Kronos 입력의 관측하지 않은 amount는 0이고 `amount_observed=false`로 표시했습니다. MarketGPT의 ITCH와 MacroHFT의 ETH 상태는 명시적으로 표기한 합성 schema fixture입니다. 실제 ITCH feed/ETH 기술 feature가 연결되었다거나 매매 성과가 검증되었다는 뜻은 아닙니다. 대시보드에도 입력 기준 시각과 일부 합성 입력이라는 표시를 둡니다.

GPU residency를 0.1초 간격으로 저장했고 maximum concurrent expert=1을 확인했습니다. 코드도 앞 worker가 종료된 후 다음 expert를 시작하며, 동일 artifact root의 OS lock이 wrapper 간 중복 실행을 차단합니다. 다른 애플리케이션/기존 live agent의 CUDA 작업까지 이 lock이 제어하지는 않습니다. 이번 검증 동안 기존 agent/feed는 정지 상태였습니다.

## 확인 방법

- 대시보드: `http://127.0.0.1:8766/#experts`
- 각 expert: 네 가지 시간, 원본 shape, 원본 JSON 상세와 다운로드.
- 통합 추론 카드: 완료 수, 전체 시간, CPU fusion 시간, 입력 날짜, 미학습/계좌 미실행 표시.
- 통합 상세 버튼: 종목·비중·현금·행동·value 출력과 fusion shape. 전체 원본 evidence/변환 feature는 통합 JSON 다운로드에 포함됩니다.
- 저장된 검증 결과 확인만 수행: `scripts/verify_trading_moe_integration.py --root <artifact root>`. GPU 모델을 다시 적재하지 않습니다.

CPU unit test 24개, 기존 UI field 2,210개/독립 mode action 24개/behavior 67개 검사, 실제 Edge 브라우저의 14개 원본 출력과 시간 56개 일치, 7개 화면, mobile overflow, stable DOM card 검사를 통과했습니다. 브라우저 콘솔 오류 0, 계좌/모드 POST 0입니다.
