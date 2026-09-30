# Continual Trading Agent (StockRL)

시장 경험과 비용 차감 순손익으로 스스로 매매 정책을 발전시키는 지속학습형 자율 트레이딩 에이전트입니다.

## 프로젝트 목표

특정 trader나 고정 전략을 영구 모방하지 않습니다. 시장·종목·시간축·포트폴리오 상태를 입력으로 받아 종목 선택, BUY/HOLD/SELL, 자금 배분, 포지션 유지·조절·교체·청산을 경험으로 배웁니다. Teacher, 공개 모델, 과거 trader 기록은 초기 금융 문법을 익히는 자료이며 최종 정책이 아닙니다.

수수료, 세금, 스프레드, 슬리피지, 현금, 보유수량, 평단, 실현·평가손익, 유동성, 자본 규모는 거래 환경이 계산합니다. 실제 매매 정책은 가상매매 경험과 순손익으로 개선합니다.

## 운영 구조

`시장 관찰 → Champion 판단과 가상체결 → 결과·비용 반영 → 성숙 경험을 제한된 replay에 저장 → Candidate batch 학습 → 고정된 Candidate snapshot과 Champion을 같은 미래 구간에서 paper 비교 → 나을 때만 승급`

- **Champion:** 현재 기준 모델이며 운영 시장을 관찰하고 Champion paper 계좌의 판단·가상체결을 담당합니다.
- **Candidate:** Champion에서 시작해 replay 경험으로 계속 학습합니다. 학습 중 실시간 추론은 Candidate 관찰용 별도 paper 계좌에서 수행하며, 이 계좌는 학습 보상이나 승급 점수에 포함되지 않습니다.
- **승급 검증:** 학습 시점의 Candidate를 RAM에 고정하고, Candidate와 Champion을 별도 paper 계좌로 같은 미래 bar·초기 자금·비용 조건에서 비교합니다. 현재 구현은 128개 미래 bar를 사용합니다. 64개 미만의 비교는 승급할 수 없습니다. 비교 중 Champion 가중치가 바뀌면 그 검증은 무효입니다. Candidate의 순손익률이 Champion보다 높을 때만 승급합니다.
- **실제 주문:** 기본 OFF입니다. 시세 provider가 연결되거나 키움 인증이 성공해도 실제 주문이 켜지는 것은 아닙니다. 사용자가 명시적으로 허용하기 전에는 주문을 내지 않습니다.
- **학습과 관찰:** 추론과 Candidate 학습은 별도 작업으로 진행해 시장 관찰을 막지 않도록 합니다. 실제 진행 여부는 실행 중 API와 운영 화면에서 확인합니다.

## 모델 구성

기본 글로벌 Transformer 설정은 hidden size 1,408, attention head 16개(각 88차원), FFN 5,632, Transformer block 21개, 최대 시간축 128개, 종목 특징 17개, 시장 맥락 특징 16개입니다. 종목 수 제한과 시간축 길이는 서로 다릅니다. 출력은 종목별 SELL/HOLD/BUY 점수와 가치, 포트폴리오 자금 배분입니다. 체크포인트마다 실제 파라미터 수가 다를 수 있으므로 운영 화면의 API 값이 현재 로드 모델의 수입니다.

운영 런처는 Candidate 학습에서 batch 8, optimizer update 8회를 지정하며 새 경험 누적 간격은 16건입니다. 표본 중복 여부와 실제 완료량은 API의 학습 기록으로 확인합니다. 장치와 학습 설정은 실행 환경·코드에 따라 달라질 수 있으므로 과거 벤치마크를 현재 성능으로 간주하지 않습니다.

## 실행

Windows에서 Python 3.10 이상과 CUDA 사용 시 호환되는 PyTorch가 필요합니다.

```powershell
python -m pip install -e .
```

프로젝트 루트에서 `StockRL Start.bat`을 실행하면 로컬 운영 서버와 브라우저를 시작합니다. 운영 주소는 `http://127.0.0.1:8766/`이며 상태 API는 `http://127.0.0.1:8766/api/status`입니다. 런처는 이미 정상 서버가 있으면 재사용하고, 8766이 다른 프로세스에 점유되어 있으면 기존 프로세스를 임의로 바꾸지 않고 종료합니다.

운영 화면은 시장 수신, Champion·Candidate 판단, 양쪽 paper 계좌, replay, 학습 update·optimizer 횟수, 검증 진행·점수, 승급·기각, GPU·VRAM, 실제 주문 OFF 여부를 표시합니다. API가 응답하지 않으면 현재 운영 상태를 확인한 것으로 보지 않습니다. 이 README는 실시간 상태판이 아닙니다.

## 파일과 자격 증명

- 소스 저장소: 이 프로젝트 폴더. 저장소는 공개이며 코드와 문서를 보관합니다.
- 모델 가중치: 바탕화면 `모델` 폴더의 `champion.pt`, `candidate.pt`만 지속 보관합니다. 가중치 파일은 Git에서 제외합니다. 검증용 Candidate snapshot은 RAM에만 둡니다.
- 실행 데이터: 프로젝트 안 `runtime/markets/<market>/live`에 둡니다. 한국 기본 경로는 `runtime/markets/korea/live`입니다. replay, 계좌, cursor, 시세 cache, 로그, 판단 기록, 검증 상태는 모델 파일이 아니며 Git에 올리지 않습니다.
- 런타임 경로: 실행기는 프로젝트 바깥 runtime 경로를 거부해야 합니다. 프로젝트 폴더 밖에 새 runtime 디렉터리를 만들지 않습니다.
- 인증 정보: 키움 App Key, Secret, 계좌 정보는 운영체제 자격 증명 보관함에 저장합니다. 공개 저장소의 코드·설정·문서에 실제 키나 비밀번호를 넣지 않습니다. `configs/local/provider_settings.json`에는 provider 선택 등 비밀이 아닌 설정만 둡니다.
- Git 규칙: `.gitignore`는 runtime 디렉터리와 `.pt`, `.pth`, `.ckpt`, `.safetensors`를 제외합니다. 과거 Git 커밋에 runtime 기록이 남을 수 있지만 새 커밋에 runtime snapshot을 추가하지 않습니다.

## 데이터와 실험 코드

운영 feed 설정은 `configs/live_symbols_korea.json`을 사용합니다. 실시간 provider의 종목 범위·갱신 주기·장애는 실제 운영 환경에 따라 달라지며, 공개 시세가 거래소 전체 호가·체결 원장을 뜻하지 않습니다. ITCH, 공개 teacher, 오프라인 학습·증류·benchmark 코드는 `scripts/`와 `src/stockrl/market_training.py`에 있는 연구 도구입니다. 이 도구들의 결과나 checkpoint를 운영 Champion으로 자동 적용하지 않습니다.

## 현재 상태 확인 방법

실행 후 `GET /api/status`에서 `updates`, `last_update_utc`, `candidate_training`, `candidate_learning_enabled`, `candidate_stage`, `candidate_skip_reason`, replay 건수, Candidate 표본·optimizer 횟수, 검증 bar·점수, 승급·기각 수, agent/feed 진행 상태를 확인합니다. 프로세스가 살아 있는지만으로 판단·학습이 진행 중이라고 결론 내리지 않습니다.

현재 운영 상태, 학습 횟수, 점수, 파일 크기, GPU 사용량은 실행 중인 API에서 확인하십시오. 과거 측정치나 저장된 지표는 그 기록 시점의 값입니다.

## 라이선스와 안전

이 프로젝트는 연구·엔지니어링 도구이며 수익을 보장하거나 투자 조언을 제공하지 않습니다. 기본 운용은 paper trading입니다. 실주문은 사용자의 명시적인 설정 없이는 실행되지 않습니다.