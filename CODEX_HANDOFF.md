# StockRL 인수인계 요약

이 문서는 다음 Codex가 대화 전체를 다시 읽지 않고도 프로젝트 상태를 이어받기 위한 기록이다.

## 사용자 목표

- 약 5억 파라미터 Transformer 기반 글로벌 시장 트레이딩 에이전트
- BUY / HOLD / SELL과 가치평가 출력
- 수수료·슬리피지를 뺀 실제 순손익을 보상으로 사용
- 실시간 시장 관찰, 가상매매, replay 저장, candidate 학습, 검증 승격
- champion은 추론 담당, candidate는 별도 학습
- candidate가 미학습 검증 구간에서 champion보다 좋아질 때만 자동 승격
- teacher·기존 거래기록은 초기 지식이며 장기적으로는 자기 경험과 보상으로 독립
- 실제 주문은 기본 OFF. 키움 WebSocket은 시세 수집용으로 연결
- UI에서 사용자가 매번 명령하지 않고 시작/정지·관찰만/가상매매를 조작

## 현재 실행 위치

- Windows 실행본: `C:\Users\hushm\Desktop\StockRL-Portable-Backup`
- 웹 UI: `http://127.0.0.1:8765/`
- 런타임: `runtime-global-korea-live`
- 현재 모델: `runtime-global-korea-live/live/agent/champion.pt`
- candidate: `runtime-global-korea-live/live/agent/candidate.pt`
- replay: `runtime-global-korea-live/live/agent/replay.pt`
- API provider: Kiwoom real environment
- 실제 주문: OFF

## 모델과 온라인 학습

- 구조: 21-layer Transformer, temporal/cross-symbol attention, BUY/HOLD/SELL head, value head
- 파라미터 수: 약 512.7M
- RTX 3070 CUDA에서 실행
- replay 경험 하나는 시장상태·행동·후속 가격·수수료·슬리피지·순손익을 가진다.
- replay 경험이 256개 이상 누적될 때 candidate 학습을 시작한다. 고정 5분 타이머가 아니다.
- candidate 학습 중에도 champion 추론은 계속된다.
- candidate 검증 점수가 높을 때만 champion으로 교체하고 이전 champion을 백업한다.
- 현재 metrics에는 `candidate_training`, `updates`, `promotions`, `rejections`, `replay_count`, `last_update_utc`가 있다.

## 현재 시장 입력

설정 파일은 `configs/live_symbols_korea.json`이다.

- 한국 주식 80종목
- 미국 주식·지수 구성
- 글로벌 지수
- 지수·원자재 선물: ES=F, NQ=F, GC=F, SI=F, CL=F, HG=F
- 미국 국채 금리: ^TNX, ^FVX, ^IRX, ^TYX
- VIX, 환율, BTC/ETH
- KRX와 NXT Kiwoom 체결을 별도 통계로 표시

장외 상품도 마지막 관측값을 사용해 BUY/HOLD/SELL을 생성한다. UI에는 `현재 새 시세 없음`으로 오래된 가격임을 표시한다.

## 최근 수정 내용

- 웹 UI가 등록된 170개 종목 전체를 표로 표시
- 종목명·코드·가격·시세 시각·BUY/HOLD/SELL·확률·가치 표시
- 시장 필터에서 한국·미국·글로벌·선물·국채·환율·코인 선택 가능
- KRX와 NXT 수신 현황을 별도 카드로 표시
- candidate 적용/적용 안 함을 자동 검증 결과로 명확히 표시
- 실제 GPU 사용률과 VRAM을 nvidia-smi로 표시
- M5 Mac에서는 MPS와 통합 메모리로 표시
- 로그 HTTP polling access log는 숨김
- `global_transformer.py`가 장외 비주식 reference market을 최근 데이터 창에서 보존
- `global_online.py`가 마지막 관측값이 있는 장외 상품에도 판단 생성
- source mirror와 portable 실행본을 동기화

## Mac 이전

Mac용 폴더는 바탕화면의 `StockRL-Mac-Transfer`이다.

- M5 Apple Silicon: CUDA 사용 안 함, MPS 우선, CPU fallback
- CPU와 GPU가 통합 메모리를 공유하므로 별도 VRAM으로 해석하지 않는다.
- 실행 안내는 `README_MAC_TRANSFER.md`
- 실행 파일은 `scripts/start_mac.sh`, `scripts/start_mac.command`
- Mac에서 `python3 -m venv .venv`, `pip install -r requirements-mac.txt`, `./scripts/start_mac.sh`
- 웹 UI 주소는 `http://127.0.0.1:8765/`

## Kiwoom 키

- Windows 키움 자격 증명 보관함에서 real App Key, Secret, account를 읽어
  `runtime-global-korea-live/credentials_migration.json`에 넣었다.
- Mac용 `provider_credentials.py`가 이 파일을 자동으로 읽는다.
- 키 값은 채팅에 출력하지 않았다.
- 이 Mac 폴더를 다른 사람에게 공유하지 않는다.

## 주의할 점

- replay와 validation 파일이 매우 크다. Mac M5 통합 메모리 용량에 따라 candidate 학습은 RTX 3070보다 오래 걸릴 수 있다.
- 장시간 학습이 필요 없으면 UI에서 `관찰만`을 선택해 추론·수집은 유지하고 가상 포지션 반영만 멈춘다.
- API 키가 필요 없으면 공개 Yahoo/Kraken 입력으로도 실행할 수 있다.
- 실전 주문 adapter는 아직 OFF이며, 현재 Kiwoom 연결은 체결 시세 수집 중심이다.
- US 개별주식은 거래 시간과 무료 API 갱신 시각에 따라 판단이 없거나 오래된 시세로 표시될 수 있다.
- 기존 champion을 임의로 삭제하거나 덮어쓰지 않는다. 승격 전 백업 파일을 유지한다.
