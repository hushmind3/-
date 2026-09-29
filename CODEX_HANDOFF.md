# StockRL 인수인계 요약

이 문서는 다음 Codex가 대화 전체를 다시 읽지 않고도 프로젝트 상태를 이어받기 위한 기록이다.

## 사용자 목표

- 약 5억 파라미터 Transformer 기반 글로벌 시장 트레이딩 에이전트
- BUY / HOLD / SELL과 가치평가 출력
- 수수료·슬리피지를 뺀 실제 순손익을 보상으로 사용
- 실시간 시장 관찰, 가상매매, 메모리 replay, candidate 학습, 검증 승격
- champion은 추론 담당, candidate는 별도 학습
- candidate가 미학습 검증 구간에서 champion보다 좋아질 때만 자동 승격
- teacher·기존 거래기록은 초기 지식이며 장기적으로는 자기 경험과 보상으로 독립
- 실제 주문은 기본 OFF. 키움 WebSocket은 시세 수집용으로 연결
- UI에서 사용자가 매번 명령하지 않고 시작/정지·관찰만/가상매매를 조작

## 현재 실행 위치

- Windows 소스 프로젝트: `C:\Users\hushm\OneDrive\문서\ChatGPT\금융매매모델`
- Portable backup: 보존된 이전 실행본이며 현재 기본 소스 경로가 아니다.
- 웹 UI: `http://127.0.0.1:8766/`
- 런타임: 프로젝트의 `runtime/markets/korea` 아래. live 자료는 `runtime/markets/korea/live`
- champion: `C:\Users\hushm\Desktop\모델\champion.pt`
- candidate: `C:\Users\hushm\Desktop\모델\candidate.pt`
- replay는 runtime의 bounded SQLite, 계좌와 검증 상태는 JSON으로 보존한다. 미성숙 pending 경험은 메모리에 있어 재시작 전에 주의가 필요하다.
- API provider: Kiwoom real environment
- 실제 주문: OFF

## 모델과 온라인 학습

- 구조: 21-layer Transformer, temporal/cross-symbol attention, BUY/HOLD/SELL head, value head
- 파라미터 수: 약 512.7M
- RTX 3070 CUDA에서 실행
- replay 경험 하나는 시장상태·행동·후속 가격·수수료·슬리피지·순손익을 가진다.
- 현재 웹 실행기는 새 replay 경험이 4,096개 이상 쌓일 때 candidate 학습을 시작한다. 설정은 batch 1, update 1이다. 재시작하면 메모리 replay와 미완료 판단은 사라지고, 새 candidate는 champion에서 다시 시작한다.
- candidate 학습 중에도 champion 추론은 계속된다.
- 현재 승급 점수는 paper-account를 순차 실행한 최종 순자산이 아니라 수익률 합산 근사치다. 실제 paper-account 순손익 비교는 아직 구현·검증되지 않았다.
- champion은 정지 시 다시 저장하지 않는다. promotion gate만 champion을 교체한다.
- 사용자가 보호한 champion SHA256 `2D0D…37797A`와 현재 파일 SHA256 `F0B1…D7725F02`가 다르다. 계보를 확인할 때까지 64-bar 결과와 무관하게 promotion을 보류한다.
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
- 웹 UI 주소는 `http://127.0.0.1:8766/`

## Kiwoom 키

- Windows 키움 자격 증명은 OS 자격 증명 보관함에 둔다. 설정 메타데이터는 `configs/local/provider_settings.json`에 있다.
- 키 값은 프로젝트 폴더나 runtime 파일에 복사하지 않는다.
- 키 값은 채팅에 출력하지 않았다.
- 이 Mac 폴더를 다른 사람에게 공유하지 않는다.

## 주의할 점

- replay, pending, validation은 메모리에서만 처리한다. candidate update에 실제 사용한 replay 항목은 제거하고, update에 못 쓴 항목은 다음 시도까지 남긴다.
- 검증 경험은 최근 64개 시각까지만 유지하며 candidate와 champion은 같은 검증 자료로 비교한다.
- candidate가 champion을 이기지 못하거나 값이 잘못되면 `candidate.pt`를 champion 복사본으로 되돌린다.
- 모델 폴더에는 `champion.pt`, `candidate.pt`만 둔다.
- pending 경험은 판단 당시 가격을 함께 보관해 rolling CSV가 짧아져도 이어서 평가한다. 30일 동안 해당 종목의 다음 봉이 오지 않으면 만료한다.
- feed cache는 64MB 기준으로 최근 512개 시각까지만 줄이고, 오래된 참고시장 봉은 종목당 20개만 보존한다. 중복 방지 색인은 최근 8일이다.
- 장시간 학습이 필요 없으면 UI에서 `관찰만`을 선택해 추론·수집은 유지하고 가상 포지션 반영만 멈춘다.
- API 키가 필요 없으면 공개 Yahoo/Kraken 입력으로도 실행할 수 있다.
- 실전 주문 adapter는 아직 OFF이며, 현재 Kiwoom 연결은 체결 시세 수집 중심이다.
- US 개별주식은 거래 시간과 무료 API 갱신 시각에 따라 판단이 없거나 오래된 시세로 표시될 수 있다.
- champion은 candidate가 같은 미학습 검증 구간에서 이길 때만 교체한다. 정지 시 checkpoint를 다시 저장하지 않는다.
- 2026-09-29 현재 8766은 agent=true, paper=false, observe=false, validation 60/64, promotions 0, actual orders OFF로 일시 정지했다. runtime 이전과 승급 hold 적용 전까지 관찰/paper를 다시 켜지 않는다.


## 기존 실행 상태 기록 — 2026-09-29 KST

이 항목이 위의 과거 실행 상태 기록보다 최신이다.

- 8766에서 system/feed/agent가 실행 중이다. Kiwoom은 실시간 시세 수신용이다.
- paper와 관찰은 켜져 있고 실제 주문은 꺼져 있다.
- side-monitor 최신 확인에서 순차 paper validation은 12/64, replay는 4,096건, 승급은 0회였다.
- 보호 champion SHA와 현재 champion SHA가 다르다. 보호 장치가 candidate 학습과 승급을 막고 있다.
- 당시 runtime은 프로젝트의 runtime-global-korea-live를 사용했다. 아래 디렉터리 이전 후에는 이 경로가 아니다.
- champion과 candidate 파일은 수정하지 않았다. champion SHA256은 F0B1759A30262C81C957CCBA555048AC0C4B993587D795F30C59BE96D7725F02다.
- 기존 %LOCALAPPDATA%\StockRL\runtime-global-korea-live 복사본은 남아 있다. 현재 실행에서 사용하지 않지만, 삭제 시도는 도구 정책에 거부되어 미완료다.

## 현재 디렉터리 구조 — 2026-09-29 KST

- 시장별 runtime 기준 경로는 `runtime/markets/<market>/`이다. 한국은 `runtime/markets/korea/live/`, 나스닥은 별도 운영을 추가할 때 `runtime/markets/nasdaq/live/`를 사용한다.
- `STOCKRL_MARKET`으로 시장 이름을 지정하고, Windows 시작 파일과 한국 실행기는 `korea`를 기본값으로 둔다. `run_global_paper.ps1`은 `-Market` 값을 받을 수 있다.
- 기존 프로젝트 runtime의 `runtime-global-korea-live`는 `runtime/markets/korea`로 이동했다. SQLite/계좌/replay 파일을 삭제하거나 모델 checkpoint를 수정하지 않았다.
- runtime 이동 직전 validation은 64/64였으나 보호 SHA 불일치로 승급 0회, candidate 학습 OFF, 실제 주문 OFF였다.
- 이후 새 구조로 8766을 재시작했다. 아래 최신 확인 상태를 따른다. 기존 %LOCALAPPDATA%\StockRL\runtime-global-korea-live 삭제 요청은 별도 정책 차단으로 아직 미완료다.

## 재구성 후 운영 확인 — 2026-09-29 KST

- 새 runtime 기준 경로는 `runtime/markets/korea`; 실제 live 상태는 `runtime/markets/korea/live`다. 22개 파일, 46,029,050 bytes가 이동됐고 replay.sqlite3와 market.csv.sqlite3 모두 `integrity_check=ok`, runtime 안 `.pt` 없음이다.
- 8766 listener PID 31776, feed PID 17576, agent PID 8572다. API와 프로세스 command line의 feed/state 경로가 모두 `runtime/markets/korea/live`를 가리킨다.
- system/feed/agent/paper/observe는 켜져 있고 실제 주문은 OFF다. `candidate_stage=promotion_held`, validation bars=0, replay=4096, promotions=0, candidate_learning=false. champion lineage blocker가 유지된다. 이동 직전의 64/64 완료 기록과 재시작 뒤 bars=0은 서로 다른 시점이다.
- champion SHA256은 `F0B1759A30262C81C957CCBA555048AC0C4B993587D795F30C59BE96D7725F02`로 유지됐다. 8767 listener는 없다.

## GitHub runtime 백업 정책 정정 — 2026-09-29 KST

- 사용자는 모델 가중치만 제외하고 프로젝트 파일 전체를 비공개 GitHub에 보관하도록 지시했다. 따라서 현재 runtime snapshot도 추적 대상이다.
- 직전 커밋 `ad6238e`는 실수로 `/runtime/` 전체를 Git에서 제외했다. 후속 커밋에서 이 규칙을 해제하고 현재 `runtime/markets/korea` snapshot을 추가한다.
- `.pt`, `.pth`, `.ckpt`, `.safetensors`는 계속 제외한다. 모델 파일과 대시보드 HTML은 수정하지 않는다.
