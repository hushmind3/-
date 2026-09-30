# 사이드 검토 인수인계 목록

> 이 문서는 과거 시간순 검토 기록이며 현재 지침이 아닙니다. 현재 규칙은 루트 `AGENTS.md`, 현재 프로젝트 설명은 `README.md`, 실시간 상태는 실행 중인 `GET /api/status` 응답을 따릅니다. 문서에 남은 SHA 잠금, GitHub 비공개 설정, runtime 백업·경로 주장은 과거 기록입니다.

최종 갱신: 2026-09-29 KST

사이드 검토는 메인 작업이 멈추거나 결과를 보고한 뒤 우선순위 순서로 확인한다. 각 항목은 `수정함`, `재현 안 됨`, `보류`, `중복` 중 하나로 응답하고 근거를 기록한다.

## P1 — 런타임 판단 정체 및 소스 버전 차이

- **응답: 재현 안 됨 (현재 판단 정체) / 보류 (프로세스와 소스 버전 차이).**
- **현재 확인:** 8766 PID 30896, feed와 agent 모두 실행 중, 실제 주문 OFF, 가상매매 ON. 12:17 KST 상태에서 `last_market_timestamp=2026-09-29T03:17:00Z`, metrics의 판단 수는 105,886이었다. feed·decisions·metrics 파일도 12:15 KST 무렵 갱신됐다. 이 확인 시점에는 feed만 진행되고 판단은 멈춘 상태가 재현되지 않았다.
- **남은 사실:** agent PID 7068은 11:40:53 KST에 시작했고 `global_online.py`는 11:58 KST에 수정됐다. 실행 중인 프로세스가 수정 전 코드를 메모리에 올렸을 가능성이 있다. 현재 판단이 갱신된다는 사실만으로 프로세스가 최신 코드라는 뜻은 아니다.
- **보류 이유/재확인 조건:** 실행 프로세스가 최신 소스를 사용하도록 하려면 재시작이 필요할 수 있지만, 미성숙 경험은 현재 메모리에만 있어 재시작 시 잃을 수 있다. 먼저 미성숙 경험 저장·복구를 마련하고, 그 뒤 프로세스 버전과 heartbeat를 대조한다. 현재 실행 프로세스는 중단하지 않았다.

## P1 — 재시작 시 미성숙 경험 유실

- **응답: 보류.**
- **근거:** `GlobalLearner.follow_csv()`에서 `pending`과 `portfolio_pending`은 함수 내부 메모리 목록으로 시작한다. 재시작 시 cursor와 paper account는 읽지만 이 두 목록을 복구하지 않는다.
- **범위:** 이는 결과가 아직 성숙하지 않은 학습 경험과 해당 판단 입력의 유실 가능성이다. paper account에 저장된 미결 주문 유실과는 다르다. 실제 재시작으로 유실된 양을 확인한 것은 아니며, 정적 코드 구조에서 가능성을 확인했다.
- **보류 이유/재확인 조건:** 라이브 프로세스의 미성숙 경험을 안전하게 옮기거나 잃지 않도록 SQLite 저장·복구와 판단 시각/종목 기준 중복 방지를 설계해야 한다. 구현 후 재시작 복구 경로를 확인하기 전에는 의도적으로 프로세스를 재시작하지 않는다.

## P1 — candidate 학습 표본 수

- **기존 항목 응답: 보류. 신규 제안 응답: 중복 (이 항목과 연결).**
- **근거:** 최신 8766 GET에서 `candidate_batch_size=1`, `candidate_optimizer_steps_target=1`, `candidate_samples_target=1`, `candidate_every=4096`, `replay_current=3975/4096`을 확인했다.
- **보류 이유/재확인 조건:** 학습량을 임의로 올리지 않는다. champion을 고정한 뒤 동일한 미학습 구간의 순손익과 학습 시간을 비교할 수 있는 조건을 갖추면 작은 단계로 평가한다.

## P1 — 성공한 학습 표본을 replay에서 삭제하던 문제

- **응답: 수정함.**
- **변경:** `src/stockrl/global_online.py`의 candidate 학습 완료 경로에서 사용한 경험을 `self.replay.discard(consumed)`로 지우던 호출을 제거했다. bounded replay의 용량 정리만 남긴다.
- **확인:** 현재 source의 candidate 경로에는 discard 호출이 없고, replay 삭제 메서드 정의만 남아 있다. 현재 API는 `replay_persistence=bounded_sqlite_runtime`, SQLite capacity 4,096을 보고했다.
- **미확인:** 실행 중 agent는 source 수정 전에 시작했으므로 이 수정이 현재 프로세스에 적용되지 않았다. 최근 API에서 replay 4,039/4,096, candidate threshold 4,096, paper ON, 실제 주문 OFF였고, 다음 candidate 학습 뒤 item/SQLite 행 수가 유지되는지는 아직 확인하지 않았다.
- **최신 런타임 후속 확인:** candidate가 `sequential_paper_validation`에 진입해 이후 2/64 validation bar까지 진행했다. `paper_examples_trained=2`, `replay_examples_discarded=2`였고 같은 읽기 전용 확인에서 SQLite `experiences`는 4,096행이었다. feed가 새 경험을 추가할 수 있으므로 4,096행만으로 학습에 쓴 행이 보존됐다고 입증되지는 않는다. 12:30 KST 재확인에서도 agent PID는 패치 전 시작한 상태였다.
- **후속 보고 응답: 중복.** replay 행/sequence ID 대조와 재시작 전 경험 보존 요구는 이 미확인 항목과 연결한다. 다음 검토는 discard 카운터 변화 및 실제 sampled row ID를 대조한다.
- **재확인 조건:** pending 경험을 먼저 영속화하고 안전하게 복구한 뒤, 최신 소스로 실행하여 학습 전후 replay item 수와 SQLite row 수를 대조한다.

## P1 — 포지션 목표 비중 조절

- **기존 항목 응답: 보류. 신규 제안 응답: 중복 (이 항목과 연결).**
- **근거:** `paper_account.py:192-247`에서 allocation은 BUY 신호가 난 미보유 종목의 신규 예산에만 적용된다. 보유 종목은 SELL일 때만 매도 예약을 만들며, BUY 신호로 추가 매수하지 않고 일부 축소도 지원하지 않는다.
- **보류 이유/재확인 조건:** 목표 비중과 주문량 규칙은 비용·회전율·평가 결과를 바꾸므로, paper-account의 목표 비중 정의와 동일 조건 비교 절차를 먼저 마련한다.

## P1 — portfolio 보상 중복 및 종목 귀속

- **응답: 중복 (기존 P1 portfolio 보상 중복 기록 항목과 연결).**
- **근거:** 기존 검토에서는 같은 시점의 계좌 전체 손익이 여러 종목 경험에 반복 연결되는 경로를 지적했다. 이번 검토의 “거래 대상이 아닌 종목에도 portfolio 보상 연결”은 같은 보상 귀속 경로의 더 좁은 사례다.
- **상태:** 아직 수정하지 않았다. 계좌 단위 보상을 하나의 transition으로 기록할지, 거래 가능한 종목별 기여 손익으로 분해할지 검증 후 결정한다.

## P1 — 개별 SELL 보상이 공매도를 가정함

- **응답: 보류 (소스에서 재현됨).**
- **근거:** `global_online.py:271-280`은 `position=action-1`로 SELL에 -1 노출을 주고 하락 수익률에서 양의 보상을 만든다. `previous_position` 인자는 계산에 쓰이지 않는다. `_mature()`는 이 값을 계산해 `source="paper"` 경험으로 replay에 넣고, 경험 기본 `reward_version`은 `net_trade_v2`다. `follow_csv()`는 일반 pending 경험을 paper enabled 여부와 관계없이 만들며, paper가 켜져 있으면 별도의 계좌 경험도 추가한다. `paper_account.py:_fill()`은 보유 포지션이 없을 때 SELL을 무시한다.
- **영향:** 동일 replay에 isolated short proxy와 cash-only 계좌 결과가 섞일 수 있다. 관찰 전용에서도 short 형태 경험을 만들 수 있다.
- **후속 보고 응답: 중복.** 보상 의미가 정리되기 전에는 이 candidate 학습을 온전한 cash-only 정책 학습 증거로 해석하지 않는다. 순차 검증 자체는 paper 계좌의 비용 차감 손익 비교지만, 검증 완료/승급도 아직 확인되지 않았다.
- **보류 이유/재확인 조건:** 두 경로를 하나로 정리할지, cash-only 보유 전이로 보상을 다시 정의할지 아직 확정하지 않는다. `previous_position`은 실제 가상계좌 보유 상태가 아니라 이전 모델 행동으로 관리되므로 그대로 소유 상태로 취급할 수 없다. SELL 시 현금/보유, BUY 시 현금/보유 각각의 상승·하락 보상을 paper 체결 의미와 대조하고, replay reward version 분리 및 동일 cash-only 검증 기준을 마련한 뒤 수정한다.

## P2 — API의 승급 상태 필드가 현재 검증 단계와 혼동됨

- **기존 항목 응답: 보류 (필드 의미 분리 필요). 후속 보고 응답: 중복 (이 항목과 연결).**
- **근거:** 8766의 최신 GET에서 `candidate_stage=waiting`, `replay_current=3656/4096`, `holdout_timestamps_current=0/64`인데 `promotion_blocked_reason="sequential paper-account validation is not implemented"` 및 `promotion_gate_ready=false`가 반환됐다. `candidate_gate_history`의 해당 결과 시각은 2026-09-28T19:57:26Z다. 현재 소스에는 그 문구가 없고 sequential paper-account validation 코드가 있다.
- **관련 원인:** agent PID 7068은 11:40:53 KST 시작, `global_online.py`는 11:58 KST 수정이다. 실행 중인 프로세스가 이전 모듈을 메모리에 올렸을 가능성과 오래된 metrics 값이 함께 확인됐다. 현재 단계는 candidate 시작 전 대기이므로 과거 미구현 차단 문구가 현재 gate 상태처럼 보인다.
- **범위 정정:** 이 값은 마지막 후보 비교의 과거 거부 사유가 현재 상태처럼 API에 노출된 것이다. 현재 candidate는 대기 중이며, 다음 후보의 승급이 영구적으로 막혔다고 확인된 것은 아니다. 현재 source는 새 candidate 검증을 시작할 때 이 사유를 초기화한다.
- **최신 런타임 후속 확인:** 12:30 KST에는 `candidate_validation_bars=2/64`, updates=2, queue 0이었다. 실제 주문 OFF, outer/metrics paper ON, feed·agent 기록 갱신 중으로 earlier agent-stall/mode mismatch는 이 재확인 시점에 재현되지 않았다. `promotion_gate_ready=true`와 과거 차단 사유 문구의 혼선 및 이전 점수/거부 기록은 남아 있다. 검증 완료나 승급은 확인되지 않았다.
- **후속 보고 응답: 중복.** 최신 관찰도 같은 2/64 진행 상태와 gate 필드 혼선을 가리킨다. 64개 bar 완료와 성공 승급 확인 전에는 champion 개선으로 간주하지 않는다.
- **필드 의미 정밀화:** 현재 source의 `_write_metrics()`는 `promotion_gate_ready`를 `not bool(promotion_blocked_reason)`로 계산한다. 즉 “차단 사유 문자열 없음”을 뜻하지만 검증 완료/승급 자격처럼 읽힐 수 있다. 최신 GET에서는 bars=7/64, metrics gate_ready=true, metrics reason=null이었고 API의 learning blocker에는 예전 fallback 문구가 채워졌다. 승급 0회, `last_candidate_promoted=false`라 완료/승급은 아니다. 이 추가 P2 보고는 같은 상태 표시 혼선 항목에 중복 연결한다.
- **보류 이유/재확인 조건:** 실제 검증 진행률, validation complete/eligible, 명시적 blocker를 별도 상태로 분리한다. API semantics를 정리하고 대시보드 targeted edit가 필요해지면 dashboard preservation rule의 백업·원본 확인 요건을 먼저 충족한다. 64 bars와 비교 완료 전 승급 및 champion 교체를 허용하지 않는다.
- **보류 이유/재확인 조건:** 오래된 결과와 현재 gate 상태가 구분되어 보이도록 API/UI 상태 산정 및 실행 코드 세대를 함께 점검한다. 미성숙 경험이 메모리에만 있는 상태에서 재시작하면 학습 경험이 유실될 수 있어 운영 프로세스를 건드리지 않았다. 경험 저장·복구 후 최신 코드를 올리고, 64개 미학습 bar 진행 및 동일 paper 계좌 비교 결과와 API gate를 대조한다.

## P2 — 비거래 자산의 학습 보상 마스크

- **응답: 중복 (P1 portfolio 보상 귀속 항목에 포함).**
- **상태:** P1 보상 귀속 문제를 해결할 때 paper 계좌에서 실제 거래 가능한 자산과 문맥 전용 자산을 함께 구분한다.

## P2 — 미관측 종목 방향이 현재 판단처럼 표시됨

- **응답: 보류 (소스에서 경로 확인, 실제 렌더링은 미확인).**
- **근거:** `GlobalMarketPanel`은 가격·feature를 forward-fill하면서 관측 mask는 false로 둔다. 추론 뒤 미관측 token 표현은 mask로 0이 되지만, `follow_csv()`는 과거 관측이 있는 종목이면 현재 bar 관측 여부와 관계없이 decisions row를 추가한다. 주문/replay는 현재 관측일 때만 만든다. `web_app.py`의 `model_directions` 생성은 freshness 필터 없이 종목별 마지막 action을 노출한다.
- **영향:** 휴장 등으로 입력이 없는 종목도 최근 action을 현재 판단처럼 보여줄 수 있다. 그 값은 해당 시점의 유효한 최신 입력에 근거한 판단이 아닐 수 있다. 미관측 종목의 replay/체결은 이 경로에서 생성하지 않는 것으로 확인했다.
- **보류 이유/재확인 조건:** API에서 freshness를 분리하거나 마지막 판단 시각/상태를 표시하는 방식이 필요하다. 제공된 dashboard preservation rule은 원본 HTML이 복구되지 않았을 때 dashboard를 수정하지 못하게 하므로 화면 HTML을 건드리지 않는다. 실제 화면 렌더링은 이번 확인에서 검증하지 않았다.

## P3 — 명시적 시간축 ID가 학습·추론 호출에 연결되지 않음

- **응답: 보류.**
- **근거:** `time_scale_ids`는 transformer/wrapper의 선택 인자지만 `src/stockrl` 검색에서 `None`이 아닌 값을 넘기는 train/inference 호출은 확인되지 않았다. 운영 설정은 한 번에 하나의 horizon을 선택한다.
- **영향:** 모델에 명시적인 bar cadence 식별값이 전달되지 않고, 한 실행에서 단기·중기·장기 scale을 함께 비교하는 구조도 확인되지 않았다.
- **보류 이유/재확인 조건:** 현재 모델 파일이나 호출 경로를 임의 변경하지 않는다. horizon/time-scale 입력 규칙을 먼저 정하고 train/live에서 동일 ID와 feature cadence를 보장한 뒤, 시간축별 동일 미학습 paper 구간 결과를 비교한다.

## P2 — 시장·판단 로그 보관 한도

- **기존 항목 응답: 보류. 신규 P2 제안 응답: 중복 (이 항목과 연결).**
- **근거:** 검토 당시 market/decision CSV는 각각 약 7.6MB/9.3MB였고, 이번 확인에서는 약 7.9MB/11.3MB였다. 확인한 경로에는 보관 회전 한도가 드러나지 않았다.
- **보류 이유/재확인 조건:** 현재 크기만으로 삭제·회전을 적용하지 않는다. 미성숙 경험 복구와 cursor 연속성을 보장하고, 보존 기간 및 최대 크기를 정한 뒤 날짜별 보관과 실시간 파일 회전을 설계한다.

## 운영 안전 상태

- 확인 당시 8766은 실행 중이고 실제 주문 플래그는 `false`, paper trading은 `true`였다.
- champion/candidate checkpoint는 이 검토에서 수정하지 않았다.
- 대시보드는 이 검토에서 수정하지 않았다.

## 후속 작업 순서 제안 (아직 실행하지 않음)

1. 미성숙 경험을 보존·복구할 수 있게 만든 뒤 현재 agent의 경험을 안전하게 보존한다.
2. 그 다음에 최신 소스를 실행 프로세스에 반영하고 PID와 source version을 확인한다. replay 사례 삭제 방지가 실제 프로세스에 적용됐는지도 확인한다.
3. candidate가 replay 4,096건과 미학습 검증 bar 64개를 채우고, 동일 paper-account 조건의 비교가 작동하는지 확인한다. 과거 gate 결과와 현재 대기 상태는 분리해 표시한다.
4. 보상 귀속과 기존 포지션 목표 비중 항목은 미해결 상태로 두고, 동일 미학습 구간 비교 조건을 정해 후속 처리한다.
5. 확인 과정 내내 실제 주문 OFF와 champion 보호를 유지한다. 두 checkpoint 이외의 `.pt` 생성 경로와 로그 증가 제한도 별도로 확인한다.
## P0 — 실주문 경로 안전 조건

### 미관측/오래된 signal이 dispatcher에 도달할 수 있음

- **응답: 보류 (위험 경로 확인, 현재 실주문 경로 비활성).**
- **근거:** `follow_csv()`는 현재 bar 미관측이어도 과거 관측이 있는 종목 action을 `decisions.csv`에 쓸 수 있다. `_dispatch_live_decisions()`는 action·symbol 지원·수량만 검사하고 signal 시각/quote freshness/세션을 검사하지 않는다.
- **현재 상태:** 최신 8766 API는 실제 주문 OFF. `STOCKRL_LIVE_BROKER_ADAPTER` 미설정, 별도 desktop Python 프로세스 없음. API의 Kiwoom real provider 표시는 시세 provider 상태이며 live order adapter가 구성됐다는 뜻이 아니다.
- **보류 이유/재확인 조건:** 실주문을 켜기 전 observed=true와 허용 시세 나이를 보장하는 주문 전 quote/session 검사 경로가 필요하다. 이 조건이 없으면 adapter activation을 허용하지 않는다. 실제 주문은 발생하지 않았다.

### 목표 포지션/중복 주문 관리가 없음

- **응답: 보류 (dispatcher 위험 확인, 현재 실주문 경로 비활성).**
- **근거:** dispatcher는 각 BUY/SELL decision row마다 adapter의 `size_order(symbol, side)`를 호출한다. app 쪽에서 실제 보유량, 미체결 주문, 목표 수량, 직전 제출 여부를 대조하지 않는다.
- **보류 이유/재확인 조건:** adapter contract가 현재 계좌 수량·미체결 수량을 읽고 target quantity와 차이만 한 번 주문하도록 정의돼야 한다. 지원되지 않는 adapter는 live activation을 거부한다. actual orders OFF와 adapter 미설정 상태를 유지한다.

### 주문 실패 뒤 대기 주문이 이어질 수 있음

- **응답: 수정함.**
- **변경:** `src/stockrl/broker.py`의 `BrokerWorker`에서 주문 예외 발생 시 stop flag 설정, 대기 queue 폐기, adapter emergency stop 호출, worker loop 중단을 추가했다. 중단 뒤 `submit()`은 새 요청을 거부한다.
- **확인:** 변경 후 소스 흐름을 읽어 실패 branch가 queue를 비우고 `break`하는 것을 확인했다. mock adapter 동작 검증은 아직 하지 않았다. 실제 broker는 호출하지 않았다.
## 최신 런타임 후속 확인 — candidate 학습/검증 진행

- **응답: 중복** (기존 P1 replay 보존 및 SELL reward 의미 항목에 연결).
- 8766 GET: live/feed/agent 실행 중, paper ON, 실제 주문 OFF, candidate `sequential_paper_validation` 27/64 bars, replay capacity/count 4,096, trained 2, `replay_examples_discarded=2`, promotions 0, last promoted false, blocker null, last_error null.
- 64 bars 검증과 승급은 아직 끝나지 않았다. 실행 중 agent는 replay discard 수정 전 소스로 시작했다. 전체 replay가 capacity에 찬 사실만으로 이전에 사용한 두 sample row가 보존됐다고 볼 수 없다.
- SELL short-proxy와 cash-only paper 계좌 보상 불일치를 해결하기 전에는 이 candidate 결과를 cash-only 정책의 개선 증거라고 주장하지 않는다. 주문 worker 실패 차단은 소스 수정됐으나 mock 검증 전이며, live adapter 경로는 안전 요건 확인 전 계속 비활성으로 둔다.

## 최신 런타임 후속 확인 — 검증 진행 및 학습 표본 출처

- **진행량 보고 응답: 중복 (기존 순차 검증 진행 항목에 연결).** 새 읽기 전용 확인에서 `candidate_stage=sequential_paper_validation`, `candidate_validation_bars=50/64`, replay 4096, `paper_examples_trained=2`, 승격 0회, `last_candidate_promoted=false`, 실주문 OFF였다. 이전 27/64 및 47/64 보고는 이 snapshot으로 대체한다. `promotion_gate_ready=true`는 64개 bar 완료나 승격을 뜻하지 않는다.
- **학습 오염 재확인 응답: 보류 (표본 출처를 지표에서 판별할 수 없음).** active agent PID 7068은 2026-09-29 11:40:53 KST 시작했고 `global_online.py` 현재 파일은 12:44:44 KST 수정된 것으로 확인됐다. runtime 지표는 사용 경험 2, 직전 update 표본 1, optimizer step 1, 누적 discard 2를 기록한다. 현재 소스는 `paper_account`의 `net_trade_v2` 경험과 teacher만 학습하며 성공 update 뒤 replay 표본을 버리지 않는다. 실행 중 프로세스는 수정 전 모듈을 메모리에 적재했을 수 있다. SQLite에는 현재 4096행이 있지만, 이것만으로 discard된 두 표본의 원래 출처나 재보존을 증명하지 못한다. 이번 읽기 전용 확인에서는 pickle payload를 실행·역직렬화하지 않았으며, 샘플별 provenance도 runtime 지표에 없다.
- **평가 영향 판단:** 현재 검증 ledger는 `start_after=2026-09-29T03:27:00Z` 이후의 순차 bar로 candidate/champion을 각각 paper account에서 평가 중이고 50/64 bar다. 따라서 학습 reward 불일치는 candidate 학습 근거의 한계지만, 현재 순차 paper-account net-return 평가 자체가 오염됐다는 증거는 확인되지 않았다. 다만 완료 전 결과로 승급을 판단할 수 없고, 이 candidate를 수정된 학습 코드의 결과라고 표시할 수도 없다. 64개 완료 및 비교 결과, active source generation을 다시 대조한다.

## 운영 지침 — champion 계보 확인 전 승급 보류

- **응답: 수정함.** 사용자의 지침에 따라 보호 SHA256 `2D0D…37797A`와 현재 champion SHA256 `F0B1…D7725F02`의 계보가 확인될 때까지, 64/64가 되어도 candidate를 champion으로 승급하지 않는다.
- **변경:** `src/stockrl/global_online.py`는 비교 기준 champion SHA가 보호 SHA와 다르면 candidate를 유지한 채 `promotion_held`로 전환하고 추가 candidate 학습/승급을 막는다. 그 전에는 8766의 `observe=false`, `paper=false`로 구버전 agent의 60/64 검증 진행을 멈췄다.
- **현재 적용 상태:** API는 `agent=true`, `paper=false`, `observe=false`, validation 60/64, `promotions=0`, `last_candidate_promoted=false`, `real_orders=false`를 반환했다. 실행 중인 PID 7068은 guard 이전에 시작한 구버전이므로 source/runtime 경로 이전과 함께 새 코드로 재시작하기 전까지 API의 `promotion_blocked_reason=null`은 보호가 적용됐다는 증거가 아니다. 관찰/paper를 다시 켜지 않는다.
- **재확인 조건:** runtime을 프로젝트 폴더로 이전한 뒤 새 agent를 시작하고, 64/64 이후에도 `candidate_stage=promotion_held`, 명시적 blocker, 승격 0회를 확인한다. SHA 계보를 확인하고 보호 기준을 정하기 전에는 checkpoint를 수정하거나 교체하지 않는다.

## Runtime 위치 이전 — 과거 중간 상태 (이후 시장별 경로로 대체)

- **응답: 수정 진행 중.** 사용자가 기존 LocalAppData runtime을 프로젝트 폴더로 옮기고, 성공 확인 뒤 기존 외부 복사본을 정리하라고 지시했다. 프로젝트 밖에는 새 폴더를 만들지 않는다.
- 실행기 기본 경로와 `STOCKRL_RUNTIME_DIR` 검증을 프로젝트 내부 `runtime-global-korea-live`로 통일했다. 현재 8766의 agent는 아직 외부 runtime 경로의 old process다.
- migration 전 API 상태: feed/agent 실행 중, `paper=false`, `observe=false`, 실제 주문 OFF. runtime DB/CSV/JSON 파일은 agent/feed를 정지한 뒤 복사·무결성을 확인하고, 새 launcher가 프로젝트 경로를 쓰는 것을 확인한 다음에만 기존 복사본을 지운다.
- SQLite replay는 파일로 보존되지만, 현재 agent 메모리의 미성숙 경험 목록은 runtime 폴더 안에 있지 않아 이전할 수 없다. 정지 시 이 미성숙 경험이 남아 있으면 소실될 수 있으며, 이를 막는 persistence 기능은 후속 작업이다.


## 최신 인수인계 응답 — 2026-09-29 KST

### P0: 운영/API 상태와 candidate 안전 제한

- **응답: 수정함.** champion 보호 SHA 불일치 guard가 적용되어 있고, API에서 blocker와 candidate 학습 비활성화를 확인했다.
- **현재 확인:** 8766에서 feed/agent/paper/observe가 켜져 있고 실제 주문은 꺼져 있다. side-monitor 최신 확인은 validation 12/64, replay 4,096, 승급 0회, candidate_learning=false, 보호 SHA blocker 유지다. Kiwoom은 실시간 시세 provider이며 실주문 활성화를 뜻하지 않는다.

### validation 연속성

- **응답: 수정함.** runtime 이전 뒤 기존 60/64 validation 세대는 이어지지 않았다. 새 generation이 feed cursor 이후 미학습 구간에서 시작했다. 이전 결과가 보존·연속됐다고 주장하지 않는다.

### runtime 이전과 기존 폴더

- **응답: 수정함.** runtime 22개 파일, 43,960,537 bytes를 프로젝트의 runtime-global-korea-live로 복사했고 복사 당시 SHA256이 일치했다. 두 SQLite DB는 integrity_check=ok였고 복사본에 .pt는 없었다. 현재 8766의 feed 출력 경로는 프로젝트 runtime이다.
- **기존 LocalAppData 삭제: 보류.** 기존 %LOCALAPPDATA%\StockRL\runtime-global-korea-live는 현재 프로세스가 쓰지 않지만 남아 있다. 삭제 요청은 도구 정책에 거부됐다. 우회 삭제는 하지 않았다.

### GitHub runtime snapshot과 추적 — 과거 중간 정책 (이후 정정)

- **응답: 수정함.** 기존 커밋 613c330에는 runtime 21개 파일, 43,897,551 bytes가 포함됐다. 각 파일은 100MB 미만이며 .pt는 없다. 현재 로컬 runtime은 실행 중 갱신되어 44,139,072 bytes다.
- 기존 커밋과 이력은 되돌리거나 다시 쓰지 않는다. 로컬 runtime 파일은 그대로 두고, Git 추적에서 제외해 향후 변경분이 자동으로 커밋되지 않게 했다. 이전 snapshot은 과거 커밋에 남는다.
- 서로 다른 side-review 제안은 이 방식으로 처리한다. 기존 업로드 이력은 유지하면서, 계속 바뀌는 runtime은 이후 Git 저장에서 제외한다.

### 추가 항목 분류

- **새 .pt 생성: 재현 안 됨.** 확인한 runtime 폴더에는 .pt가 없고 모델 checkpoint는 건드리지 않았다.
- **대시보드 수정: 재현 안 됨.** dashboard HTML을 수정하지 않았다.
- **8767 실행: 재현 안 됨.** 확인 시점에 8767 listener가 없었다.
- **최신 side-monitor 상태: 중복.** 12/64 진행, 승급 0회, candidate 학습 차단, SHA blocker 유지는 위 P0 상태 항목에 연결한다。



### GitHub 이력에 남은 runtime snapshot — 과거 side-monitor 시점 확인

- **응답: 중복.** 현재 HEAD에서 runtime 추적이 빠졌고 로컬 파일은 보존됐다는 항목과 연결한다. 이전 커밋 613c330에는 21개 runtime 파일(43,897,551 bytes)이 남아 있다. 따라서 현재 main 트리에서 제외된 것이며 Git 이력 전체에서 제거된 것은 아니다.
- **처리:** 커밋 이력을 다시 쓰거나 이전 snapshot을 지우지 않았다. 전체 이력에서 제거할지는 영향 범위를 검토한 뒤 별도 결정이 필요하다.

## 시장별 runtime 디렉터리 변경 및 최신 확인 — 2026-09-29 KST

### 시장별 runtime 경로

- **응답: 수정함.** 기존 프로젝트 runtime을 `runtime/markets/korea`로 이동했다. runtime API 기준 경로는 `runtime/markets/<market>/`; 실제 live 상태는 그 안의 `live/`다. NASDAQ용 경로는 `runtime/markets/nasdaq/live/`가 된다.
- `STOCKRL_MARKET`은 `korea` 또는 `nasdaq` 같은 안전한 시장 이름을 받아 경로를 나눈다. 외부 경로 차단은 유지된다. `run_global_paper.ps1`은 `-Market`을 지원하고 현재 한국 launcher들은 korea를 지정한다.
- 최초 이동 기록 당시 runtime은 Git 추적에서 제외된 상태였다. 그 정책은 사용자의 “모델 가중치만 제외하고 비공개 GitHub에 보관” 지시에 따라 후속 정정됐다. runtime snapshot은 `da5dd24`에 처음 포함됐고 `c98ef4c`에서 온라인 SQLite 백업으로 갱신됐다. 이동 당시 파일 수·크기 및 SQLite 무결성 결과는 그 시점의 기록이다.

### side-review 항목별 응답

- **P1 생성 파일 추적 — 수정함:** `__pycache__/*.pyc` 18개와 `src/stockrl_distill.egg-info/*` 5개를 Git index에서 제외했다. `.gitignore` 규칙을 유지·보강했다. 로컬 생성물은 남아 있다.
- **P1 GitHub runtime 이력 설명 — 중복:** 현재 HEAD는 runtime을 추적하지 않지만 기존 `613c330` commit history에는 21-file snapshot이 있다. README와 앞선 인수인계 항목에 구분해 기록했다.
- **P2 전체 소스 폴더 정리 — 보류:** `src`, `configs`, `data`, `scripts`, `docs`, `backups`는 유지한다. 41개 script를 이동하기 전에는 용도 index와 상대경로 참조 조사가 필요하므로 이번 runtime 변경에 포함하지 않았다.
- **P2 diagnostics/backups 정리 — 보류:** dashboard 백업 5개 및 diagnostics 자료는 참조·복구 가치를 확인하기 전에는 지우거나 옮기지 않는다.
- **정지 후 재기동 상태 — 수정함:** 정지 당시 paper/observe 설정은 true였으나 시스템은 멈춰 있었다. 현재는 새 경로로 8766/feed/agent/paper/observe가 다시 ON이다.

### 새 경로에서 확인한 최신 상태

- 8766 listener PID 31776, feed PID 17576, agent PID 8572. process command line과 API feed output은 `runtime/markets/korea/live`를 가리킨다.
- API: system/feed/agent/paper/observe ON, real_orders=false, `candidate_stage=promotion_held`, bars=0, replay=4096, promotions=0, candidate_learning=false, protected-SHA blocker 유지.
- 이동 직전 검증은 64/64였지만 승급은 차단되어 0회였다. 재시작 후 API bars=0을 별도 현재 시점으로 기록한다.
- **Relaunch report response: 수정함.** server/feed/agent runtime path와 promotion blocker를 새 프로세스에서 다시 확인했다. 8767 listener는 없다.

## GitHub runtime 정책 검토 정정 — 2026-09-29 KST

- **응답: 수정함.** 사용자의 지시는 모델 가중치만 제외하고 프로젝트 전체를 비공개 GitHub에 보관하는 것이다. 이전의 “runtime은 현재 Git에서 제외”라는 분류는 이 지시와 충돌했다.
- 직전 커밋 `ad6238e`가 `/runtime/` 전체를 제외한 것을 확인했다. README와 `.gitignore`를 정정하고, 프로젝트 안의 현재 한국 runtime snapshot을 다음 커밋에 포함한다.
- runtime 파일을 지우거나 서버를 중지하지 않는다. `.pt`, `.pth`, `.ckpt`, `.safetensors` 가중치는 계속 제외한다.
- 별도 검토에서 지적된 `web --runtime`, `STOCKRL_LOCAL_CONFIG_DIR`, PowerShell `Data`/`State` 외부 경로 제한 누락은 코드에서 확인했다. 이번 GitHub 백업과 별도 경로 안전성 수정으로 남긴다.

## P1 이전 AppData runtime 미삭제 — 2026-09-29 KST

- **응답: 보류.** `%LOCALAPPDATA%\StockRL\runtime-global-korea-live`가 여전히 존재한다. 읽기 전용 확인 결과 22개 파일, 43,960,537 bytes이며 `feed.stop`, `replay.sqlite3` 및 `-wal`/`-shm` sidecar가 포함된다.
- 현재 feed/agent의 process command line은 프로젝트 `runtime/markets/korea/live`를 사용하며 AppData 복사본을 참조하지 않는다.
- 삭제가 완료되지 않은 이유: 이전 재귀 삭제 실행이 자동 도구 검토에 의해 거부됐다. 재시도나 정책 우회 삭제는 하지 않았으며 완료라고 표현하지 않는다. 자동 검토가 허용하는 명시적 삭제 수단을 사용할 수 있게 되면, 현재 미사용 상태와 DB sidecar를 재확인한 뒤 제거한다.

## P1 CLI runtime 경계 우회 가능성 — 2026-09-29 KST

- **응답: 보류.** `cli.py`의 `web --runtime`은 임의 값을 받고, `run_web()`이 이를 `web_app.serve()`에 넘긴다. `serve()`는 절대 경로를 프로젝트 경계로 제한하지 않는다.
- `Supervisor.__init__()`의 `self.runtime.mkdir(parents=True, exist_ok=True)` 때문에 외부 절대 경로가 전달되면 실제 외부 폴더 생성으로 이어지는 코드 경로가 확인된다. 외부 디렉터리를 만들지는 않았다.
- `paths.default_runtime_dir()` 및 `start_stockrl.py`의 기본 runtime 경계 검사는 있지만, CLI override에서는 호출되지 않는다. 이는 다른 외부 경로 우회 지적과 중복될 수 있으나 web CLI의 직접 생성 지점으로 별도 기록한다.
- 재확인 조건: `serve()` 경계 검사를 추가해 프로젝트 밖 경로를 생성 전에 거부하고, 거부 동작을 확인한다.

## P1 GitHub snapshot과 live runtime 시점 차이 — 2026-09-29 KST

- **응답: 수정함.** `da5dd24` runtime snapshot은 15:53:18 KST였고 이후 11개 tracked 파일이 달라진 것을 확인했다. 새 온라인 snapshot을 `c98ef4c`(16:05:57 KST)에 비공개 GitHub `main`으로 올렸다.
- `replay.sqlite3`와 `market.csv.sqlite3`는 실행 중인 파일을 복사하지 않고 SQLite online backup API로 메모리 snapshot을 만들었다. 두 DB 모두 `PRAGMA integrity_check=ok`; 업로드한 snapshot 크기는 각각 13,438,976 bytes와 5,767,168 bytes였다.
- **현재 재확인:** 16:07:48 KST에 live writer가 다시 여섯 파일을 갱신했다: `agent/metrics.json`, `agent/replay.sqlite3`, `live_feed_metrics.json`, `logs/feed.log`, `market.csv`, `market.csv.sqlite3`. 현재 온라인 DB snapshot도 integrity `ok`지만 저장된 DB blob과 내용이 다르다. 따라서 c98ef4c는 특정 시점 복구본이며 live runtime은 그 뒤에도 계속 변한다.
- GitHub runtime snapshot을 다시 만들 때 온라인 DB backup을 유지한다. live tail의 자동 동기화는 설정하지 않았다. 별도 백업 실행 시점이 정해지면 그때 새 snapshot으로 기록한다.

## P1 재시작 시 미성숙 RL 경험 복구 — 2026-09-29 KST

- **응답: 보류.** `global_online.py::follow_csv()`의 `pending`과 `portfolio_pending`은 함수 지역 목록이며 시작 때 둘 다 빈 값으로 만든다. 목록을 저장/복구하는 코드는 확인되지 않았다.
- 지속되는 값은 `live_cursor.json`의 cursor, `paper_account.json`의 계좌 및 next-bar paper-order pending, `replay.sqlite3`에 이미 기록된 성숙 경험이다. PaperAccount의 pending 주문은 미성숙 RL experience 목록을 대신하지 않는다.
- cursor는 처리한 bar마다 저장되고 재시작 시 이후 bar부터 계속한다. 따라서 재시작 전에 아직 horizon outcome을 기다리던 experience는 복구되지 않아 replay에 성숙 경험으로 들어가지 못할 가능성이 있다.
- 이는 데이터 유실 가능성의 코드 근거이며 실제 유실이 발생한 건수는 확인하지 못했다. 재시작 이력과 당시 pending 목록이 영속 기록에 없으므로 실제 손실이라고 단정하지 않는다.
- 재확인 조건: pending experience를 SQLite에 영속화하고, 재시작 후 중복 없이 한 번만 replay에 반영되는지 검증한다.

## P2 README/CODEX 인수인계 문서 드리프트 — 2026-09-29 KST

- **응답: 수정함.** `README.md`와 `CODEX_HANDOFF.md`의 현재형 설명을 현재 source와 대조해 고쳤다: mature replay는 bounded SQLite, live cursor/account/validation ledger는 runtime JSON, outcome `pending`과 portfolio pending 및 validation work queue는 memory-only, 미성숙 경험은 restart 후 복구되지 않는다.
- 승급 비교 설명은 현재 sequential paper-account net return 비교로 수정했고, champion lineage hold 때문에 현재 학습/승급이 보류 중이라는 점을 구분했다.
- 경로 및 보관 정책은 `runtime/markets/korea/live`, private GitHub 시점 snapshot, 모델 checkpoint 제외로 정리했다. `web --runtime` 외부 경로 누락도 주의사항에 유지했다.
- SIDE_REVIEW_HANDOFF 맨 앞에 시간순 상태 안내를 추가했다. 이전 runtime-global 경로와 당시 Git 제외 정책을 삭제하지 않고 과거 중간 상태로 표시했으며, 후속 `da5dd24`/`c98ef4c` 보관 이력을 연결했다.
- **확인:** source `GlobalReplayBuffer`/`follow_csv()`/`PaperAccount.save()`/validation state code와 대조했다. 코드·모델·dashboard HTML은 바꾸지 않았다.

## P2 운영 외 `.pt` 생성 경로 점검 — 2026-09-29 KST

- **현재 실제 상태 응답: 재현 안 됨.** `Desktop\모델`에는 `champion.pt`, `candidate.pt` 2개만 있다. 프로젝트 전체 `.pt` 검색 결과 0개이며 `runtime/markets/korea` `.pt`도 0개다. `runtime-global-cuda-final`, `runtime-global-research-pretrain`, `runtime-global-market-training` 기본 출력 디렉터리도 없다.
- 현재 Python 프로세스는 8766 web server, `live-feed`, `global-online`뿐이다. global-online command line은 Desktop `모델`을 model-dir로 지정한다. warmstart/pretrain/distill/legacy continuous 연구 스크립트는 현재 실행 중이 아니다. 이는 과거 실행 이력이 없다는 증거는 아니다.
- **잠재 생성 경로 응답: 보류.** 운영 launcher의 live `global-online`은 model_dir에 `candidate.pt`를 사용하고 replay는 SQLite다. 반면 비운영 코드 경로에는 추가 checkpoint 출력이 남아 있다:
  - `warmstart_fi2010_candidate.py`: `runtime-global-research-pretrain/candidate.pt`; `warmstart_macrophft_candidate.py` 및 `build_public_teacher_replay.py`: 같은 경로의 `candidate.pt`/`teacher_replay.pt`.
  - `pretrain_portfolio_agent.py`: `runtime-global-market-training/portfolio-pretraining/candidate.pt`.
  - `distill_fincast_offline.py` / `distill_fincast_temporal.py`: `runtime-global-market-training/.../candidate.pt`; 일부 `distill_*`은 필수 `--runtime`에 `candidate.pt`를 쓴다.
  - legacy `continuous` / `replay` 기본 state 경로는 `runtime/markets/<market>/...` 안에 `candidate.pt` 및 `replay.pt`를 생성할 수 있고, legacy `train`은 `baseline-checkpoints/*.pt`를 만들 수 있다.
- 따라서 **실제 운영 폴더의 두 파일 원칙은 현재 지켜지고 있고, 프로젝트 전체에 추가 `.pt`를 만들 수 있는 잠재 경로는 남아 있다.** 요청대로 코드 수정/연구 스크립트 실행은 하지 않았다. 해당 도구를 사용할 경우 별도 승인 또는 저장 정책 변경 뒤 다시 확인한다.


## 2026-09-29 runtime 경로 정리

- **확인함:** feed와 agent는 프로젝트 안 `runtime/markets/korea/live`를 사용한다. `StockRL Start.bat`, `start_stockrl.py`, `src/stockrl/paths.py`의 기본 경로도 프로젝트 runtime으로 맞췄다. 새 시장은 `runtime/markets/<market>/live`로 구분한다.
- 당시 8766 API는 시스템·feed·agent 실행 상태를 반환했고 실제 주문은 OFF였다. 이는 프로세스 상태이며, agent 판단이 최신 feed를 따라잡았다는 증거는 아니다.
- **보존함:** `%LOCALAPPDATA%/StockRL/runtime-global-korea-live/live`의 기존 자료 18개, 약 39.3 MB를 `runtime/markets/korea/archive/localcache-runtime-2026-09-29/live`에 보관했다. SQLite는 online backup API로 복사하고 `integrity_check`를 확인했다. WAL/SHM sidecar는 복사하지 않았다. 일반 파일 SHA256도 원본과 일치했다.
- 이전 runtime cursor는 `2026-09-29T04:43:00Z`, 프로젝트 live cursor는 `2026-09-29T08:00:00Z`였다. 두 replay는 각각 4,096개 경험이지만 validation window는 이전 68개, 프로젝트 111개로 내용이 같지 않다. CSV 기록은 이전 파일이 프로젝트 파일의 앞부분이었다.
- **보류:** 바깥 `%LOCALAPPDATA%` 폴더 삭제. 자동 도구 검토가 삭제를 거부했다. 기존 사본은 프로젝트 안 archive에 보존했으며, 외부 사본은 삭제 완료로 표시하지 않는다.
- 활성 `live` 파일은 `.gitignore`로 Git에서 제외하고, 복구 snapshot만 별도로 저장한다. SQLite는 파일 복사 대신 online backup을 사용하며 WAL/SHM을 따로 다루지 않는다.

## P1: SQLite runtime snapshot 안전성

- **수정함:** 실행 중인 SQLite 데이터베이스를 파일 복사로 백업하면 WAL에 남은 최신 변경을 놓칠 수 있다. `scripts/snapshot_runtime.py`는 SQLite online backup API를 사용하고, 일반 파일은 복사 중 변화를 확인하며, staging 폴더를 완성한 뒤 snapshot으로 확정한다.
- 실행 중 snapshot은 `--allow-live`를 명시해야 한다. WAL/SHM 파일과 checkpoint는 따로 복사하지 않는다. 원본 runtime 데이터베이스는 건드리지 않는다.
- 한국 runtime snapshot `runtime/markets/korea/snapshots/20260929T093208Z`를 확인했다. replay `experiences=4096`, `windows=111`, market index `seen=102597`; SQLite integrity check는 모두 `ok`, 별도 sidecar 복사는 0개다.
- 활성 `runtime/markets/*/live/**` 파일은 Git에서 제외하고, 복구 snapshot만 커밋 대상으로 둔다. 실제 runtime 파일은 작업 중 그대로 보존했다.

## P0: agent 지연과 candidate 학습 분리

- **수정함:** `global_online.py`에서 보호 champion SHA lineage 문제가 candidate 학습까지 막던 조건을 제거했다. candidate 학습은 이어가되 `_commit_candidate()`는 보호 SHA가 일치할 때까지 승격을 계속 차단한다. 검증이 lineage 문제로 끝나도 학습된 candidate 파일은 남겨 다음 경험에 사용할 수 있게 했다.
- **확인함:** Python 컴파일과 격리된 동작 확인을 통과했다. lineage hold 상태에서도 학습 루프가 paper 경험 대기로 진행하고, 보호 SHA가 불일치하면 승격을 거절하며 모델 파일을 쓰지 않는다.
- **수정함:** `web_app.py`가 market SQLite의 최신 bar와 agent `live_cursor.json`을 비교한다. 5분보다 뒤처지면 API health를 `stale`로 표시하고 agent 건강 상태를 반영한다. dashboard HTML은 바꾸지 않았다.
- **확인함:** 당시 feed 최신 시각은 `2026-09-29T09:50:00Z`, agent cursor는 `2026-09-29T08:00:00Z`였다. 차이는 6,600초, 1,183개 bar였다. agent 프로세스는 살아 있었지만 판단 기록은 오래되어 최신 source health 판정은 `stale`이다.
- **중복:** 후속 API 읽기에서도 `candidate_learning_enabled=false`, `candidate_stage=promotion_held`, replay 4,096개, 승격 차단 사유가 확인됐다. 추가로 `updates=2`, `paper_examples_trained=2`, `candidate_validation_bars=0`, `replay-since-update=96`, agent PID 8572가 보고됐다. 이는 위 candidate 학습 차단 항목과 같은 원인이다. 새 source에서 학습 차단은 제거했지만 API를 제공하는 8766 프로세스는 구버전이라 변경을 아직 반영하지 않았다.
- **보류:** 8766 프로세스 재시작. 미성숙 `pending` 경험이 메모리에만 있어 재시작하면 결과를 잃을 수 있다. 복구 저장이 준비되기 전까지 새 source는 live에 반영되지 않는다.
- champion SHA256 `F0B1759A30262C81C957CCBA555048AC0C4B993587D795F30C59BE96D7725F02`, candidate SHA256 `5A9E8B8027EC739CDBD01422FB18A9E7FB0934321773DA51671CE0FDE675F0E5`는 기록 당시 값이다. 검증 중 모델 파일을 수정하지 않았고 실제 주문은 OFF였다.

## P1 경로 경계 및 P2 설계 항목 (2026-09-29)

- **수정함:** `paths.py`, `web_app.py`, `provider_credentials.py`, `global_online.py`, `desktop.py`, `live_feed.py`, `scripts/run_global_paper.ps1`. 외부 runtime/provider 설정/feed 출력 경로를 거부하고 checkpoint를 바탕화면 모델 폴더로 제한한다. Windows launcher는 경로 검증을 디렉터리 생성보다 앞에 둔다.
- **수정함:** `global_online.py`가 `pending`과 `portfolio_pending` metadata를 replay SQLite에 저장한다. 입력 window는 보존된 feed CSV에서 복원해 pending 저장이 전체 feature 배열을 다시 복제하지 않게 했다. 성숙한 경험의 replay 추가와 pending 제거는 같은 DB transaction으로 처리한다. 판단 시각과 종목을 키로 upsert하므로 bar 재처리 시 중복 pending row를 만들지 않는다.
- **보류:** 실행 중 8766 agent 재시작. 현재 프로세스는 이전 source로 시작되어 이미 메모리에 쌓인 pending을 새 SQLite 형식으로 내보내지 못한다. 신 source가 앞으로 쌓는 pending은 보존되지만, 기존 메모리 pending은 재시작 전 별도 상태 확인이 필요하다.
- **보류:** transformer의 선택적 `time_scale_ids`를 live inference에 연결하는 설계 작업. 현재 feed는 주로 1분 bar이고 inference에서 time-scale ID를 넘기지 않는다. 다중 시간축 사용은 목표와 현재 동작을 README에 구분한다.
- **확인함:** feed 최신 bar와 agent cursor 사이는 6,600초, 1,183 bar였다. agent 프로세스는 살아 있으나 최신 코드로 재시작하지 않아 API health 개선은 live에 반영되지 않았다.
