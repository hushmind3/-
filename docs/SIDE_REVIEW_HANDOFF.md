# 사이드 검토 인수인계 목록

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
- **champion 보호 확인:** 사용자가 지정했던 SHA256 `2D0D45702C30EE167B0E982628D21D48CD54C00F572495B06585E326AC37797A`와 현재 `Desktop\\모델\\champion.pt` SHA256은 일치하지 않는다. 현재 파일은 1,025,823,889 bytes, SHA256 `F0B1759A30262C81C957CCBA555048AC0C4B993587D795F30C59BE96D7725F02`; validation state도 이 현재 SHA를 비교 시작 champion으로 기록한다. metrics는 promotions 0, last promoted false이며 파일 수정 시각은 2026-09-28 22:19:53 KST다. 불일치 원인은 이번 확인에서 규명되지 않았고 어떤 checkpoint도 수정하지 않았다. 비교 결과를 보고하기 전에 사용자가 보호 대상으로 지목한 파일/버전을 다시 확인해야 한다.

## 운영 지침 — champion 계보 확인 전 승급 보류

- **응답: 수정함.** 사용자의 지침에 따라 보호 SHA256 `2D0D…37797A`와 현재 champion SHA256 `F0B1…D7725F02`의 계보가 확인될 때까지, 64/64가 되어도 candidate를 champion으로 승급하지 않는다.
- **변경:** `src/stockrl/global_online.py`는 비교 기준 champion SHA가 보호 SHA와 다르면 candidate를 유지한 채 `promotion_held`로 전환하고 추가 candidate 학습/승급을 막는다. 그 전에는 8766의 `observe=false`, `paper=false`로 구버전 agent의 60/64 검증 진행을 멈췄다.
- **현재 적용 상태:** API는 `agent=true`, `paper=false`, `observe=false`, validation 60/64, `promotions=0`, `last_candidate_promoted=false`, `real_orders=false`를 반환했다. 실행 중인 PID 7068은 guard 이전에 시작한 구버전이므로 source/runtime 경로 이전과 함께 새 코드로 재시작하기 전까지 API의 `promotion_blocked_reason=null`은 보호가 적용됐다는 증거가 아니다. 관찰/paper를 다시 켜지 않는다.
- **재확인 조건:** runtime을 프로젝트 폴더로 이전한 뒤 새 agent를 시작하고, 64/64 이후에도 `candidate_stage=promotion_held`, 명시적 blocker, 승격 0회를 확인한다. SHA 계보를 확인하고 보호 기준을 정하기 전에는 checkpoint를 수정하거나 교체하지 않는다.

## Runtime 위치 이전

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

### GitHub runtime snapshot과 추적

- **응답: 수정함.** 기존 커밋 613c330에는 runtime 21개 파일, 43,897,551 bytes가 포함됐다. 각 파일은 100MB 미만이며 .pt는 없다. 현재 로컬 runtime은 실행 중 갱신되어 44,139,072 bytes다.
- 기존 커밋과 이력은 되돌리거나 다시 쓰지 않는다. 로컬 runtime 파일은 그대로 두고, Git 추적에서 제외해 향후 변경분이 자동으로 커밋되지 않게 했다. 이전 snapshot은 과거 커밋에 남는다.
- 서로 다른 side-review 제안은 이 방식으로 처리한다. 기존 업로드 이력은 유지하면서, 계속 바뀌는 runtime은 이후 Git 저장에서 제외한다.

### 추가 항목 분류

- **새 .pt 생성: 재현 안 됨.** 확인한 runtime 폴더에는 .pt가 없고 모델 checkpoint는 건드리지 않았다.
- **대시보드 수정: 재현 안 됨.** dashboard HTML을 수정하지 않았다.
- **8767 실행: 재현 안 됨.** 확인 시점에 8767 listener가 없었다.
- **최신 side-monitor 상태: 중복.** 12/64 진행, 승급 0회, candidate 학습 차단, SHA blocker 유지는 위 P0 상태 항목에 연결한다。

