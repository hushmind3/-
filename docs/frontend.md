# Frontend maintenance

Plain HTML/CSS/JavaScript; no frontend build or runtime dependency was added.

## Screen ownership

| File | Edit here |
| --- | --- |
| app.js | Shared API/state, polling, navigation, cached DOM updates |
| controls.js | All commands, mode payloads, request/result/error feedback |
| dashboard.js | Operations overview and market/decision screen |
| accounts.js | Continuing Champion/Candidate paper accounts |
| learning.js | Experience processing, learning state, model comparisons |
| trial.js | Frozen promotion-trial accounts, progress and history |
| details.js | GPU, input coverage, processing history and runtime diagnostics |
| experts.js | Actual TradingMoE registry, residency, raw output (read-only) |
| web_dashboard.html | Layout, labels, stable element IDs and expandable panels |
| dashboard.css | Shared cards, badges, typography and responsive rules |

JavaScript files are under src/stockrl/web/assets. Each large screen owns its fields; shared formatting and DOM helpers live in app.js. Account and trial renderers have separate lifecycles. No training, reward or account-reset rules were changed.

## Feature inventory preserved

Six views: operations/accounts, markets/decisions, experience learning, promotion trial, connections and diagnostics.

The regression inventory preserves 13 action/filter buttons, all existing element IDs, independent decision/paper/learning mode controls, credentials/provider controls, feed/agent operations, account display, trial progress/history, market search/filters and expandable diagnostics.

Twelve API routes remain: see the executable inventory in tests/frontend_regression.cjs. No endpoint or command payload was replaced. Every mode POST contains only its own flag. Long-term account reset remains distinct from promotion-trial accounts; its confirmation describes both continuing accounts explicitly.

## Update flow

1. One in-flight status request updates shared state.
2. The current screen renders; hidden screens do not run their renderers.
3. Cached DOM nodes receive final changed text/properties once per frame.
4. Stable account cards update their numbers without recreating the cards.
5. Folded positions, fills, costs and model comparisons skip their calculations.
6. Opening a detail panel immediately uses the latest state. All closed ancestors are checked.

Commands use one pending-command state. Request labels survive polling; conflicting buttons are disabled while a request runs. Success requires refreshed server state, mode confirmation checks the actual flag, and failure restores the control with visible feedback. Unknown mode state is disabled instead of guessed.

UI assets are served from files with no-store: refresh the browser to apply UI changes. The only Python change adds three screen scripts to the asset allowlist in web/resources.py. No model/feed restart is required for UI edits.

## Labels and measurements

Cards show process health, both account results, learner state, trial state, experience counts and frequently used controls before resources. Technical detail is folded.

- Model decision, paper execution and experience learning show their independent permission/state.
- Two continuing accounts reset is named explicitly; web-only restart is named explicitly.
- Last-round samples, total_seconds and loss_mean come from confirmed round metrics.
- Pending outcomes are distinct from eligible replay rows and confirmed completion.
- Completed date-record totals cover returned dates, not lifetime history.
- Replay size uses replay_file_bytes; blocked rows use quarantined plus unsupported rows.
- Partial timeframe input is not labeled complete history.
- Missing measurements show unknown rather than an invented zero or healthy state.

## Verification

Tests need Node and the existing development-only jsdom installation. The running dashboard does not need Node/jsdom. API mutations in tests are mocked; tests do not reset or trade live accounts.

- tests/frontend_regression.cjs compares against pre-refactor commit a0f21b7: existing fields, 13 buttons, 12 routes, 24 independent mode combinations and six-view navigation.
- tests/frontend_behavior.cjs covers stable account DOM, folded work, screen routing/direct first entry, control feedback/failures, payloads, reset cancellation and request coalescing.
- tests/frontend_learning_status.cjs covers learner/pending/trial/input requirements and operator-card states without jsdom.

Run the regression with --status-file pointing to a temporary /api/status snapshot to include current server data. Do not commit snapshots or credentials.

Before/after timings printed by the tests measure frontend work in jsdom, not GPU training throughput or browser frame rate. One measured real-status run: collapsed render 35.28 to 5.94 ms, expanded render 142.33 to 79.36 ms; unchanged-refresh DOM mutations 4/6 to 0/0. Hidden model-comparison calls during 20 overview refreshes fell from 20 to 0. Re-run timings vary with machine load.

Removed duplicated screen writes, account-to-trial rendering, render-time listener registration, a second command busy flag, 37 overwritten CSS declarations and 35 obsolete global-strip rules. Account UI edits now start in accounts.js instead of the previous 3,773 mixed dashboard/details lines; trial edits start in trial.js.

Final validation: 2,210 field comparisons, 24 mode combinations, 61 behavior checks and 46 dependency-free state/requirement/card checks passed. Actual browser checks covered navigation, search/filters, expanded details and 390/1280px responsive layouts with no horizontal page overflow. Browser console errors: zero. Current runtime state is reported from model_runtime; historical errors are shown separately. The replay reward_settlement schema mismatch was fixed on 2026-10-03.

Layout compatibility fix: sidebar/workspace columns and full-width command feedback are explicit. All HTML assets use cards-layout-2; old documents receiving new scripts refresh before renderer startup. The old global metric strip keeps a full-width compatibility layout. Verified the legacy-document/new-CSS case, visible command feedback, and legacy-to-current document navigation at 1920px in a read-only preview, then the actual 8766 page; no page overflow or console errors. Six version/layout regression checks were added (67 behavior checks total).

## 2026-10-03 운영 제어 점검

- 작업 선택과 시스템 명령을 운영 화면 맨 위로 이동했습니다.
- Champion/Candidate 시작·저장 후 정지를 시스템 명령에 배치했습니다. 계좌 카드에는 메모리·판단·손익만 남겼습니다.
- 모호한 `모델 코드 적용` 버튼과 프론트 이벤트를 제거했습니다. 기존 서버 API는 유지합니다.
- 모델 실행 상태를 시세 수집 상태와 분리해 모든 운영 카드에서 동일하게 판정합니다. 허용 설정을 실행 중으로 표시하지 않습니다.
- 과거 학습 오류는 접힌 기록으로 보존하고, 실행 중인 모델의 현재 오류와 구분합니다.
- 모델 버튼은 계좌 API의 유무/렌더링에 의존하지 않습니다. 실패 메시지는 주기 조회가 덮어쓰지 않습니다.
- named TradingMoE worker도 기존 autonomy.json의 판단/가상 체결/학습 설정을 읽습니다. 판단이 꺼져도 저장 경험 학습을 허용할 수 있습니다. 전용 자동매매 화면은 별도 실행 제어를 유지합니다.
- Experience의 종료 시세 보상 필드 3개를 복원해 기존 replay 기록을 삭제 없이 로드합니다.
- Python 126개, 프론트 상태 설명 46개, 조작 93개, 필드 비교 2,200개, 독립 모드 조작 24개 통과. 실제 브라우저 8개 메뉴의 페이지 전환·가로 넘침과 콘솔 오류 확인.
- 반복 조회의 DOM 변경은 접힌/펼친 상태 모두 0회. jsdom 측정: 접힌 상태 2.25→1.49ms, 펼친 상태 2.63→1.29ms. GPU 학습 처리량 측정이 아닙니다.
- 사용하지 않는 옛 UI 자료는 휴지통/2026-10-03에 보관합니다.
