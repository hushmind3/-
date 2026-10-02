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

Final validation: 2,210 field comparisons, 24 mode combinations, 61 behavior checks and 46 dependency-free state/requirement/card checks passed. Actual browser checks covered navigation, search/filters, expanded details and 390/1280px responsive layouts with no horizontal page overflow. Browser console errors: zero. The server currently reports stopped Feed/Agent and a previous reward_settlement learning exception; these are displayed as operational problems, not frontend success.

Layout compatibility fix: sidebar/workspace columns and full-width command feedback are explicit. All HTML assets use cards-layout-2; old documents receiving new scripts refresh before renderer startup. The old global metric strip keeps a full-width compatibility layout. Verified the legacy-document/new-CSS case, visible command feedback, and legacy-to-current document navigation at 1920px in a read-only preview, then the actual 8766 page; no page overflow or console errors. Six version/layout regression checks were added (67 behavior checks total).
