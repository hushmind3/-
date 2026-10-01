# Frontend maintenance

The dashboard uses plain HTML/CSS and four JavaScript files. No frontend build or Node runtime is needed to operate it.

| File | Responsibility | Typical edit |
| --- | --- | --- |
| `src/stockrl/web/assets/app.js` | API, shared state, DOM updates, polling, explicit render sequence | Refresh cycle, shared number/date formatting |
| `src/stockrl/web/assets/dashboard.js` | Markets, decisions, paper accounts and operator summaries | Account cards, experience lifecycle, learner cards, trial scores |
| `src/stockrl/web/assets/controls.js` | All operator actions, credentials, independent mode controls | Button behavior, API payload, command feedback |
| `src/stockrl/web/assets/details.js` | Replay, learning, models, promotion and runtime diagnostics | Learning counters, GPU metrics, expandable diagnostics |
| `src/stockrl/web_dashboard.html` | Page structure, existing IDs, buttons, expandable panels | Layout and labels |
| `src/stockrl/web/assets/dashboard.css` | All styles | Spacing, colors, responsive layout |

## Edit rules

The six navigation views are operations/accounts, markets/decisions, experience learning, promotion trial, connections, and diagnostics. Navigation hides other views without destroying controls or state. Original anchors and control IDs remain available. Each screen starts with its relevant summary; technical details expand on demand. GPU scheduling belongs in diagnostics, and frozen trial accounts are explicitly separate from the continuing live paper accounts.

- Start at the relevant named function. Render functions are declared once; do not wrap or redefine a previous renderer.
- `app.js: render()` lists the complete render order. The final text and property values are committed once per frame.
- Use `text(id, value)`, `badge(id, value, tone)`, `property(id, key, value)` and `html(id, () => markup)` rather than writing the DOM directly.
- `textOf(id)` reads the current frame's pending value when building a longer summary.
- Escape values from the API with `esc()` before inserting HTML. `tableRows()` and `accountTable()` escape every cell.
- HTML updates are cached by element and skipped when unchanged. The HTML callback is not evaluated for a collapsed panel.
- Opening a details panel immediately renders the most recent status. It does not wait for the next poll or request model inference.
- Keep the three `/api/modes` flags independent. Each button sends only its own flag.
- UI files are read at request time with `no-store`: browser refresh applies changes. Model/feed restarts are unnecessary.
- The asset allowlist is `src/stockrl/web/resources.py`. Changes to that Python file require only a web-server reload.

## Contract regression and measurements

`tests/frontend_regression.cjs` compares this frontend with Git commit `71a2488`. It checks preservation of existing HTML IDs, 13 action/filter buttons, 12 API routes, rendered fields in five system states with details collapsed and expanded, search/filter behavior, 24 independent mode actions, six-view navigation and trial score calculations. Additional summaries and expandable panels are allowed. All HTTP actions are mocked: it never changes the live server or accounts.

Node and jsdom are needed only for this development check. Keep tooling outside the project:

```powershell
npm install --prefix "$env:TEMP/cta-frontend-tools" --no-audit --no-fund jsdom@26
$env:NODE_PATH = Join-Path $env:TEMP 'cta-frontend-tools/node_modules'
node tests/frontend_regression.cjs
```

To compare a real status snapshot as well, first save `/api/status` to a temporary file, then pass `--status-file <path>`. Do not commit runtime snapshots or credentials.

The script prints field/action counts and before/after median render time and DOM mutation counts. These timings are jsdom measurements of frontend work, not browser frame rates, GPU throughput, or model learning speed. Real browser checks must additionally cover loading, search and opening detail panels.

Intentional changes include screen organization, operator labels, explicit learner activity and result-wait states. Account arithmetic and existing numeric fields retain their meaning. Trial scores use the same average of KRW/USD net return rates as the comparison, and remain provisional while the trial runs. Zero trial return is explained alongside actual fills, holdings and latest decision counts.

Experience counts distinguish results still being evaluated, eligible replay backlog, and completed training. Completed rows disappear after both model checkpoints are confirmed; the date ledger preserves their counts. The displayed date-record completion sum covers the dates returned by the API (currently up to 14), rather than claiming lifetime coverage. Daily remaining counts come from actual retained DB rows; removed malformed rows are reported separately and are never relabeled as trained.

`learningSituation()` in `dashboard.js` is the shared source for the header learning link, learning badge, visible explanation, and per-model state. The explanation shows current work, why it waits, the condition for the next round, and each model's last confirmed completion. Zero eligible backlog is distinguished from learning OFF, GPU scheduling waits, errors and missing measurements. Run `node tests/frontend_learning_status.cjs` for its dependency-free state checks.

`tests/test_pending_rewards.py` checks closed-session settlement against authentic saved quotes, missing/open-market quotes, and unfilled-order expiration. `tests/test_online_pipeline.py` checks compaction retaining real closed-market history and daily counts after quarantined rows are removed. These fixes preserve learning inputs and never fabricate a next-day fill.

## Operator status contract

The five-part header shows collection, decision processing lag, paper execution, learning, and trial progress from one status response. Pending outcome records are distinguished from eligible training experience. The pending card shows the configured reward interval and aggregated reasons from both models; asynchronous differences between the current DB total and worker reason counts are explicitly labeled as an updating reason tally. Required timeframe history is shown as completed bar counts (lookback + 1), with sufficient and insufficient symbol counts. Inclusion in a training sample is never labeled as complete history. The input shortage warning stays visible on the learning page. Formatting coverage to two decimal places avoids displaying incomplete monthly history as 100.0%.

Existing buttons and API mutations are preserved. UI-only changes need browser reload, not a model or server restart.
