# Frontend maintenance

The dashboard uses plain HTML/CSS and four JavaScript files. No frontend build or Node runtime is needed to operate it.

| File | Responsibility | Typical edit |
| --- | --- | --- |
| `src/stockrl/web/assets/app.js` | API, shared state, DOM updates, polling, explicit render sequence | Refresh cycle, shared number/date formatting |
| `src/stockrl/web/assets/dashboard.js` | Markets, decisions, Champion/Candidate paper accounts | Account cards, positions, decision table |
| `src/stockrl/web/assets/controls.js` | All operator actions, credentials, independent mode controls | Button behavior, API payload, command feedback |
| `src/stockrl/web/assets/details.js` | Replay, learning, models, promotion and runtime diagnostics | Learning counters, GPU metrics, expandable diagnostics |
| `src/stockrl/web_dashboard.html` | Page structure, existing IDs, buttons, expandable panels | Layout and labels |
| `src/stockrl/web/assets/dashboard.css` | All styles | Spacing, colors, responsive layout |

## Edit rules

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

`tests/frontend_regression.cjs` compares this frontend with Git commit `71a2488`. It checks existing HTML IDs, 13 action/filter buttons, 23 expandable panels (including the generated runtime panel), API routes, rendered fields in five system states, search/filter behavior, and 24 independent mode actions. All HTTP actions are mocked: it never changes the live server or accounts.

Node and jsdom are needed only for this development check. Keep tooling outside the project:

```powershell
npm install --prefix "$env:TEMP/cta-frontend-tools" --no-audit --no-fund jsdom@26
$env:NODE_PATH = Join-Path $env:TEMP 'cta-frontend-tools/node_modules'
node tests/frontend_regression.cjs
```

To compare a real status snapshot as well, first save `/api/status` to a temporary file, then pass `--status-file <path>`. Do not commit runtime snapshots or credentials.

The script prints field/action counts and before/after median render time and DOM mutation counts. These timings are jsdom measurements of frontend work, not browser frame rates, GPU throughput, or model learning speed. Real browser checks must additionally cover loading, search and opening detail panels.

The only intentional wording difference from the baseline is the separator between Champion and Candidate lag values, which previously ran together.
