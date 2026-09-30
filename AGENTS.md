# Dashboard preservation rule

Do not recreate, replace, or redesign the dashboard from scratch. Preserve the existing `src/stockrl/web_dashboard.html` structure, section order, navigation, labels, controls, and styling.

For future dashboard work:
- Make only the smallest targeted change needed for the request.
- Do not create local filesystem backup copies of dashboard files.
- Before every dashboard HTML edit, commit and push the current project state to GitHub and verify the push succeeded. The GitHub commit is the recovery point; do not create a separate local backup file.
- Do not claim a commit is the original if it was made after an accidental rewrite. Do not reconstruct a dashboard from screenshots and call that a restoration.
- If the original file cannot be located, stop before changing the dashboard and report that limitation. Do not generate a substitute UI.
- After a permitted targeted edit, verify that all existing navigation anchors and section IDs remain, inspect the rendered page, and confirm the live controls still work.
- Keep the portable, source, and Mac-transfer copies synchronized without replacing their dashboard with a newly designed page.

This instruction applies to every agent and contributor working in this project.

## Project storage boundary

- Do not create folders outside this project directory.
- Store each market's StockRL runtime under the project root at `runtime/markets/<market>/live` (for example, `runtime/markets/korea/live` or `runtime/markets/nasdaq/live`).
- Launchers must reject runtime paths outside this project instead of silently creating them.
- Update `README.md` whenever project paths, runtime behavior, or operating rules change.
- The champion selected by the user is the current baseline. A SHA256 identifies which champion was present during a validation window; it is not a fixed allowlist or promotion lock. Promote only after at least 64 identical future paper-account bars show higher net return, and reject promotion if the champion changes during that window.
- Keep each active runtime under `runtime/markets/<market>/live` and include project files and runtime state in GitHub so a downloaded project retains its account, replay, cursor, metrics, market data, and other saved state. Exclude only model weight files (`.pt`, `.pth`, `.ckpt`, `.safetensors`). Never place runtime state outside the project.
- This repository is public: never commit API keys, secrets, access tokens, or real account credentials. Paper-account state is portable project data; OS credential-store contents are not.
