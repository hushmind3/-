# Dashboard preservation rule

Do not recreate, replace, or redesign the dashboard from scratch. Preserve the existing `src/stockrl/web_dashboard.html` structure, section order, navigation, labels, controls, and styling.

For future dashboard work:
- Make only the smallest targeted change needed for the request.
- Before editing, create a timestamped byte-for-byte backup of the current HTML in a clearly named `backups/dashboard/` folder. Never overwrite an earlier backup.
- Do not claim a backup is the original if it was made after an accidental rewrite. The original dashboard is currently not recovered; do not reconstruct it from screenshots and call that a restoration.
- If the original file cannot be located, stop before changing the dashboard and report that limitation. Do not generate a substitute UI.
- After a permitted targeted edit, verify that all existing navigation anchors and section IDs remain, inspect the rendered page, and confirm the live controls still work.
- Keep the portable, source, and Mac-transfer copies synchronized without replacing their dashboard with a newly designed page.

This instruction applies to every agent and contributor working in this project.