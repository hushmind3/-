# Independent Champion/Candidate execution

`POST /api/start` starts only the shared market feed. Each model has its own
`POST /api/models/{champion,candidate}/{start,stop}` operation. Start is idempotent.
Stop completes current work, saves the named checkpoint and account, and releases
that role. It does not stop the other role or the feed. Both OFF is supported.

The model is selected by its current filename in the configured model directory:
`champion.pt` or `candidate.pt`. No previous Champion hash allowlist or manual
confirmation is used. The file's own architecture configuration is loaded.

## Existing runtime reuse

- Transformer checkpoints use the existing shared OnlineGlobalAgent, inference,
  independent paper ledgers, reward/replay and learner. Explicit role requests
  replace eager dual-model loading. Candidate alone does not load Champion.
- A named TradingMoE checkpoint uses the existing resident native worker, with
  `--checkpoint` pointing at that exact file. The resident model loads once.
  This existing path runs official ETHUSDT historical PAPER observations; it is
  not presented as live stock execution. Its role ledger persists in
  `agent/{role}_moe`; previous stock ledgers remain intact and are labeled in UI.
- An idle Transformer coordinator exits after both roles save/unload, releasing
  its CUDA context. Role workers and feed survive an HTTP-only server restart.
- A stopped role cannot be silently loaded for learning or a promotion trial.
  A trial requires both Transformer roles; stopping one pauses the trial and
  leaves the long-term accounts intact.

The operating cards show actual loaded state, residency, recent decision and
account P&L. Transformer memory is tensor storage (including observer/trial
snapshots); TradingMoE RAM is worker RSS. Cached CUDA allocator space is shared
and is not falsely attributed to a specific role. CPU residency does not mean
CPU inference: the calculation device is displayed separately.

## Change locations

- Model residency and safe save/unload: `online/model_lifecycle.py`
- Supervisor requests, feed and role process ownership: `web/runtime.py`
- Existing TradingMoE resident worker: `web/trading_moe.py`
- Operator buttons/state: `web/assets/controls.js`, `web/assets/accounts.js`
- Status and account projection: `web/status.py`

Actual brief operations observed: feed ON with both models unloaded; Candidate
alone loaded; both named models loaded; Champion saved/unloaded while Candidate
and feed continued; Candidate then saved/unloaded. Browser displayed all four
buttons and actual unloaded RAM/GPU state without console errors. No long model
probe, expert download or test suite was run.
