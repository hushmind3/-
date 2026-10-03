# Registered vertical TradingMoE

Baseline: `1c497ef`. Original checkpoints remain unmodified backups. Runtime:
`stockrl.trading_moe.TradingMoE`, a `torch.nn.Module` with fourteen native experts
registered in `experts: ModuleDict` (not manifest-only placeholder models).

## Execution

1. TimesFM, Kronos, Toto, MarketGPT, EXAONE, Chronos, Time-MoE and FinCast produce
   their preserved native outputs. Diagonal adapters normalize evidence; a soft
   router and attention fusion produce symbol market states.
2. Six trained MacroHFT modules remain a separate policy tier: slope/1–3 and
   vol/1–3. `MacroHFTInputAdapter` executes the original `Testing_Env.reset`
   preprocessing with official ETHUSDT observations and the previous position.
   Native `single_state[36]`, `trend_state[9]` and Q(flat,long) are retained.
3. One controller combines market states, policy Q evidence and sixteen existing
   account features. The initial action and allocation prior comes from trained
   MacroHFT Q votes; the learnable residual starts at zero. Unavailable policy
   evidence is masked, never unregistered.
4. `TradingMoEPaper` calls the existing PaperAccount order queue, next-observed-bar
   fills, fees/slippage, NAV, `_RewardMixin` and GlobalReplayBuffer. Real broker
   execution remains false; `paper_executable` is a separate explicit gate.
5. Executed portfolio outcomes update adapter and controller/router/fusion.
   Action credit uses actions for actually tradable symbols. Native groups are
   frozen for this run; `parameter_groups(train_experts=...)` and
   `NativeExpert.forward_tensors` expose native tensor training separately.
6. Atomic `TradingMoE.pt` stores all fourteen native states, architecture
   source/config metadata, mappings, adapters/controller, optimizer and update
   count. Load reconstructs from this one file without original weight files.
   Architecture dependencies are still required. Expert weights are not averaged.

## Run the existing checkpoint

Use the provisioned expert Python environment. Its native dependencies include
`unit-scaling==0.3.5` and `docstring-parser==0.18` for Toto.

```powershell
& 'C:\Users\hushm\OneDrive\문서\ChatGPT\금융매매모델\artifacts\experts\venv\Scripts\python.exe' `
  scripts/run_native_vertical_trading.py `
  --root 'C:\Users\hushm\OneDrive\문서\ChatGPT\금융매매모델\artifacts\experts' `
  --state runtime/trading_moe/native_vertical_run --steps 20 --resume
```

This command runs a finite consecutive historical interval. The dedicated
`http://127.0.0.1:8766/#trading-moe` screen starts a continuous historical worker
using `--resume --continuous --interval 0.1`. It shows decisions, orders, fills,
positions/cash/NAV, costs, rewards and optimizer updates. `#experts` retains
native output diagnostics. This is historical PAPER operation, not a live feed.
The ledger persists across reruns. Champion/Candidate and their accounts are not
used or reset by this command.

The eight market experts refresh every 120 historical observations in continuous
mode; their original packets are reused between refreshes. Six native MacroHFT Q
outputs and the account-aware controller execute each decision. GPU ownership is
locked and at most one native expert is transferred at a time. Native inference,
adapters, router, fusion, controller and optimizer updates use CUDA by default.
Original frozen CPU parameter views remain backed by the memory-mapped PT, rather
than being copied back into new RAM buffers after each native forward. Windows
can reclaim inactive mapped pages after loading and market refreshes.

The worker loads PT once; status polling and HTTP server restart do not reload it.
Start is idempotent and has a separate OS ownership lock. Stop finishes the current
cycle, persists account and pending reward evidence, saves the full PT/optimizer,
acknowledges trained replay rows, then exits. Resume continues the same account,
native market timestamp and optimizer count. Replay IDs are scoped to the account
episode to prevent cross-account ID collisions.

Dedicated APIs: `GET /api/trading-moe/status`, `POST /api/trading-moe/start`,
`POST /api/trading-moe/stop`. They read a small status snapshot, not model weights.

## Actual input provenance and units

- MacroHFT: official `df_val.feather`, ETHUSDT February–May 2023. Runtime uses
  consecutive real minute observations, original feature lists and reset
  preprocessing. No synthetic 36+9 vectors.
- MarketGPT: real AAPL NASDAQ ITCH from December 30, 2019, encoded using original
  Vocab/encode_msgs. Its original timestamp is preserved as archival reference,
  **not** represented as the 2023 ETH order book. This mixed-date run establishes
  execution, not an aligned profitability study.
- TimesFM/Chronos: official historical AAPL excess returns ending before the
  ETH input date. Other price experts: real ETH OHLCV/price. Kronos amount uses
  an explicitly marked price × volume proxy.
- Existing paper engine uses integer units. One explicit ETH contract unit is
  0.001 ETH, with quote per unit equal to ETH price / 1000. Expert input prices
  are unchanged. UI reports ETH quantity and price. USDT is evaluated at USD
  parity in the isolated cash ledger; no leverage.
- AAPL is context only without a current executable quote. A controller opinion
  then produces no fabricated AAPL fill.

Official sources: [MacroHFT](https://github.com/ZongweiLiang/MacroHFT),
[MarketGPT](https://github.com/aaronwheeler/MarketGPT),
[real MarketGPT input](https://huggingface.co/datasets/aaronwheeler/MarketGPT-datasets).

## Files to change

| Concern | File |
|---|---|
| Native architecture reconstruction | `moe_native.py` |
| Native MacroHFT input adapter | `moe_inputs.py` |
| Vertical controller, native model/save/load | `trading_moe.py` |
| Existing paper/reward/replay connector | `moe_paper.py` |
| Controller update from paper reward | `moe_training.py` |
| Consecutive run and registry publishing | `scripts/run_native_vertical_trading.py` |
| Dedicated worker lifecycle/API | `web/trading_moe.py`, `web/server.py` |
| Dedicated operating screen | `web/assets/trading-moe.js`, `web_dashboard.html` |

Local logs: `cycles.jsonl` and `report.json` under the chosen state path. The
tracked result summary records actual numbers. No additional long validation or
profitability testing was performed for this fast integration request.
