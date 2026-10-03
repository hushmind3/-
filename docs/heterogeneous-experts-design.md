# Frozen heterogeneous experts: integration design before implementation

Date: 2026-10-02. Baseline: `c4a61a2` on main. Phase: research and standalone inference only; no training and no live replacement.

## Repository review and boundaries

The initial audit read 203 repository text files, 2,599,129 bytes / 57,067 lines, including source, scripts, tests, configuration and documentation. Binary runtime databases and model files are inspected separately, not interpreted as source. The audit records each file's SHA256, size, definitions and imports. Credentials are never copied into the report. Runtime files that change while the server runs are excluded from the integration artifact.

| Existing component | Contract to preserve |
|---|---|
| `global_online.py`, `online/observation.py` | independent Champion and Candidate inference, shared durable observations, independent actions and account state |
| `global_transformer.py` | `[batch,time,symbol,feature]`, explicit valid masks, fixed symbol identity; action order SELL=0, HOLD=1, BUY=2 |
| `market_training.py`, `multiscale.py`, `daily_encoder.py` | market context, portfolio/account state, completed timeframe histories, no future observations |
| `paper_account.py` | independent KRW/USD books; cash is the LAST allocation entry; action-gated orders fill at later real quotes with costs, integer shares and available cash |
| `online/rewards.py`, `online/losses.py` | realized action and account credit, delayed successor value, cost-adjusted reward in percentage points; goal reward separate from contest score |
| `replay_store.py`, `online/replay_updates.py` | durable FIFO, independent training acknowledgements, delete only after BOTH models' checkpoints confirm consumption; retain incomplete observations and delayed credit |
| `online/checkpoint.py`, `state_io.py` | atomic write/fsync/replace; model schema and symbol-map validation; no checkpoint acknowledgement before durable save |
| `gpu_scheduler.py`, `online/learning.py` | cooperative priority queue, synchronize GPU before releasing a segment, candidate CPU offload; the lock is thread-local to one process, NOT a cross-process GPU mutex |
| `online/validation.py`, `web/accounts.py`, `web/workers.py` | frozen contest snapshots, isolated trial accounts, long-term operating account not implicitly reset |
| `paths.py` | old online agent only accepts champion.pt/candidate.pt in Desktop/model; the research system must not bypass this guard to overwrite them |
| `public_teachers.py`, distillation scripts | legacy teacher-to-student training is a separate mechanism; none of these training scripts run in this phase |
| `web/*`, assets | no API/UI changes or worker startup needed for this phase |

No live component imports the new subsystem. Standalone commands only read explicitly supplied market snapshots and originals. Output goes to a research report directory, never the live replay, account or cursor. Before CUDA verification, check that the live Agent is stopped: an independently created FairGpuScheduler cannot coordinate with the live process.

## Artifact identity and immutable experts

Pin each official repository and checkpoint revision, retain model card/license/config/source, validate file length and upstream LFS SHA256. Inspect safetensors headers and safe PyTorch state dictionaries on CPU. Separate tensor elements (including buffers/duplicated serialization) from actual module parameters; sparse MoE TOTAL and ACTIVE counts are different. Inspect original dtype; do not infer it from a filename or silently replace FP32 originals with quantized ones.

Vendor code/config, native data and Python environments live in project `artifacts/experts`. Operating PT files and all original expert weights live in `Desktop/모델`; the shared GPU ownership lock lives in project `runtime/gpu-owner.lock`. No large binaries are added to Git. FinText Global has yearly vintages, not one universal checkpoint: use the latest public 2023 Small/20M revisions for this audit, and record training cutoff. A historical backtest must select a vintage trained BEFORE its evaluation period.

EXAONE's downloaded license restricts model/derivative/output use to research and education without a separate commercial agreement. MacroHFT and EarnHFT repository license availability must be recorded; public access is not a license grant. EIIE author's implementation is GPL-3.0 and legacy TensorFlow. These affect later distribution/live deployment, not whether local artifact research can be performed.

## Input and output adapters

Use a typed market snapshot: symbol order, currency, sampling cadence, as-of time, completed-bar timestamps, OHLCV/amount, available masks, portfolio weights, actual LOB/ITCH records, and explicit historical risk-free returns when excess returns are required. Availability is a first-class input, not fabricated zeros that masquerade as measured data.

| Expert | Native input | Native output and role |
|---|---|---|
| FinCast | contiguous univariate numerical series, native patch/frequency conventions | point and quantile future series; slower forecast/regime evidence |
| EXAONE Finance | masked univariate numerical series | 21 quantile trajectories; uncertainty/finance context, not a language model |
| Kronos Base + tokenizer | timestamped OHLCV plus amount, native six fields | sampled future OHLCV/amount; bar trajectory evidence |
| MarketGPT | official encoded ITCH order messages and its vocabulary | next-message token distribution / generated order flow; NOT a BUY/SELL policy or an OHLCV forecast |
| FinText Chronos Small Global 2023 | daily excess returns, native Chronos tokenizer | sampled daily excess-return trajectories; long-horizon distribution |
| FinText TimesFM 20M Global 2023 | daily excess returns, native small TimesFM decoder | daily return point/quantile forecast; cheap long-horizon evidence |
| Time-MoE Large | native normalized time series, time-token format | autoregressive numerical forecast; sparse temporal expert |
| Toto 2.0 313m | multivariate series with timestamp/padding/ID masks | multivariate probabilistic samples/quantiles; common-market dynamics |
| EarnHFT | exact second-level feature schema and inventory; minute-level routing | learned inventory policies only if authentic trained policy files are available |
| MacroHFT | original 36 market + 9 context features and previous binary action (verified from tensor shapes and feature-list files) | six ETHUSDT sub-agent Q distributions; inventory target, not a generic stock predictor; published hyper-agent absent unless verified separately |
| EarnMore | original masked stock-pool features + portfolio observation | portfolio actor weights if a trained actor checkpoint is published |
| DeepScalper | original macro/micro embedding, inventory, action branches | directional/size Q policy if published; SARL LSTM is NOT DeepScalper |
| EIIE | relative price tensors, previous weights, original cash convention | portfolio weights including cash if authentic trained evaluator weights exist |

Never resample 1m OHLCV into fictitious 1s order messages. Never label raw daily price returns as excess returns. Synthetic shape verification is permitted only when explicitly labeled; it is not market validation. Preserve native policy action conventions and translate inventory changes against CURRENT holdings; a short action cannot be mapped directly to a cash-only account's sell without an explicit capability limitation.

## Shared representation, router and fusion

Each selected expert keeps its architecture/tokenizer/normalization and frozen parameters. Its adapter emits an evidence packet with symbol, modality, source horizon/cadence, as-of time, validity, native prediction or policy logits/weights, uncertainty, units, and artifact revision. Raw native representations remain available for future learned projections. Do not compare a policy Q-value to a return quantile as if they were the same unit.

Recommended learned architecture (not trained in this phase): separate small projection for each native packet -> common evidence tokens + market/account tokens -> cross-attention fusion -> per-symbol actor/value and cash-inclusive allocation heads. A capability mask precedes a budget-aware top-k router. Router/fusion/head have their own checkpoint and version. Experts are frozen, and gradients stop at their boundary. Missing modalities remove experts from routing; they do not produce invented evidence.

Initially router is deterministic and documented: eligible native input -> modality/cadence tier -> measured latency/memory budget -> at most one expert in each selected tier, executed sequentially. It does not pretend to be a trained router. Same daily input may reuse a forecast cache keyed by expert revision, exact input hash and horizon; changing portfolio state must still change final policy evaluation. Only a trained fusion head can implement learned joint trading behavior. An explicit conservative diagnostic head may generate the required output schema before training, but its output must be labeled UNTRAINED DIAGNOSTIC, and cannot be called a profitable learned policy or enabled live.

This is NOT weight averaging and NOT prediction averaging. Multiple evidence packets preserve provenance and different horizons; learned fusion uses them jointly. A standalone inference proof must expose head readiness and selected experts so a completed forward pass cannot be confused with completed trading-policy training.

## Single GPU execution

1. CPU validates snapshot/schema/as-of history and forms router inputs.
2. Router selects eligible experts within top-k, predicted latency and memory budget. Cheap daily expert first when eligible; event model only for authentic event input.
3. Enter the existing scheduler's `work(role)` contract. Future live integration must inject the SAME scheduler instance into the SAME GPU-owner process; do not add competing locks.
4. Load one frozen expert on CPU, transfer it to GPU in supported inference dtype, run under `inference_mode`, copy its small evidence packet to CPU, synchronize, offload/release that model, then release the slot. No optimizer, gradient or activation checkpointing.
5. Repeat for next selected expert. Default residency is ONE expert, not all experts plus both old models. Offload controls ownership; `empty_cache` releases unused allocations only, not live tensors.
6. Fuse CPU packets / run the small fusion head, produce target weights, cash, action changes and selected/replaced symbols. Existing paper fills/reward generation would be a later, explicit integration step.

Weight bytes are only a lower bound. Report original bytes, optional FP16/BF16 weight estimate, actual measured allocated/reserved peak and device-wide free memory separately. Include attention/KV/sampling buffers and Windows display use in runtime admission. Never claim aggregate weights fit just because their sum is below 8GB.

## Upper checkpoint and acceptance

An upper manifest is possible immediately: expert IDs/revisions/hashes, source revisions/licenses, adapter schemas, router config, fusion/head readiness and hashes. A self-contained archive can later contain those originals plus manifest and head state; this is packaging, not weight merge. Reject modified/missing members. Shared original files avoid duplicating several GB for every Champion/Candidate save. Redistributability depends on each license, so a local package is not automatically publishable.

Acceptance for phase one: download every genuinely public requested artifact (record code-only/missing weights honestly); inspect actual files; preserve originals; new standalone inference with native experts, frozen checks, deterministic contract tests and no writes to existing runtime; report measured vs estimated memory, actual parameter totals, policies unavailable, and remaining learned-router/head work. No original architecture compatibility is assumed from matching advertised parameter counts.
