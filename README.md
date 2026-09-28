# StockRL Distill

PyTorch sample project for intraday/single-bar market decision research. It combines a weighted teacher ensemble, a GRU actor-critic, teacher distillation, and a compact PPO fine-tuning pass. Actions are **SELL / HOLD / BUY**; the scalar critic output is exposed as a state-value estimate. Fees are charged on position turnover in reward and backtests.

> This is a research and engineering example, not a profitable strategy or investment advice. The included heuristic policies are runnable demo teachers, **not** pretrained public market models. A generic pretrained stock-trading checkpoint cannot be assumed to match this project's observation shape/action semantics. The project accepts local Stable-Baselines3 PPO/A2C/DQN checkpoints through an explicit adapter; verify their input features, action meanings, and training provenance before use. FinRL is an open-source framework for training trading agents, not a universal compatible pretrained checkpoint repository.

## Requirements and setup

Python 3.10+ and PyTorch 2.2+. On Apple Silicon, create an arm64 environment and install the official MPS-enabled PyTorch wheel for your Python/macOS combination; `--device auto` selects MPS when available and otherwise CPU/CUDA.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

To load Stable-Baselines3 teachers as well:

```bash
python -m pip install -e '.[teachers]'
```

## End-to-end sample

```bash
python -m stockrl train --data data/sample_ohlcv.csv --epochs 2 --ppo-updates 1
python -m stockrl predict --data data/sample_ohlcv.csv --checkpoint checkpoints/final.pt
```

For Windows PowerShell run `python -m pip install -e .` then `python -m stockrl ...`. A training run writes `checkpoints/latest.pt` after each distillation epoch and PPO update, plus `checkpoints/final.pt` at completion. It prints validation metrics separately from the held-out final test backtest. The train/validation/test cuts are chronological 70/15/15 and no future return is used in features or teacher observations.

## Data format

CSV needs `date,open,high,low,close,volume` (column name case is ignored). The included tiny sample is synthetic solely for smoke checks. For real data, use a properly adjusted, timestamped intraday OHLCV dataset, sort/deduplicate timestamps, account for splits, and include realistic slippage, spread, fills, and market hours before drawing conclusions.

## Teacher configuration

Edit `configs/teachers.json`. Heuristic teachers support `trend`, `mean_reversion`, and `volatility_aware`; the nonnegative `weight` is the ensemble weight. These produce common 3-action logits plus a scalar value. Local SB3 examples:

```json
{"name":"local_ppo","type":"sb3","algorithm":"ppo","path":"models/ppo.zip","weight":1.5}
```

SB3 observations are flattened window feature arrays and actions must be discrete indices 0=SELL, 1=HOLD, 2=BUY. The adapter maps action predictions to common logits; SB3 critic values are unavailable through this normalized inference API and map to 0. Use the `Teacher` interface in `src/stockrl/core.py` to adapt any other PyTorch checkpoint. It is the user's responsibility to normalize teacher features/action/value semantics consistently. Teacher evaluation weights are configured explicitly; this project does not claim those weights are empirically validated.

## Commands

```bash
python -m stockrl train --data PATH.csv --teachers configs/teachers.json --device auto
python -m stockrl predict --data PATH.csv --checkpoint checkpoints/final.pt --device auto
```

This implementation is deliberately a minimal educational PPO loop, not a production execution engine. Inspect the source and add transaction/tax rules and a realistic market simulator for a live deployment workflow.

## 지속 관찰과 가상매매

초기 모델을 한 번 만든 뒤, 툴/피드가 최신 봉을 계속 추가하는 CSV를 감시할 수 있습니다.

```bash
python -m stockrl continuous --data path/to/live_bars.csv --champion checkpoints/final.pt
```

프로세스는 `Ctrl+C`까지 실행되며, 새 timestamp마다 판단을 기록하고 `runtime/latest_signal.json`을 원자적으로 갱신합니다. 신호의 `mode`는 항상 `paper`입니다. 이 JSON은 외부 자동매매 툴이 읽을 연동 경계입니다. 툴에서 사용자가 자동매매를 눌렀을 때 주문 실행은 그 툴이 담당하고, 모델 루프는 브로커 주문 API를 호출하지 않습니다. 실행 결과를 다시 학습하려면 툴이 `timestamp,action,reward` 열을 가진 CSV를 갱신하고 `--feedback path/to/feedback.csv`를 지정합니다. 실제 주문 체결과 손익 귀속은 툴에서 계산해 `reward`로 전달해야 합니다.

완료된 가상 판단은 지정 horizon 봉 뒤에 수수료를 차감해 replay에 쌓입니다. 최근 관측으로 만든 후보는 replay로 incremental update한 뒤, 해당 replay 학습 구간보다 뒤의 시간 검증 구간에서 기존 champion과 비교합니다. 검증 수익률이 개선되지 않으면 champion은 유지됩니다. 기록은 `runtime/paper_decisions.jsonl`, replay는 `runtime/replay.pt`에 저장됩니다. 과거 데이터를 가상 체결로 흘려보내는 확인 명령:

```bash
python -m stockrl replay --data data/sample_ohlcv.csv --checkpoint checkpoints/final.pt --state-dir runtime-replay
```

## 시장과 입력 호환성

보편적인 “모든 시장 호환” 기능은 없습니다. 미국/한국 주식, 선물, 가상자산 등에서도 **timestamp와 OHLCV 규약을 정규화한 바 데이터**는 공통 기반으로 처리할 수 있습니다. 시간대와 거래 세션, 액면분할/배당 조정, 종목 식별자, 가격 단위, 숏 가능 여부, 수수료·세금·슬리피지와 체결 모델은 공급자와 시뮬레이터 설정에 맞춰야 합니다. 기본 모델은 종목별 단일 시계열이므로 여러 종목/시장 혼합 학습에는 심볼/시장 식별 특징과 별도 검증이 필요합니다.

CSV 필수 열은 `date,open,high,low,close,volume`입니다. 선택 `bid,ask,bid_size,ask_size,buy_volume,sell_volume,trade_count` 열에서 호가 spread/imbalance와 체결 imbalance/intensity 특징을 만듭니다. 영상은 프레임 자체가 아니라 프레임 처리 단계가 추출한 수치로 입력합니다. 차트/거래량 영상에서 timestamp별 `video_chart_signal,video_volume_signal`을 추출해 CSV 행에 넣을 수 있게 열을 준비했습니다. 다른 트레이더 기록은 timestamp, 행동, 실현 손익을 시장 데이터와 정렬해 feedback CSV로 전달할 수 있습니다.

지속 루프는 파일에 새로운 timestamp가 추가되지 않으면 새 판단/학습을 하지 않습니다. 공급기는 확정된 봉을 append하거나 원자적 교체로 게시해야 합니다. 샘플 데이터만으로 실시간 피드, MPS 장기 실행, 외부 툴의 주문/피드백 연동을 검증할 수는 없습니다.

## 실제 공개 pretrained teacher 체크포인트

`models/teachers/`에 공개된 **학습 가중치 파일 6개**를 내려받아 저장했고, 전부 SB3/PyTorch로 로드해 Yahoo Finance 실제 MSFT 일봉 데이터에 대해 추론했습니다. source별 전처리·관측공간이 달라 각 모델 전용 adapter를 거쳐 공통 `SELL/HOLD/BUY` logits와 value 숫자로 바꿉니다. action probabilities는 공개 모델의 calibrated probabilities가 아니라, 공개 정책이 고른 행동에 큰 logit을 준 **distillation용 pseudo target**입니다. PPO 모델의 value는 critic 출력, DeepBio DQN의 value는 max Q, Pfizer recurrent model은 공개 추론 인터페이스에서 critic 값을 꺼내지 않아 0으로 기록합니다.

| 공개 모델 / 파일 | 구조·실제 입력 | 원래 행동 → 공통 행동 | 라이선스·비고 |
|---|---|---|---|
| [maksimprivalov/RLTradingAgent](https://github.com/maksimprivalov/RLTradingAgent), `maksimprivalov_ppo_trader.zip` (276,393 B) | SB3 PPO MLP; 90 float = 15×6 (`log_return,sma20,sma50,rsi14,macd,volume_change`) | Discrete 0 SELL, 1 HOLD, 2 BUY → 그대로 | 저장소 LICENSE 파일을 찾지 못함. 연구 추론용으로 다운로드. 공개 README는 daily OHLCV/MSFT를 설명. 실제 출력은 435행 중 HOLD 433, SELL 1, BUY 1이라 편향을 확인하고 가중치 1.0으로 낮은 영향의 teacher로 둠. |
| [jk2500/Pfizer-trader](https://github.com/jk2500/Pfizer-trader), `jk2500_pfizer_recurrentppo_lstm.zip` (28,532,063 B) | SB3-Contrib RecurrentPPO `MlpLstmPolicy`; (100,9) (`Open,High,Low,Close,Volume,Adj Close,MA10,MA50,RSI`) | Discrete 0 short, 1 long → SELL, BUY; 이 모델 자체에는 HOLD가 없음 | 저장소 archive에서 LICENSE 파일 미발견. 저장소 설명상 PFE 데이터에 학습. 입력 scaler는 별도 공개 파일이 없어 MSFT 앞 70%로 fit 후 전체 적용. LSTM hidden state를 bar 간 유지. |
| [Adilbai/stock-trading-rl-agent](https://huggingface.co/Adilbai/stock-trading-rl-agent), `adilbai_final_model.zip` (4,875,406 B) + `adilbai_scaler.pkl` (2,423 B) | SB3 PPO MLP; 60×50 standardized market features + 8 portfolio values = 3,008 float; scaler도 함께 받음 | 실제 SB3 action space는 `Box([action_type, size])`, 예측 첫 요소를 반올림해 0 HOLD, 1 BUY, 2 SELL로 처리하고 size로 paper portfolio 상태 갱신 | Hugging Face model card는 MIT. 카드에 연속/이산 출력 설명이 섞여 있으나 checkpoint action space를 직접 읽어 Box(2)임을 확인하고 adapter는 명시적으로 보수 변환. 공개 성능 수치는 자체 보고라 검증된 성과로 취급하지 않음. |
| [deepbiolab/drl-trading](https://github.com/deepbiolab/drl-trading), `checkpoint_deepbiolab.pth` (13,668 B), `model_fold_1_deepbiolab.pth` (13,692 B), `model_fold_2_deepbiolab.pth` (13,692 B) | PyTorch Double-DQN Q MLP `3→64→32→8→3`; window 1의 정규화된 연속 차이값 (`Close,BB_upper,BB_lower`) | Q action 0 HOLD, 1 BUY, 2 SELL → HOLD, BUY, SELL | MIT. checkpoint 3개가 실제 state dict임을 로드/forward로 확인. 공개 저장소는 AAPL 데이터·기술지표를 설명. 원본 normalizer artifact가 없어 adapter는 MSFT 앞 70%에서 열별 정규화값을 fit하고 이후 시점에 적용. |

정리하면 **4개 공개 프로젝트에서 6개 체크포인트**를 연결했습니다: PPO 계열 2개, LSTM recurrent PPO 1개, DQN 3개. 이들은 HFT/scalping 전용이라고 주장할 수 없습니다. 확인된 source는 일봉/단일 종목 또는 포트폴리오 정책이며, sub-minute 주문장 전략과는 다릅니다. 별도 공개 checkpoint를 찾지 못한 A2C 전용 모델은 등록하지 않았습니다.

다운로드 후 생성한 출력:

- `outputs/public_teachers/<teacher>.csv`: MSFT 각 435 timestamp에서 모델별 행동, 세 행동 pseudo probability, value
- `outputs/public_teachers/ensemble.csv` 및 `distillation_targets.npz`: 여섯 교사의 균등 가중 앙상블 target
- `checkpoints/public_distilled/final.pt`: 실제 MSFT 시계열의 앞 70%에서 2 epoch distillation + 1 PPO update를 수행한 새 학생 체크포인트
- 테스트 구간(마지막 15%)의 단순 일봉 backtest: 이 실행에서는 누적 수익률 **-34.11%**, 최대 낙폭 **36.42%**. 이것은 모델 품질이 입증됐다는 결과가 아니라 실행 검증이며, 성과는 음수였습니다.

실행 명령:

```bash
python -m pip install -e '.[teachers]'
python -m stockrl teacher-dataset --data data/msft_real_daily.csv --teachers configs/teachers.json --output outputs/public_teachers
python -m stockrl train --data data/msft_real_daily.csv --teachers configs/teachers.json --checkpoint-dir checkpoints/public_distilled --epochs 2 --ppo-updates 1 --device auto
```

### 다운로드했지만 teacher로 쓰지 않은 공개 후보

- [xinghao2003/fyp Hugging Face dataset](https://huggingface.co/datasets/xinghao2003/fyp): dataset/model-results 설명은 있으나 공개 GitHub archive와 HF dataset 파일 확인에서 직접 로드할 RL checkpoint가 없어 제외.
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL): PPO/A2C/DDPG 등의 학습 코드와 환경은 제공하지만 이 프로젝트에 바로 붙일 공개 pretrained checkpoint 파일은 확인하지 못해 checkpoint teacher로 등록하지 않음.
- A2C/HFT/scalping 키워드의 여러 저장소는 코드만 있거나 실제 weight가 저장소에 없거나, action/observation 계약을 재구성할 파일이 없어 이번 실행 registry에서는 제외. 공개 체크포인트를 확인하지 않은 알고리즘을 연결했다고 세지 않았습니다.

이번에 검토한 후보 중 라이선스 표기만을 이유로 다운로드/연결에서 제외한 모델은 없습니다. 위 두 저장소는 LICENSE 파일을 찾지 못했지만 요청한 연구용 범위로 체크포인트를 받아 실행했고, 그 상태를 manifest/표에 그대로 표시했습니다. 저장소 archive에서 weight 자체를 찾지 못한 경우와 모델이 실제로 호환되지 않는 경우는 라이선스 이슈와 구분해 제외했습니다.

실제 가격 파일 `data/msft_real_daily.csv`는 Yahoo Finance chart data에서 받아 2024-01-02부터 2025-09-25까지 유효한 435개 일봉으로 정리했습니다. 이는 샘플 합성 데이터가 아니지만, 매매 빈도는 daily입니다. teacher가 학습된 종목·기간·정규화와 MSFT 2024–2025가 달라 분포 이동이 크므로 결과는 teacher 인터페이스의 실제 실행 검증으로만 해석해야 합니다. Adilbai scaler는 scikit-learn 1.2.2에서 pickle 된 파일로 확인되어 현재 버전에서 호환 경고가 나오지만 로딩/추론은 완료했습니다.

## 글로벌 continual Transformer 에이전트

기존 `TemporalActorCritic` GRU는 작은 종목별 비교 baseline으로 남겨두고, 글로벌 정책은 `src/stockrl/global_transformer.py`의 시간축/시장축 교차 attention Transformer를 사용합니다. 기본 모델의 정확한 크기는 **511,848,836 parameters**입니다: width 1,408, 16 heads, 21개 교차 Transformer block, 17개 시장 특징, 상품/시장/자산/시간 embedding, SELL/HOLD/BUY 정책 head와 value head. 설정을 `d_model=1792, n_layers=26, n_heads=16`으로 확장하면 약 1B급이 됩니다. 실행 기본값을 소형으로 바꾸지 않습니다.

CUDA에서는 모델과 optimizer를 FP16으로 실행합니다. RTX 3070 8 GiB 측정에서 학습 peak allocated VRAM은 6,391,110,656 bytes였고, 저장되는 champion FP16 weight는 약 1.02 GB입니다. AdamW half-state를 쓸 때 epsilon을 1e-4로 두어 NaN을 막습니다. Apple Silicon은 기존 `device=auto` 경로에서 MPS를 선택하고 FP32로 실행합니다. 실제 MPS 장치에서 이번 실행 검증은 하지 않았습니다.

### 실제 글로벌 데이터 받기

다음 명령은 Yahoo Finance chart의 실제 일봉 OHLCV를 `data/global_market_daily.csv`로 저장합니다. 실행 검증 데이터는 2023-01-02~2026-09-25, 52개 상품, 47,860행입니다.

```powershell
python scripts/download_global_data.py --output data/global_market_daily.csv --start 2023-01-01
```

포함 상품군은 한국 주식/지수, 미국 주식·지수·지수선물·섹터 ETF, 유럽/일본/홍콩/호주/인도 지수, 미국 국채금리, 채권 ETF, 통화쌍, 원유·천연가스·곡물·금·은·구리 선물, VIX 및 상품 ETF입니다. 이는 글로벌 상품의 **일봉 가격 패널**이지 옵션 체인, 실시간 호가/체결 피드가 아닙니다. 옵션·호가 입력 열은 어댑터에 준비되어 있으며 아래 CSV 칼럼을 실시간/외부 provider에서 제공하면 변환됩니다.

필수 칼럼: `date,symbol,market,asset_class,open,high,low,close,volume`. `date`는 일봉이면 날짜, 실시간 feed이면 UTC ISO timestamp를 씁니다. 선택 칼럼은 `bid,ask,bid_size,ask_size,buy_volume,sell_volume,trade_count,implied_volatility,open_interest,yield_change,days_to_expiry,video_chart_signal,video_volume_signal`입니다. 다른 트레이더 기록 및 영상에서 뽑은 신호도 이 특징 열과 별도 `date,symbol,action` teacher CSV로 정규화해 넣을 수 있습니다. 미제공 옵션/호가 값은 0으로 채워지므로 실제 입력이 있는 것처럼 취급하면 안 됩니다.

### GPU 설치와 실행

Windows RTX GPU에서는 CUDA wheel을 먼저 설치합니다. 아래는 이 환경에서 검증한 PyTorch 2.14.0 CUDA 13.2 명령입니다. Apple Silicon Mac은 이 CUDA 명령 대신 `python -m pip install -e '.[teachers]'`를 사용합니다.

```powershell
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
python -m pip install -e ".[teachers]"
python -m stockrl global-info --data data/global_market_daily.csv --device auto
```

실시간 feed 프로세스는 위 형식의 append-only CSV에 모든 시장 bar를 timestamp별로 추가합니다. 아래 명령은 매 timestamp 관찰/판단, horizon 뒤 가상 손익·수수료·slippage 반영, replay 저장을 수행하며 learner는 별도 thread에서 업데이트합니다. 종료는 Ctrl+C입니다. 이 CSV 연결은 feed adapter이지 거래소 구독기나 주문 API가 아닙니다.

```powershell
python -m stockrl global-online --data data/global_live.csv --state-dir runtime-global --follow --poll-seconds 1 --device auto
```

공개 teacher 판단을 먼저 만들고 글로벌 에이전트의 imitation replay에 추가할 수 있습니다. 파일에는 `date,action` 칼럼이 있어야 하며 기본 teacher symbol은 MSFT입니다.

```powershell
python -m stockrl teacher-dataset --data data/msft_real_daily.csv --teachers configs/teachers.json --output outputs/public_teachers
python -m stockrl global-online --data data/global_live.csv --state-dir runtime-global --follow --teacher-decisions outputs/public_teachers/ensemble.csv --teacher-symbol MSFT --device auto
```

실제 teacher 연결 확인에서는 public teacher ensemble 435행 중 시간순 학습 구간의 404행을 imitation replay에 넣었습니다. 총 10개 짧은 update 중 2개는 teacher imitation, 8개는 paper 경험이었고 weight가 바뀌었습니다. 해당 산출물은 `runtime-global-teacher-final/metrics.json`입니다.

### 온라인 학습 검증 결과

분리는 시간순 앞 70% replay 학습, 다음 15% champion 검증, 마지막 15% 별도 backtest입니다. RTX 3070에서 합성 입력이 아닌 `data/global_market_daily.csv`로 전체 과정을 실행했습니다. 52개 시장 상품에서 관찰 10회, 판단 492건, 지연 outcome/reward 492건이 replay에 들어갔고 replay 파일은 9,005,607 bytes였습니다. 한 update 뒤 가중치 L1 변화량 2,139.50을 확인했습니다. candidate net validation score `0.000108`이 champion `-0.006814`보다 높아 이 실행에서는 실제 승격됐고, 이전 champion을 `champion.previous.pt`로 보관했습니다. 다른 실측 실행에서는 candidate score가 낮아 자동 기각하고 champion을 유지했습니다.

계측: inference p50 **29.5 ms** (cold-start 포함 p95 122.4 ms), batch 1의 1-step update **0.717 s**, peak VRAM **6.39 GB / 8.59 GB**, process peak RSS **2.46 GB**. checkpoint 저장·재로딩에 성공했고 출력 shape은 `[1,52,3]`, parameter count도 동일했습니다. 마지막 15% backtest는 253 symbol-time 판단/6단계에서 net return **-0.123%**, max drawdown **0.123%**였습니다. 짧은 smoke 구간 성적이지 전략 수익성의 증거는 아닙니다. CUDA 실행에 사용한 `torch==2.14.0+cu132`, GPU `NVIDIA GeForce RTX 3070`입니다. 교체 기준은 수수료·slippage를 포함한 시간순 validation net return입니다.

산출물은 `runtime-global-verified/metrics.json`, `decisions.csv`, `backtest.csv`, `replay.pt`, `champion.pt`, `candidate.pt`입니다. follow CSV 모드는 `live_cursor.json`, 미결 가상거래 `live_pending.pt`, 현재 target position `live_positions.json`을 함께 저장해 재시작 시 이어갑니다. replay capacity 기본값은 100,000건이며 teacher가 없을 때 recent 50%, old 30%, 극단 변동/보상 20%를 섞습니다. teacher가 있으면 10%를 teacher imitation 예제로 할당합니다. 후보 업데이트는 새 경험이 256건 쌓일 때마다 실행하고 전체 과거 데이터를 epoch 재학습하지 않습니다. 과거 teacher row는 historical run에서 첫 70%에만 넣어 검증/테스트 구간 누수를 막습니다.

All promotions and rejections remain research-only paper evaluation. The dedicated desktop now connects Yahoo chart and Kraken public OHLC minute bars; it does not provide full order-book/option-chain feeds or a real broker. Live order routing requires an explicitly installed broker adapter and is OFF by default.

## Dedicated desktop control panel and live feed

The fixed 0.5B Transformer and existing online agent remain unchanged in shape. The Windows desktop panel supervises the public-feed process and the `global-online --follow` process. The latter keeps champion inference on its observer loop while a separate learner thread updates and validates candidates. The GUI is a separate process; stopping/restarting it does not discard replay, champion, cursor, pending paper outcomes, validation samples, positions, or settings.

Install the GUI dependency once, then launch the panel with one command:

```powershell
python -m pip install -e ".[desktop]"
python -m stockrl desktop --device auto
# or scripts\launch_global_agent.bat
```

The default desktop profile is `runtime-global-desktop/live` and starts the public live-data collector when you press Start. Select `Historical real data (mock feed)` to replay recent rows from `data/global_market_daily.csv` at a controlled pace. The panel shows provider/process status, latest per-symbol BUY/HOLD/SELL probabilities and value, champion checkpoint time, replay count, paper reward, candidate activity, promotion/rejection counts, CUDA device/VRAM, and tracked paper positions. Start, graceful stop, and emergency stop are available on the same screen. Runtime profiles are separate from `runtime-global-verified`; the verified model is copied into a new desktop profile on first start, never overwritten.

The collector polls Yahoo Finance chart bars and Kraken public OHLC. Providers implement `MarketDataProviderAdapter.fetch(instrument)` and register through `register_market_data_provider`; extra modules can be loaded with `STOCKRL_MARKET_PROVIDER_PLUGINS=module1,module2` or repeated `--provider-plugin` flags. It appends finalized bars with canonical UTC timestamps into the CSV consumed by `--follow`, uses a SQLite unique-key index across restarts, records provider errors, and backs off/retries failed symbols. Yahoo chart is an unauthenticated best-effort endpoint, not a guaranteed streaming contract. The configured universe is 36 instruments: Korean and US shares, major indices and equity-index futures, FX, US Treasury yield indices, crude/gold/silver/copper futures, VIX, and BTC/ETH. Public free feeds do not provide a uniform real-time options-chain/history service here; option IV/open interest fields stay empty unless a later adapter supplies them. Yahoo and Kraken minute candles do not provide full order-book depth or trade-side classification. Market data availability, exchange hours, provider rate limits, and delayed/stale bars vary by instrument. The collector reconnects after request errors but cannot manufacture bars while an exchange is closed. After adding the completed-candle guard, a real Yahoo/Kraken check appended 2,397 AAPL/BTC/ETH bars with zero provider failures; a second fetch appended zero duplicate timestamps.

The run/stop controls start or stop both provider and agent processes. The stop request is persisted via a local marker; the learner completes its current update, the agent saves champion/replay/pending experiences/cursor/positions, and the provider closes its SQLite index. Unexpected child exit triggers a restart from saved files. Repeating `python -m stockrl desktop` opens the same settings and profile. Candidate updates default to each 256 additional replay experiences; this can be changed in the desktop panel. Candidate training does not block feed polling or champion inference. Delayed rewards mature after the configured number of observed bars for that symbol, so a quiet stock market is not rewarded at an unrelated FX/crypto timestamp. Promotion still requires a higher held-out net validation score; the former champion is saved as `champion.previous.pt`.

Live order routing is OFF by default. This project does not include an actual brokerage integration. A broker plugin must subclass `stockrl.broker.BrokerAdapter`, set `is_live = True`, implement `supports_symbol`, `size_order`, `connect`, `place_order`, `emergency_stop`, and `close`, and be configured through `STOCKRL_LIVE_BROKER_ADAPTER=module:Class`. The GUI then requires a yes/no confirmation plus typing `ENABLE LIVE ORDERS`; unsupported symbols and any order whose adapter returns a nonpositive risk-limited size are not routed. The bundled `MockBroker` never sends orders. A connected broker must handle its own market-specific lot size, price precision, risk limits, authentication, and kill switch before it is appropriate for live use.

Desktop and collector checks performed on Windows with CUDA PyTorch 2.14.0+cu132 and an RTX 3070: the public collector fetched 26,015 one-minute OHLCV rows across all 36 configured instruments on its first pass. A second pass wrote 2 new rows and zero duplicates; all timestamps parsed as UTC and `(symbol,date)` duplicates were zero. The market CSV was then watched by `global-online --follow`: 14 observations produced 28 predictions, 26 delayed outcomes, and 22 saved replay items. One candidate gradient update ran on CUDA; its weights changed by L1 3.503, update time was 8.00 seconds, while warmed champion inference p50 was about 22 ms. Model device was `cuda` / `NVIDIA GeForce RTX 3070`, with 6.42 GB peak allocated VRAM. This verified live-feed-to-agent behavior is paper research plumbing, not evidence of strategy profitability or order execution. The historical sample backtest recorded earlier remains negative (-0.123% on its held-out slice).

The dashboard was run offscreen with genuine historical market rows played at 0.15 seconds per bar. It observed 24 timestamps, displayed 1,180 per-symbol decisions, matured 1,136 outcomes, and saved 772 replay examples. The learner performed 2 candidate updates; both failed validation and champion remained unchanged. Candidate weight deltas L1 were 0.780 and 3.867, warmed inference p50 was 24.2 ms, candidate update p50 was 0.385 s, and observed peak CUDA allocation was 6.39 GB. The replay paper reward sum was -0.11% for this short smoke sample. The UI showed 80 latest decision rows, replay count, GPU, candidate and live-order-OFF states. A clean stop then a second GUI start restored replay (772), update count (2), and the same market cursor; decisions remained 1,181 rows and the restarted mock feed appended zero duplicate bars. The saved run is in `runtime-global-desktop-verified2/mock/agent/`; `metrics.json`, `replay.pt`, `champion.pt`, `live_cursor.json`, and `live_validation.pt` can be inspected there.

A second slower replay explicitly checked concurrency while new bars were still arriving. By the first candidate's completion the agent had observed 29 timestamps, made 1,447 decisions, matured 1,395 outcomes, and saved 1,179 replay examples. Seventeen additional champion inferences occurred while `candidate_training` was true, and the mock-feed process was still running as rows were appended. The 0.5B candidate completed one update (0.759 s), changed weights by L1 3.589, then failed validation and was rejected; champion inference continued. Inference p50 was 27.7 ms, peak CUDA allocation 6.40 GB, and the clean-stop state shows `candidate_training: false`. Its files are in `runtime-global-desktop-concurrency/mock/agent/`. The standalone MockBroker queue check returned a simulated fill with `live_order: false`. No real broker is configured, and there is no exchange websocket subscription or full option-chain/order-book feed in the bundled default providers.

## ������ ��ú��� (���� ����)

Windows���� �� �� �����ϸ� ���� �� ��ú���, ���� �ü� ������, ���� champion ���� �Բ� ���۵ǰ� ������ â�� �����ϴ�. �⺻ ������ �ֹ� OFF paper ����Դϴ�.

```powershell
python -m pip install -e .
python -m stockrl web
```

�Ǵ� ���� Ž���⿡�� `scripts/launch_global_web.bat`�� �����ϼ���. ��ú���� `http://127.0.0.1:8765`���� �����ϴ�. `�ǽð� �ü�` �Ǵ� `���� ������ ���`�� ��� ������ �� �ְ�, ����/������� �� ���� ������ ��û�մϴ�. �� ȭ���� ����� ���� �°� ���������� �����߽��ϴ�. ���� ���´� `runtime-global-web/`�� ���� �����ϹǷ� ���� ���� runtime/champion�� ����� �ʽ��ϴ�. â���� ���ŵǴ� �׸��� �ü�/�� ����, champion, �ֱ� BUY/HOLD/SELL �Ǵ� �� Ȯ��/��ġ, ���� ������, reward, replay, candidate update�� �°�/�Ⱒ ���, GPU/VRAM�Դϴ�.

���� PC�� �޴��� ���������� ������ LAN ���ῡ�� `python -m stockrl web --host 0.0.0.0`�� �����ϰ� PC�� �缳 LAN �ּҸ� ������. ���� ������ ����/HTTPS�� �����Ƿ� ���ͳݿ� ���� �����ϰų� Ŭ���忡 �ٷ� �������� ������. Ŭ���� ��� ����, HTTPS reverse proxy, persistent volume�� �տ� �־� �մϴ�. �ǽð� �����ʹ� ���� ���� API�� best-effort �����̸�, ���� ���Ŀ �ֹ� ����� ���ǿ� �������� �ʾҽ��ϴ�.

������ health/status Ȯ��: `GET /api/status`; ��Ʈ��: `POST /api/start` (`{"mode":"live"}` �Ǵ� `{"mode":"mock"}`), `POST /api/stop`.

## Shared multi-market training loader

`src/stockrl/market_training.py` defines a common loader for multiple market sources. Source-specific adapters normalize UTC events/bars into one schema; one loader handles symbol IDs, chronological windows, coverage-aware symbol sampling, whole-universe context, target returns, and 70/15/15 time splits. The normalized adapter supports equities, ETFs, futures, options, rates, and crypto. Optional fields include bid/ask, trade flow, implied volatility, open interest changes, yield changes, days to expiry, and video-derived chart/volume signals. Missing values map to the existing 17-feature input. The first source-specific adapter is `ITCHSnapshotAdapter` for corrected Nasdaq ITCH snapshots.

The loader assigns deterministic contiguous instrument IDs, records per-symbol exposure, samples mostly from least-exposed symbols, and reserves a volume-weighted portion. Each sequence receives 16 global context statistics computed once from the full current universe. The original 0.5B backbone and champion stay intact; the isolated candidate adds a zero-initialized context residual and an expanded symbol embedding initialized from the legacy embedding. The dry-run never promotes or overwrites the champion.

After the official-spec ITCH 5.0 reparse completes, run the CUDA dry-run and 64/128/256 symbol-group benchmark on the RTX 3070:

```powershell
python -m pip install -e .
python scripts/bench_market_training_loader.py
```

Outputs go under `runtime-global-market-training/`: disk-backed panel, symbol map, complete exposure CSV, measured metrics, and an isolated one-update candidate. The command does not start long training and never writes to `runtime-global-cuda-final/champion.pt`.
### RTX 3070 dry-run results (2026-09-26)

The full corrected ITCH snapshot CSV was loaded through the shared adapter and loader. The observed universe was **8,694 instruments / 49,446 UTC time rows**; the memory-mapped shared panel occupies **16,765,457,436 bytes (15.62 GiB)**. Its splits are strictly chronological: train `2019-01-30 09:00:00`–`19:08:45 UTC`, validation `19:08:46`–`21:12:28 UTC`, test `21:12:29`–`2019-01-31 01:00:00 UTC`. Inputs have 17 per-instrument features and 16 full-universe context features at sequence length 128.

Measured one-step CUDA forward/backward/update on the NVIDIA GeForce RTX 3070:

| Symbols/sample | Input tensor | Step | Symbols-windows/s | Peak allocated VRAM | Process RSS |
|---:|---|---:|---:|---:|---:|
| 64 | `[1,128,64,17]` | 1.269 s | 50.42 | 2.43 GiB | 3.39 GiB |
| 128 | `[1,128,128,17]` | 2.256 s | 56.74 | 2.95 GiB | 3.50 GiB |
| 256 | `[1,128,256,17]` | 4.126 s | 62.04 | 4.82 GiB | 3.85 GiB |

The run's recorded peak process RSS was 4.88 GiB; panel-build files remain memory-mapped on disk. Direct IDs are unique for all 8,694 instruments. The previous 8,192-way ticker hash would collide for 3,357 IDs in this universe. The sample run selected 575 distinct instruments across its three benchmark batches; 8,119 had zero exposure in this intentionally short dry-run, so full coverage is a property to accumulate during extended sampling/training, not something claimed by these three batches.

An isolated candidate completed an optimizer update and was saved/reloaded; reloaded outputs were finite with logits `[1,128,3]` and values `[1,128]`. The 30-second held-out validation minibatch had 11 reward-valid instruments and mean expected net return of `-0.00128370` for champion versus `-0.00128072` for candidate. This single tiny minibatch is **not** adequate evidence of improvement; the candidate was not promoted. Original champion SHA-256 remains `4100da96158c777426008f6c1a874951de160c17e4a5566f7dda4743a3c36128` and it was not overwritten. Results are saved in `runtime-global-market-training/dryrun_metrics.json`, and per-symbol exposure in `runtime-global-market-training/symbol_exposure.csv`.

This validates the loader and bounded CUDA dry-run on corrected ITCH snapshots; it is not broad-market training or proof of profitable policy. The included normalized-source adapter is the common entry point for other market feeds; provider-specific source parsers and longer multi-regime training/evaluation still need to be added before claiming cross-market readiness.