# Reinforcement Learning Trading Agent

A reinforcement learning agent that learns to trade a single stock using daily OHLCV data and engineered technical indicators. Built with PPO on a custom Gymnasium environment, benchmarked against Buy & Hold.

> **Status:** Research project. The goal is to explore how RL behaves in a noisy financial environment, not to build a production trading system.

## What it does

The agent observes a rolling window of the last N days of market features and decides each day whether to **BUY**, **HOLD**, or **SELL**. It's trained with **PPO (Proximal Policy Optimization)** from `stable-baselines3` and evaluated on an unseen out-of-sample period, with equity growth compared against a passive Buy & Hold strategy on the same asset.

Default asset: **MSFT**, daily data from 2015 to 2025 via `yfinance`.

## Why this project

I built this to understand three things hands-on:

1. **How to frame a financial problem as an MDP** - what the state, action, and reward should actually be when a bad reward design silently breaks the whole thing.
2. **Why RL for trading is genuinely hard** - sparse signal, non-stationary markets, easy-to-miss data leakage.
3. **How to build an ML pipeline end-to-end** - data ingestion, feature engineering, environment design, training, evaluation, and honest benchmarking.

## Architecture

```
RLTradingAgent/
├── agent/
│   ├── features.py      # yfinance download + technical indicators
│   ├── prepare_data.py  # Time-based train/val/test split saves CSVs
│   ├── env.py           # Custom Gymnasium TradingEnv
│   ├── train.py         # PPO training loop with TensorBoard logging
│   └── eval.py          # Out-of-sample evaluation vs Buy & Hold
├── data/                # Generated CSVs (train/test/val)
├── logs/                # TensorBoard logs
├── documentation.ipynb  # Walkthrough notebook
└── ppo_trader.zip       # Saved model checkpoint
```

### Pipeline flow

```
yfinance (OHLCV) -> features.py (indicators) -> prepare_data.py (time split) -> train.py (PPO) -> eval.py (vs Buy&Hold)
```

## Environment design

The custom `TradingEnv` is where most of the interesting decisions live.

**Observation space:** `window_size × 6` flattened vector of the last N days of:
- `log_return` - daily log returns
- `sma20`, `sma50` - short and long simple moving averages
- `rsi14` - Relative Strength Index (overbought/oversold signal)
- `macd` - Moving Average Convergence Divergence
- `volume_change` - daily volume percentage change

**Action space:** discrete, 3 actions - `SELL (0)`, `HOLD (1)`, `BUY (2)`.

**Position sizing:** all-in / all-out. Simplification on purpose - it keeps the action space small so the agent can actually learn something in reasonable time before I complicate it with fractional sizing.

**Transaction costs:** fixed `0.1%` fee applied to both buys and sells. Slippage and market impact are not modeled (a known limitation for any realistic deployment).

### Reward shaping

Reward is a weighted sum of several signals rather than raw PnL. Pure PnL gave an agent that learned to do nothing, because "nothing" never loses money on a fee. The current shape:

| Component | Purpose |
|---|---|
| `+10 × equity_change` | Main profitability signal |
| `+10 × trade_profit` on sell | Explicit reward for closing a winning trade |
| `+3 × log_return` when holding a position | Reward for being in the market during upward moves |
| `−0.8` for invalid actions (buy with no cash, sell with no shares) | Prevent degenerate policies |
| `−0.8` for holding with no position | Discourage passive flat sitting |
| `−0.2 × \|log_return\|` | Small volatility penalty |
| Small action bonuses/penalties | Exploration vs churn balance |

This is pragmatic, not principled - reward shaping for trading is an open problem, and I iterated on these coefficients based on training curves and agent behavior on the validation set.

## Data handling

**Time-based split** (critical for avoiding lookahead bias):

| Split | Period | Purpose |
|---|---|---|
| Train | 2015-01 -> 2020-12 | Policy learning |
| Test | 2021-01 -> 2023-12 | Out-of-sample evaluation |
| Validation | 2024-01 -> 2024-12 | Reward tuning, sanity checks |

No shuffling. Indicators are computed before the split so the agent never sees future data - but the split itself is strictly chronological, which is the only honest way to evaluate a trading strategy.

## Results

The agent is benchmarked against Buy & Hold on MSFT test data (2021-2023), starting from the same initial capital. Evaluation plots the two equity curves and reports the action distribution.

> **Note:** The agent is competitive with Buy & Hold in some training runs and underperforms in others - results are sensitive to the reward coefficients and random seed. Beating a strong upward-trending stock like MSFT with a long-only discrete policy is hard by design, and that was part of the lesson. See "What I learned" below.

TensorBoard tracks `ep_rew_mean`, `policy_loss`, `value_loss`, and `entropy_loss` during training - the script auto-launches it after evaluation.

---

## Tech stack

- **Python 3.13**
- **stable-baselines3** - PPO implementation
- **Gymnasium** - environment API
- **yfinance** - market data
- **pandas, numpy** - data handling and indicators
- **matplotlib** - equity curve plots
- **TensorBoard** - training diagnostics

---

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data and generate train/test/val splits
python agent/prepare_data.py

# 3. Train the PPO agent (~100k timesteps, logs to ./logs/)
python agent/train.py

# 4. Evaluate on test set and compare to Buy & Hold
python agent/eval.py
```

For a full walkthrough of the code and design decisions, see [`documentation.ipynb`](documentation.ipynb).

## What I learned

- **Reward design dominates everything else.** I spent more time tuning reward coefficients than tuning PPO hyperparameters. A naive "reward = PnL" agent learned to never trade. A reward with too much bonus for action learned to churn and die by fees. Getting a useful policy required carefully weighting immediate vs delayed signals.
- **Data leakage is subtle.** My first version computed indicators per-split, which meant the 20-day SMA at the start of the test set was wrong. Fix: compute indicators on the full series, *then* split.
- **Buy & Hold is a very strong baseline on a bull-market single stock.** An RL agent has to earn its keep by doing better on a risk-adjusted basis, not just total return. A fair next step is to evaluate on a sideways or bearish asset.
- **Discrete all-in/all-out actions are a big simplification.** Real trading involves position sizing, and the agent currently can't express "I'm somewhat confident, take a small position."
- **100k timesteps is too few for stable policies on this problem.** Training curves still wobble. Longer training with early stopping on validation equity is a known improvement.

## Known limitations & future work

**Limitations**
- Single asset, long-only, all-in/all-out - no portfolio, no shorting, no sizing
- Fixed transaction fee only - no slippage, no market impact model
- Results are seed-sensitive; no multi-seed evaluation yet
- No walk-forward validation; test period is a single contiguous block

**Next steps I'd want to try**
- Continuous action space for fractional position sizing
- Multi-asset environment with portfolio constraints
- Walk-forward evaluation across multiple market regimes
- Compare PPO against a simple baseline policy (e.g., RSI threshold or SMA crossover) - not just Buy & Hold
- Sharpe ratio and max drawdown as evaluation metrics, not just raw equity

## References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [yfinance](https://pypi.org/project/yfinance/)
- Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347) (2017)

---

*Built as a self-directed project to explore the intersection of reinforcement learning and financial markets. Not intended as investment advice or a production trading system.*