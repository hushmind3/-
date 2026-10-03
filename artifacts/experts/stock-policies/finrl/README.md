# FinRL_trading_bot (Phase 1.0)

> **Warning:**  
> This project is **for research and fun**. Use at your own risk.  
> No financial advice or guarantees.  
> Test thoroughly with paper trading before any real deployment.

## Overview

This repository contains a SMART live trading bot for multi-stock, multi-agent reinforcement learning (FinRL) with Tiger Brokers API integration.  
Key features include:

- **Multi-agent support:** Uses PPO, A2C, and SAC models (trained with [FinRL](https://github.com/AI4Finance-Foundation/FinRL)).
- **Live trading:** Connects to Tiger Brokers for real-time execution.
- **Daily stop-loss:** Prevents excessive losses by comparing account value to previous day.
- **Rolling Sharpe selection:** Dynamically picks the best-performing model.
- **Capital-aware execution:** Trades are limited by available cash and position sizes.
- **Comprehensive logging:** Logs trades, errors, and account status to `smart_trading_bot.log` and `smart_trading_bot_error.log` in  `log` folder.
- **Transaction history:** Saves daily transaction summaries in `transaction_actions/`.
- **NOT IDEAL FOR REAL TIME TRADING**

## Project Structure

```
automation/
log/
FinRL_models.ipynb
FinRL_trading_bot.ipynb
data/
    account_value_yesterday.txt
    trade_data.csv
    train_data.csv
trained_models/
    agent_a2c.zip
    agent_ppo.zip
    agent_sac.zip
transaction_actions/
    transaction_YYYY-MM-DD.txt
requirements.txt
```

## Requirements

- Python 3.10+
- [TigerOpen SDK](https://github.com/tigerfintech/openapi-python-sdk)
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL)
- Other dependencies: `pandas`, `numpy`, `yfinance`, `stable-baselines3`, etc.

Install dependencies:
```sh
pip install -r requirements.txt
```
or, for TigerOpen:
```sh
pip install tigeropen
```

## Usage

1. **Configure Tiger Brokers API:**  
   Set your `TIGER_ID` and `TIGER_PAPER_ACCOUNT` as environment variables.  
   Place your private key at the specified path.

2. **Train or load RL models:**  
   Place trained model files in `trained_models/`.

3. **Run the bot:**  
   Execute the notebook [`FinRL_trading_bot.ipynb`](FinRL_trading_bot.ipynb) in Jupyter or VSCode.

4. **Check logs:**  
   - Info: `smart_trading_bot.log`
   - Errors: `smart_trading_bot_error.log`
   - Daily transactions: `transaction_actions/transaction_YYYY-MM-DD.txt`
  
5. ** Configs in the scripts:** There are configuration in the scripts, please amend them accordingly 

## Disclaimer

- This bot is **experimental** and for research/testing only.
- No warranty or liability for financial loss.
- Always use paper trading for testing.

---

**Author:**  
Pei Qi
06 Sep 2025
