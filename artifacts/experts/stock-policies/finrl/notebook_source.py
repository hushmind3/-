## install required packages
!pip install swig
!pip install wrds
!pip install pyportfolioopt
## install finrl library
!pip install git+https://github.com/AI4Finance-Foundation/FinRL.git

import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
import itertools
check_and_make_directories([TRAINED_MODEL_DIR])



from datetime import datetime, timedelta

# Set the end date as today
TRADE_END_DATE = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

# Set the trade start date as 3 months before the end date
TRADE_START_DATE = (datetime.today() - timedelta(days=300)).strftime('%Y-%m-%d')

# Set the train end date as the trade start date
TRAIN_END_DATE = TRADE_START_DATE

# Set the train start date as 5 years before the train end date
TRAIN_START_DATE = (datetime.strptime(TRAIN_END_DATE, '%Y-%m-%d') - timedelta(days=5*365)).strftime('%Y-%m-%d')

print("TRAIN_START_DATE:", TRAIN_START_DATE)
print("TRAIN_END_DATE:", TRAIN_END_DATE)
print("TRADE_START_DATE:", TRADE_START_DATE)
print("TRADE_END_DATE:", TRADE_END_DATE)

symbols = ['aapl', 'amd', 'amzn', 'cat', 'crwd', 'googl', 'gs', 'hd', 'ibm',
       'intc', 'meta', 'msft', 'nvda', 'pypl', 't', 'tsla', 'v']

df_raw = YahooDownloader(start_date = TRAIN_START_DATE,
                                end_date = TRADE_END_DATE,
                                ticker_list = symbols).fetch_data()

df_raw.head()

# Step 2: Get the last available date
last_date = df_raw['date'].max()

# Convert last_date to pandas Timestamp if it's a string
if isinstance(last_date, str):
	last_date_dt = pd.to_datetime(last_date)
else:
	last_date_dt = last_date

# Step 3: Filter all rows from the last date (usually one per ticker)
last_day_data = df_raw[df_raw['date'] == last_date].copy()

# Step 4: Duplicate and increment the date by 1 day
next_day = last_date_dt + pd.Timedelta(days=1)
last_day_data['date'] = next_day.strftime('%Y-%m-%d') if isinstance(last_date, str) else next_day

# Step 5: Append the duplicated rows to the original data
df_patched = pd.concat([df_raw, last_day_data], ignore_index=True)

df_patched

fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = INDICATORS,
                     use_vix=True,
                     use_turbulence=True,
                     user_defined_feature = False)

processed = fe.preprocess_data(df_patched)

df_raw['date'].max()  # Check the latest date in the raw data

processed['date'].max()  # Check the latest date in the processed data

processed.tic.unique()


# Align to all dates and tickers
list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
combination = list(itertools.product(list_date, list_ticker))
processed_full = pd.DataFrame(combination, columns=["date", "tic"]) \
    .merge(processed, on=["date", "tic"], how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date','tic'])
processed_full = processed_full.fillna(0)

# Make sure 'date' column is in datetime format
processed_full['date'] = pd.to_datetime(processed_full['date'])

# Sort first for consistency
processed_full = processed_full.sort_values(by=['date', 'tic']).reset_index(drop=True)

# Assign the same index to all rows with the same date
processed_full.index = processed_full.groupby('date').ngroup()

print("Final processed full shape:", processed_full.shape)
print("Final processed full max date:", processed_full['date'].max())

processed_full.tic.unique()

processed_full['date'].max()

# Split the data
train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)
print(len(train))
print(len(trade))

train_path = './data/train_data.csv'
trade_path = './data/trade_data.csv'

with open(train_path, 'w', encoding = 'utf-8-sig') as f:
  train.to_csv(f)

with open(trade_path, 'w', encoding = 'utf-8-sig') as f:
  trade.to_csv(f)

train = pd.read_csv(train_path)

train = train.set_index(train.columns[0])
train.index.names = ['']

stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 20,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}


e_train_gym = StockTradingEnv(df = train, **env_kwargs)

train

env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

agent = DRLAgent(env = env_train)

# Set the corresponding values to 'True' for the algorithms that you want to use
if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True

model_a2c = agent.get_model("a2c")
model_ppo = agent.get_model('ppo')

if if_using_a2c:
  # set up logger
  tmp_path = RESULTS_DIR + '/a2c'
  new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  # Set new logger
  model_a2c.set_logger(new_logger_a2c)

if if_using_ppo:
  # set up logger
  tmp_path = RESULTS_DIR + '/ppo'
  new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
  # Set new logger
  model_ppo.set_logger(new_logger_ppo)

import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
	print(torch.cuda.get_device_name(0))
else:
	print("CUDA is not available or PyTorch is not compiled with CUDA support.")
	print(torch.__version__)

trained_a2c = agent.train_model(model=model_a2c,
                             tb_log_name='a2c',
                             total_timesteps=50000) if if_using_a2c else None

trained_ppo = agent.train_model(model=model_ppo,
                             tb_log_name='ppo',
                             total_timesteps=50000) if if_using_ppo else None

trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None
trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None


if_using_sac = True

model_sac = agent.get_model("sac")


if if_using_sac:
    tmp_path = RESULTS_DIR + '/sac'
    # Only use basic loggers to avoid rollout_buffer errors
    new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_sac.set_logger(new_logger_sac)





trained_sac = agent.train_model(model=model_sac, tb_log_name='sac', total_timesteps=50000) if if_using_sac else None

trained_sac.save(TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
import matplotlib.pyplot as plt

train = pd.read_csv('./data/train_data.csv')
trade = pd.read_csv('./data/trade_data.csv')

train = train.set_index(train.columns[0])
train.index.names = ['']
trade = trade.set_index(trade.columns[0])
trade.index.names = ['']

if_using_a2c = True
if_using_ddpg = False
if_using_ppo = True
if_using_td3 = False
if_using_sac = True

trained_a2c = A2C.load('trained_models/agent_a2c') if if_using_a2c else None
trained_ddpg = DDPG.load("trained_models/agent_ddpg") if if_using_ddpg else None
trained_ppo = PPO.load("trained_models/agent_ppo") if if_using_ppo else None
trained_td3 = TD3.load("trained_models/agent_td3") if if_using_td3 else None
trained_sac = SAC.load("trained_models/agent_sac") if if_using_sac else None

stock_dimension = len(trade.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 20,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
# env_trade, obs_trade = e_trade_gym.get_sb_env()

df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
    model=trained_a2c,
    environment = e_trade_gym) if if_using_a2c else (None, None)

df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
    model=trained_sac,
    environment = e_trade_gym) if if_using_sac else (None, None)

df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
    model=trained_ppo,
    environment = e_trade_gym) if if_using_ppo else (None, None)

print("Env attributes:", dir(e_trade_gym))

def calculate_sharpe(df_account_value):
    returns = df_account_value['account_value'].pct_change().dropna()
    if returns.std() == 0:
        return 0
    return (252**0.5) * returns.mean() / returns.std()


sharpe_a2c = calculate_sharpe(df_account_value_a2c)
sharpe_sac = calculate_sharpe(df_account_value_sac)
sharpe_ppo = calculate_sharpe(df_account_value_ppo)


def compute_rolling_sharpes(account_value_dfs, window=30):
    import pandas as pd
    import numpy as np

    sharpes = {}
    for name, df in account_value_dfs.items():
        returns = df['account_value'].pct_change().dropna()
        rolling = returns.rolling(window=window)
        rolling_sharpe = (rolling.mean() / rolling.std()) * np.sqrt(252)
        sharpes[name] = rolling_sharpe.fillna(0).values  # shape: [T]
    return sharpes

def run_rolling_best_model(df_actions_dict, account_value_dfs, env, window=30):
    import numpy as np
    import pandas as pd

    # Align by truncating to minimum length
    min_len = min(len(df) for df in df_actions_dict.values())
    df_actions_trimmed = {
        name: df.iloc[:min_len].reset_index(drop=True)
        for name, df in df_actions_dict.items()
    }
    account_value_trimmed = {
        name: df.iloc[:min_len].reset_index(drop=True)
        for name, df in account_value_dfs.items()
    }

    # Rolling sharpes per model
    sharpes = compute_rolling_sharpes(account_value_trimmed, window)

    # Select best model at each timestep
    best_model_at_t = []
    model_names = list(df_actions_trimmed.keys())

    for t in range(min_len):
        best_model = model_names[0]
        best_score = -np.inf
        for name in model_names:
            if t < len(sharpes[name]) and sharpes[name][t] > best_score:
                best_score = sharpes[name][t]
                best_model = name
        best_model_at_t.append(best_model)

    # Build ensemble action list using best model per timestep
    ensemble_rows = []
    for t in range(min_len):
        best = best_model_at_t[t]
        ensemble_rows.append(df_actions_trimmed[best].iloc[t].values)

    # Convert to DataFrame
    df_actions_ensemble = pd.DataFrame(ensemble_rows, columns=df_actions_trimmed[best].columns)

    # Backtest
    obs, _ = env.reset()
    account_values = []

    for i in range(len(df_actions_ensemble)):
        action = df_actions_ensemble.iloc[i].values
        try:
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            account_value = env.asset_memory[-1]
            account_values.append(account_value)
            if done:
                break
        except Exception as e:
            print(f"[ERROR] Step {i} failed: {e}")
            account_values.append(np.nan)
            break

    sim_dates = df_actions_ensemble.index[:len(account_values)]
    df_account_value_ensemble = pd.DataFrame({
        'date': sim_dates,
        'account_value': account_values
    })

    return df_account_value_ensemble, df_actions_ensemble, best_model_at_t


# Input dicts (make sure these are precomputed)
df_actions_dict = {
    'a2c': df_actions_a2c,
    'ppo': df_actions_ppo,
    'sac': df_actions_sac
}

account_value_dfs = {
    'a2c': df_account_value_a2c,
    'ppo': df_account_value_ppo,
    'sac': df_account_value_sac
}

# Run smarter ensemble
df_account_value_smart, df_actions_smart, model_choice = run_rolling_best_model(
    df_actions_dict, account_value_dfs, env=e_trade_gym, window=30
)


df_account_value_smart['date'] = df_account_value_a2c['date']  # Align dates with original A2C

df_account_value_smart

model_choice

from collections import Counter

last5 = Counter(model_choice[-5:]).most_common()
full = Counter(model_choice).most_common()

# Simple weighted voting
score_map = {}
for model, count in full:
    score_map[model] = count * 0.4  # weight for full history

for model, count in last5:
    score_map[model] = score_map.get(model, 0) + count * 0.6  # weight for recent

most_common = max(score_map.items(), key=lambda x: x[1])[0]

most_common

confidence = score_map[most_common] / sum(score_map.values())
print(f"Most common model: {most_common} with confidence {confidence:.2f}")

import os
import csv

csv_path = 'most_common_model.csv'

# Check if file exists
file_exists = os.path.isfile(csv_path)

# Write or append the most_common value and confidence
with open(csv_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(['most_common', 'confidence'])  # Write header if file does not exist
    writer.writerow([most_common, confidence])

print(f"Appended most_common ('{most_common}') and confidence ({confidence:.2f}) to {csv_path}")

# Split the data
train = data_split(processed_full, TRAIN_START_DATE,TRADE_END_DATE)
print(len(train))

stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 20,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}


e_train_gym = StockTradingEnv(df = train, **env_kwargs)

model_best = agent.get_model(most_common)

tmp_path = RESULTS_DIR + '/best_model'
new_logger_best = configure(tmp_path, ["stdout", "csv", "tensorboard"])
# Set new logger
model_best.set_logger(new_logger_best)


trained_model_best = agent.train_model(model=model_best, tb_log_name='best', total_timesteps=50000)

trained_model_best.save(TRAINED_MODEL_DIR + "/agent_best_model")