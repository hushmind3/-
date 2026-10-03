#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/benstaf/FinRL_DeepSeek/blob/main/FinRL_DeepSeek_backtest.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # FinRL-DeepSeek. Backtest
# 

# # Part 1. Install Packages

# In[ ]:


# get_ipython().system('pip install git+https://github.com/benstaf/FinRL.git')


# In[ ]:


# get_ipython().system('pip install selenium webdriver-manager alpaca-py datasets')


# #Download environments: https://github.com/benstaf/FinRL_DeepSeek
# #And trading agents: https://huggingface.co/benstaf/Trading_agents

# In[ ]:


# cd FinRL_LLM


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

#from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
# from env_stocktrading import StockTradingEnv


# # Import PPO-DeepSeek environments
# from env_stocktrading_llm import StockTradingEnv as StockTradingEnv_llm
# from env_stocktrading_llm_1 import StockTradingEnv as StockTradingEnv_llm_1
# from env_stocktrading_llm_01 import StockTradingEnv as StockTradingEnv_llm_01

# # Import CPPO-DeepSeek risk environments
from env_stocktrading_llm_risk import StockTradingEnv as StockTradingEnv_llm_risk
# from env_stocktrading_llm_risk_1 import StockTradingEnv as StockTradingEnv_llm_risk_1
# from env_stocktrading_llm_risk_01 import StockTradingEnv as StockTradingEnv_llm_risk_01

#from env_stocktrading_llm import StockTradingEnv as StockTradingEnv_llm

#from env_stocktrading_llm_risk import StockTradingEnv as StockTradingEnv_llm_risk


#from finrl.meta.env_stock_trading.env_stocktrading_llm import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


from datasets import load_dataset

# get_ipython().run_line_magic('matplotlib', 'inline')


# # Part 2. Backtesting

# In[ ]:


# from Huggging Face :
dataset = load_dataset("benstaf/nasdaq_2013_2023", data_files='trade_data_deepseek_sentiment_2019_2023.csv')

# Convert to pandas DataFrame
trade = pd.DataFrame(dataset['train'])

#trade= pd.read_csv('/content/machine_learning/trade_data_qwen_sentiment.csv')

trade = trade.drop('Unnamed: 0',axis=1)

# Create a new index based on unique dates
unique_dates = trade['date'].unique()
date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}

# Create new index based on the date mapping
trade['new_idx'] = trade['date'].map(date_to_idx)

# Set this as the index
trade = trade.set_index('new_idx')


#missing values with 0
trade['llm_sentiment'].fillna(0, inplace=True)
trade_llm=trade


# In[ ]:


#trade = pd.read_csv('/content/machine_learning/trade_data_qwen_risk.csv')

# from Huggging Face :
dataset = load_dataset("benstaf/nasdaq_2013_2023", data_files='trade_data_deepseek_risk_2019_2023.csv')

# Convert to pandas DataFrame
trade = pd.DataFrame(dataset['train'])

trade = trade.drop('Unnamed: 0',axis=1)

# Create a new index based on unique dates
unique_dates = trade['date'].unique()
date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}

# Create new index based on the date mapping
trade['new_idx'] = trade['date'].map(date_to_idx)

# Set this as the index
trade = trade.set_index('new_idx')


#missing values with 0
trade['llm_sentiment'].fillna(0, inplace=True)
#missing values with 3
trade['llm_risk'].fillna(3, inplace=True)
trade_llm_risk=trade


# In[ ]:


#trade = pd.read_csv('/content/machine_learning/trade_data_qwen_risk.csv')

# from Huggging Face :
dataset = load_dataset("benstaf/nasdaq_2013_2023", data_files='trade_data_2019_2023.csv')

# Convert to pandas DataFrame
trade = pd.DataFrame(dataset['train'])

trade = trade.drop('Unnamed: 0',axis=1)

# Create a new index based on unique dates
unique_dates = trade['date'].unique()
date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}

# Create new index based on the date mapping
trade['new_idx'] = trade['date'].map(date_to_idx)

# Set this as the index
trade = trade.set_index('new_idx')


# ### Trading (Out-of-sample Performance)
# 
# We update periodically in order to take full advantage of the data, e.g., retrain quarterly, monthly or weekly. We also tune the parameters along the way, in this notebook we use the in-sample data from 2009-01 to 2020-07 to tune the parameters once, so there is some alpha decay here as the length of trade date extends.
# 
# Numerous hyperparameters – e.g. the learning rate, the total number of samples to train on – influence the learning process and are usually determined by testing some variations.

# In[ ]:


stock_dimension = len(trade.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension #+ stock_dimension # +LLM sentiment
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


# In[ ]:


stock_dimension_llm = len(trade_llm.tic.unique())
state_space_llm = 1 + 2 * stock_dimension_llm + (1+len(INDICATORS)) * stock_dimension_llm #+ stock_dimension # +LLM sentiment
print(f"Stock Dimension: {stock_dimension_llm}, State Space: {state_space_llm}")


# In[ ]:


stock_dimension = len(trade.tic.unique())
state_space_llm_risk = 1 + 2 * stock_dimension + (2+len(INDICATORS)) * stock_dimension #+ stock_dimension # +LLM sentiment + LLM risk
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space_llm_risk}")


# In[ ]:


buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
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


# In[ ]:


buy_cost_list_llm = sell_cost_list_llm = [0.001] * stock_dimension_llm
num_stock_shares_llm = [0] * stock_dimension_llm

env_kwargs_llm = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares_llm,
    "buy_cost_pct": buy_cost_list_llm,
    "sell_cost_pct": sell_cost_list_llm,
    "state_space": state_space_llm,
    "stock_dim": stock_dimension_llm,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension_llm,
    "reward_scaling": 1e-4
}


# In[ ]:


buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs_llm_risk = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space_llm_risk,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}


# In[ ]:


# e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
# env_trade, obs_trade = e_trade_gym.get_sb_env()


# In[ ]:


# e_trade_llm_gym = StockTradingEnv_llm(df = trade_llm, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs_llm)
# # env_trade, obs_trade = e_trade_gym.get_sb_env()


# # In[ ]:


# # Environment for PPO-DeepSeek 10%
# e_trade_llm_gym = StockTradingEnv_llm(df=trade_llm, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs_llm)

# # Environment for PPO-DeepSeek 1%
# e_trade_llm_gym_1 = StockTradingEnv_llm_1(df=trade_llm, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs_llm)

# # Environment for PPO-DeepSeek 0.1%
# e_trade_llm_gym_01 = StockTradingEnv_llm_01(df=trade_llm, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs_llm)


# # In[ ]:


e_trade_llm_risk_gym = StockTradingEnv_llm_risk(df = trade_llm_risk, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs_llm_risk)


# # In[ ]:


# # Environment for CPPO-DeepSeek 10% risk
# e_trade_llm_risk_gym = StockTradingEnv_llm_risk(df=trade_llm_risk, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs_llm_risk)

# # Environment for CPPO-DeepSeek 1% risk
# e_trade_llm_risk_gym_1 = StockTradingEnv_llm_risk_1(df=trade_llm_risk, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs_llm_risk)

# # Environment for CPPO-DeepSeek 0.1% risk
# e_trade_llm_risk_gym_01 = StockTradingEnv_llm_risk_01(df=trade_llm_risk, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs_llm_risk)


# # In[ ]:


# observation_space=e_trade_gym.observation_space
# action_space=e_trade_gym.action_space


# # In[ ]:


# observation_space_llm=e_trade_llm_gym.observation_space
# action_space_llm=e_trade_llm_gym.action_space


# # In[ ]:


# observation_space_llm_risk=e_trade_llm_risk_gym.observation_space
# action_space_llm_risk=e_trade_llm_risk_gym.action_space


# # In[ ]:


# # Observation and action spaces for PPO-DeepSeek 10%
# observation_space_llm = e_trade_llm_gym.observation_space
# action_space_llm = e_trade_llm_gym.action_space

# # Observation and action spaces for PPO-DeepSeek 1%
# observation_space_llm_1 = e_trade_llm_gym_1.observation_space
# action_space_llm_1 = e_trade_llm_gym_1.action_space

# # Observation and action spaces for PPO-DeepSeek 0.1%
# observation_space_llm_01 = e_trade_llm_gym_01.observation_space
# action_space_llm_01 = e_trade_llm_gym_01.action_space

# # Observation and action spaces for CPPO-DeepSeek 10% risk
# observation_space_llm_risk = e_trade_llm_risk_gym.observation_space
# action_space_llm_risk = e_trade_llm_risk_gym.action_space

# # Observation and action spaces for CPPO-DeepSeek 1% risk
# observation_space_llm_risk_1 = e_trade_llm_risk_gym_1.observation_space
# action_space_llm_risk_1 = e_trade_llm_risk_gym_1.action_space

# # Observation and action spaces for CPPO-DeepSeek 0.1% risk
# observation_space_llm_risk_01 = e_trade_llm_risk_gym_01.observation_space
# action_space_llm_risk_01 = e_trade_llm_risk_gym_01.action_space


# In[ ]:


# print("State shape:", observation_space_llm.shape)


# In[ ]:


import numpy as np
import scipy.signal
from gymnasium.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


# In[ ]:


# get_ipython().system('dir')


# # In[ ]:


# # Load the model
# loaded_ppo = MLPActorCritic(observation_space,action_space, hidden_sizes=(512, 512))
# loaded_ppo.load_state_dict(torch.load('/content/FinRL_LLM/trained_models/agent_ppo_100_epochs_20k_steps.pth'))
# #loaded_ppo.load_state_dict(torch.load('//content/agent_ppo_100_epochs_20k_steps.pth'))

# loaded_ppo.eval()  # Set the model to evaluation mode


# # In[ ]:


# # Load the model
# loaded_cppo = MLPActorCritic(observation_space,action_space, hidden_sizes=(512, 512))
# loaded_cppo.load_state_dict(torch.load('/content/FinRL_LLM/trained_models/agent_cppo_100_epochs_20k_steps.pth'))
# #loaded_ppo.load_state_dict(torch.load('/kaggle/input/agent_cppo_25_epochs_20k_steps/pytorch/default/1/agent_ppo_25_epochs_20k_steps.pth'))

# loaded_cppo.eval()  # Set the model to evaluation mode


# # In[ ]:


# # Load the model
# loaded_ppo_llm = MLPActorCritic(observation_space_llm,action_space_llm, hidden_sizes=(512, 512))
# loaded_ppo_llm.load_state_dict(torch.load('/content/FinRL_LLM/trained_models/agent_ppo_deepseek_100_epochs_20k_steps.pth'))
# #loaded_ppo_llm.load_state_dict(torch.load('/kaggle/input/agent_cppo_25_epochs_20k_steps/pytorch/default/1/agent_ppo_25_epochs_20k_steps.pth'))

# loaded_ppo_llm.eval()  # Set the model to evaluation mode


# # In[ ]:


# # Load the model
# loaded_ppo_llama = MLPActorCritic(observation_space_llm,action_space_llm, hidden_sizes=(512, 512))
# loaded_ppo_llama.load_state_dict(torch.load('/content/FinRL_LLM/trained_models/agent_ppo_llama_100_epochs_20k_steps.pth'))

# loaded_ppo_llm.eval()  # Set the model to evaluation mode


# # In[ ]:


# # Load the model
# loaded_cppo_llm_risk = MLPActorCritic(observation_space_llm_risk,action_space_llm_risk, hidden_sizes=(512, 512))
# loaded_cppo_llm_risk.load_state_dict(torch.load('/content/FinRL_LLM/trained_models/agent_cppo_deepseek_100_epochs_20k_steps.pth'))

# loaded_cppo_llm_risk.eval()  # Set the model to evaluation mode


# # In[ ]:


# # Load the PPO-DeepSeek 10% model
# loaded_ppo_llm = MLPActorCritic(observation_space_llm, action_space_llm, hidden_sizes=(512, 512))
# loaded_ppo_llm.load_state_dict(torch.load('/content/FinRL_LLM/trained_models/agent_ppo_deepseek_100_epochs_20k_steps.pth'))
# loaded_ppo_llm.eval()  # Set the model to evaluation mode


# # Load the PPO-DeepSeek 1% model
# loaded_ppo_llm_1 = MLPActorCritic(observation_space_llm_1, action_space_llm_1, hidden_sizes=(512, 512))
# loaded_ppo_llm_1.load_state_dict(torch.load('/content/FinRL_LLM/trained_models/agent_ppo_deepseek_100_epochs_20k_steps_1.pth'))
# loaded_ppo_llm_1.eval()

# # Load the PPO-DeepSeek 0.1% model
# loaded_ppo_llm_01 = MLPActorCritic(observation_space_llm_01, action_space_llm_01, hidden_sizes=(512, 512))
# loaded_ppo_llm_01.load_state_dict(torch.load('/content/FinRL_LLM/trained_models/agent_ppo_deepseek_100_epochs_20k_steps_01.pth'))
# loaded_ppo_llm_01.eval()

# # Load the CPPO-DeepSeek 10% risk model
# loaded_cppo_llm_risk = MLPActorCritic(observation_space_llm_risk, action_space_llm_risk, hidden_sizes=(512, 512))
# loaded_cppo_llm_risk.load_state_dict(torch.load('/content/FinRL_LLM/trained_models/agent_cppo_deepseek_100_epochs_20k_steps.pth'))
# loaded_cppo_llm_risk.eval()

# # Load the CPPO-DeepSeek 1% risk model
# loaded_cppo_llm_risk_1 = MLPActorCritic(observation_space_llm_risk_1, action_space_llm_risk_1, hidden_sizes=(512, 512))
# loaded_cppo_llm_risk_1.load_state_dict(torch.load('/content/FinRL_LLM/trained_models/agent_cppo_deepseek_100_epochs_20k_steps_1.pth'))
# loaded_cppo_llm_risk_1.eval()

# # Load the CPPO-DeepSeek 0.1% risk model
# loaded_cppo_llm_risk_01 = MLPActorCritic(observation_space_llm_risk_01, action_space_llm_risk_01, hidden_sizes=(512, 512))
# loaded_cppo_llm_risk_01.load_state_dict(torch.load('/content/FinRL_LLM/trained_models/agent_cppo_deepseek_100_epochs_20k_steps_01.pth'))
# loaded_cppo_llm_risk_01.eval()


# # In[ ]:


# # Load the model
# loaded_cppo_llama_risk = MLPActorCritic(observation_space_llm_risk,action_space_llm_risk, hidden_sizes=(512, 512))
# loaded_cppo_llama_risk.load_state_dict(torch.load('/content/FinRL_LLM/trained_models/agent_deepseek_20_epochs_20k_steps.pth'))

# loaded_cppo_llama_risk.eval()  # Set the model to evaluation mode


# # In[ ]:

def DRL_prediction(act, environment):
    import torch
    _torch = torch
    
    # Determine the device of the model
    model_device = next(act.parameters()).device
    
    state, _ = environment.reset()
    account_memory = []  # To store portfolio values
    actions_memory = []  # To store actions taken
    portfolio_distribution = []  # To store portfolio distribution
    episode_total_assets = [environment.initial_amount]

    with _torch.no_grad():
        for i in range(len(environment.df.index.unique())):
            # Create tensor on the SAME device as the model
            s_tensor = _torch.as_tensor((state,), dtype=torch.float32, device=model_device)
            
            a_tensor, _, _ = act.step(s_tensor)  # Compute action
            action = a_tensor[0]  # Extract action

            # Step through the environment
            state, reward, done, _, _ = environment.step(action)

            # Get stock prices for the current day
            price_array = environment.df.loc[environment.day, "close"].values

            # Stock holdings and cash balance
            stock_holdings = environment.num_stock_shares
            cash_balance = environment.asset_memory[-1]

            # Calculate total portfolio value
            total_asset = cash_balance + (price_array * stock_holdings).sum()

            # Calculate portfolio distribution
            stock_values = price_array * stock_holdings
            total_invested = stock_values.sum()
            distribution = stock_values / total_asset  # Fraction of each stock in the total portfolio
            cash_fraction = cash_balance / total_asset

            # Store results
            episode_total_assets.append(total_asset)
            account_memory.append(total_asset)
            actions_memory.append(action)
            portfolio_distribution.append({"cash": cash_fraction, "stocks": distribution.tolist()})

            if done:
                break

    print("Test Finished!")
    return episode_total_assets, account_memory, actions_memory, portfolio_distribution


# # In[ ]:


# df_assets_cppo[:10]


# # In[ ]:


# df_assets_ppo, df_account_value_ppo, df_actions_ppo, df_portfolio_distribution_ppo = DRL_prediction(act=loaded_ppo, environment=e_trade_gym)
# #episode_total_assets, account_memory, actions_memory, portfolio_distribution = DRL_prediction(act=loaded_ppo, environment=e_trade_gym)


# # In[ ]:


# df_assets_cppo, df_account_value_cppo, df_actions_cppo, df_portfolio_distribution_cppo = DRL_prediction(act=loaded_cppo, environment=e_trade_gym)


# # In[ ]:


# # Prediction for PPO-DeepSeek 10%
# df_assets_ppo_llm, df_account_value_ppo_llm, df_actions_ppo_llm, df_portfolio_distribution_ppo_llm = DRL_prediction(
#     act=loaded_ppo_llm, environment=e_trade_llm_gym
# )

# # Prediction for PPO-DeepSeek 1%
# df_assets_ppo_llm_1, df_account_value_ppo_llm_1, df_actions_ppo_llm_1, df_portfolio_distribution_ppo_llm_1 = DRL_prediction(
#     act=loaded_ppo_llm_1, environment=e_trade_llm_gym_1
# )

# # Prediction for PPO-DeepSeek 0.1%
# df_assets_ppo_llm_01, df_account_value_ppo_llm_01, df_actions_ppo_llm_01, df_portfolio_distribution_ppo_llm_01 = DRL_prediction(
#     act=loaded_ppo_llm_01, environment=e_trade_llm_gym_01
# )

# # Prediction for CPPO-DeepSeek 10% risk
# df_assets_cppo_llm_risk, df_account_value_cppo_llm_risk, df_actions_cppo_llm_risk, df_portfolio_distribution_cppo_llm_risk = DRL_prediction(
#     act=loaded_cppo_llm_risk, environment=e_trade_llm_risk_gym
# )

# # Prediction for CPPO-DeepSeek 1% risk
# df_assets_cppo_llm_risk_1, df_account_value_cppo_llm_risk_1, df_actions_cppo_llm_risk_1, df_portfolio_distribution_cppo_llm_risk_1 = DRL_prediction(
#    act=loaded_cppo_llm_risk_1, environment=e_trade_llm_risk_gym_1
# )

# # Prediction for CPPO-DeepSeek 0.1% risk
# df_assets_cppo_llm_risk_01, df_account_value_cppo_llm_risk_01, df_actions_cppo_llm_risk_01, df_portfolio_distribution_cppo_llm_risk_01 = DRL_prediction(
#    act=loaded_cppo_llm_risk_01, environment=e_trade_llm_risk_gym_01
# )


# # In[ ]:


# df_assets_ppo_llm, df_account_value_ppo_llm, df_actions_ppo_llm, df_portfolio_distribution_ppo_llm = DRL_prediction(act=loaded_ppo_llm, environment=e_trade_llm_gym)


# # In[ ]:


# df_assets_ppo_llama, df_account_value_ppo_llama, df_actions_ppo_llama, df_portfolio_distribution_ppo_llama= DRL_prediction(act=loaded_ppo_llama, environment=e_trade_llm_gym)


# # In[ ]:


# df_assets_cppo_llm_risk, df_account_value_cppo_llm_risk, df_actions_cppo_llm_risk, df_portfolio_distribution_cppo_llm_risk = DRL_prediction(act=loaded_cppo_llm_risk, environment=e_trade_llm_risk_gym)


# # In[ ]:


# df_assets_cppo_llama_risk, df_account_value_cppo_llama_risk, df_actions_cppo_llama_risk, df_portfolio_distribution_cppo_llama_risk = DRL_prediction(act=loaded_cppo_llama_risk, environment=e_trade_llm_risk_gym)


# # # Part 4: NASDAQ 100 index

# # **Add** NASDAQ 100 index as a baseline to compare with.

# # In[ ]:


# TRAIN_START_DATE = '2013-01-01'
# TRAIN_END_DATE = '2018-12-31'
# TRADE_START_DATE = '2019-01-01'
# TRADE_END_DATE = '2023-12-31'


# # In[ ]:


# df_dji = YahooDownloader(
#     start_date=TRADE_START_DATE, end_date=TRADE_END_DATE, ticker_list=["ndx"]
# ).fetch_data()


# # In[ ]:


# len(df_dji)


# # In[ ]:


# df_dji[:10]


# # In[ ]:


# df_dji = df_dji[["date", "close"]]
# fst_day = df_dji["close"][0]
# dji = pd.merge(
#     df_dji["date"],
#     df_dji["close"].div(fst_day).mul(1000000),
#     how="outer",
#     left_index=True,
#     right_index=True,
# ).set_index("date")


# # In[ ]:


# fst_day = df_dji["close"].iloc[0]  # Safely get the first value
# df_dji_normalized_close = list(df_dji["close"].div(fst_day).mul(1000000))


# # In[ ]:


# len(df_dji_normalized_close),


# # <a id='4'></a>
# # # Part 5: Backtesting Results
# # Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.

# # Now, everything is ready, we can plot the backtest result.

# # In[ ]:


# fst_day_ppo = df_assets_ppo[1]  # Safely get the first value
# df_assets_ppo_series = pd.Series(df_assets_ppo[1:])
# df_ppo_normalized_close = list(df_assets_ppo_series.div(fst_day_ppo).mul(1000000))


# # In[ ]:


# # Normalize PPO-DeepSeek 10%
# fst_day_ppo_llm = df_assets_ppo_llm[1]  # Safely get the first value
# df_assets_ppo_llm_series = pd.Series(df_assets_ppo_llm[1:])
# df_ppo_llm_normalized_close = list(df_assets_ppo_llm_series.div(fst_day_ppo_llm).mul(1000000))

# # Normalize PPO-DeepSeek 1%
# fst_day_ppo_llm_1 = df_assets_ppo_llm_1[1]  # Safely get the first value
# df_assets_ppo_llm_series_1 = pd.Series(df_assets_ppo_llm_1[1:])
# df_ppo_llm_normalized_close_1 = list(df_assets_ppo_llm_series_1.div(fst_day_ppo_llm_1).mul(1000000))

# # Normalize PPO-DeepSeek 0.1%
# #fst_day_ppo_llm_01 = df_assets_ppo_llm_01[1]  # Safely get the first value
# #df_assets_ppo_llm_series_01 = pd.Series(df_assets_ppo_llm_01[1:])
# #df_ppo_llm_normalized_close_01 = list(df_assets_ppo_llm_series_01.div(fst_day_ppo_llm_01).mul(1000000))


# # In[ ]:


# # prompt: repeat the same renormalization as above for cppo, cppo_llm_risk and ppo_llm

# fst_day_ppo_llama = df_assets_ppo_llama[1]  # Safely get the first value
# df_assets_ppo_llama_series = pd.Series(df_assets_ppo_llama[1:])
# df_ppo_llama_normalized_close = list(df_assets_ppo_llama_series.div(fst_day_ppo_llama).mul(1000000))


# # In[ ]:


# fst_day_cppo = df_assets_cppo[1]  # Safely get the first value
# df_assets_cppo_series = pd.Series(df_assets_cppo[1:])
# df_cppo_normalized_close = list(df_assets_cppo_series.div(fst_day_cppo).mul(1000000))


# # In[ ]:


# # Normalize CPPO-DeepSeek 10%
# fst_day_cppo_llm_risk = df_assets_cppo_llm_risk[1]  # Safely get the first value
# df_assets_cppo_llm_risk_series = pd.Series(df_assets_cppo_llm_risk[1:])
# df_cppo_llm_risk_normalized_close = list(df_assets_cppo_llm_risk_series.div(fst_day_cppo_llm_risk).mul(1000000))

# # Normalize CPPO-DeepSeek 1%
# fst_day_cppo_llm_risk_1 = df_assets_cppo_llm_risk_1[1]  # Safely get the first value
# df_assets_cppo_llm_risk_series_1 = pd.Series(df_assets_cppo_llm_risk_1[1:])
# df_cppo_llm_risk_normalized_close_1 = list(df_assets_cppo_llm_risk_series_1.div(fst_day_cppo_llm_risk_1).mul(1000000))

# # Normalize CPPO-DeepSeek 0.1%
# fst_day_cppo_llm_risk_01 = df_assets_cppo_llm_risk_01[1]  # Safely get the first value
# df_assets_cppo_llm_risk_series_01 = pd.Series(df_assets_cppo_llm_risk_01[1:])
# df_cppo_llm_risk_normalized_close_01 = list(df_assets_cppo_llm_risk_series_01.div(fst_day_cppo_llm_risk_01).mul(1000000))


# # In[ ]:


# # prompt: repeat the same renormalization as above for cppo, cppo_llama_risk and ppo_llama

# fst_day_cppo_llama_risk = df_assets_cppo_llama_risk[1]  # Safely get the first value
# df_assets_cppo_llama_risk_series = pd.Series(df_assets_cppo_llama_risk[1:])
# df_cppo_llama_risk_normalized_close = list(df_assets_cppo_llama_risk_series.div(fst_day_cppo_llama_risk).mul(1000000))


# # In[ ]:


# len(trade['date'].drop_duplicates().values)


# In[ ]:


def filter_to_common_dates(trade, df_dji, df_assets_ppo, df_dji_normalized_close):
    """
    Filters df_assets_ppo and df_dji_normalized_close based on the common dates from trade and df_dji.

    Parameters:
        trade (pd.DataFrame): DataFrame containing a 'date' column for the trade data.
        df_dji (pd.DataFrame): DataFrame containing a 'date' column for DJI data.
        df_assets_ppo (list or array-like): Values corresponding to trade['date'].
        df_dji_normalized_close (list or array-like): Values corresponding to df_dji['date'].

    Returns:
        pd.Series, pd.Series: Filtered series for df_assets_ppo and df_dji_normalized_close.
    """
    # Extract unique trading dates from trade and DJI dates
    trade_dates = pd.to_datetime(trade['date'].unique())
    dji_dates = pd.to_datetime(df_dji['date'].unique())


  #  first_date = trade_dates[0]
   # date_before_first = first_date - pd.DateOffset(days=1)

# Prepend the date before the first date to trade_dates
    #trade_dates = pd.DatetimeIndex([date_before_first] + trade_dates.tolist())

    # Convert inputs to pandas Series with their respective dates as indices
    df_assets_ppo_series = pd.Series(df_assets_ppo, index=trade_dates)
    df_dji_normalized_close_series = pd.Series(df_dji_normalized_close, index=dji_dates)

    # Find the common dates
    common_dates = trade_dates.intersection(dji_dates)

    # Filter both series to the common dates
    df_assets_ppo_filtered = df_assets_ppo_series.reindex(common_dates)
    df_dji_normalized_close_filtered = df_dji_normalized_close_series.reindex(common_dates)

    # Return the filtered series
    return df_assets_ppo_filtered, df_dji_normalized_close_filtered, common_dates


# In[ ]:


# common_dates


# # In[ ]:





# # In[ ]:


# df_assets_ppo_filtered, df_dji_normalized_close_filtered, common_dates = filter_to_common_dates(trade, df_dji, df_ppo_normalized_close, df_dji_normalized_close)


# # In[ ]:


# df_assets_cppo_filtered, df_dji_normalized_close_filtered, common_dates = filter_to_common_dates(trade, df_dji, df_cppo_normalized_close, df_dji_normalized_close)


# # In[ ]:


# df_assets_ppo_llm_filtered, df_dji_normalized_close_filtered, common_dates = filter_to_common_dates(trade, df_dji, df_ppo_llm_normalized_close, df_dji_normalized_close)
# df_assets_ppo_llm_filtered_1, _, _ = filter_to_common_dates(
#     trade, df_dji, df_ppo_llm_normalized_close_1, df_dji_normalized_close
# )

# #df_assets_ppo_llm_filtered_01, _, _ = filter_to_common_dates(
#    # trade, df_dji, df_ppo_llm_normalized_close_01, df_dji_normalized_close
# #)


# # In[ ]:


# df_assets_ppo_llama_filtered, df_dji_normalized_close_filtered, common_dates = filter_to_common_dates(trade, df_dji, df_ppo_llama_normalized_close, df_dji_normalized_close)


# # In[ ]:


# df_assets_cppo_llm_risk_filtered, df_dji_normalized_close_filtered, common_dates = filter_to_common_dates(trade, df_dji, df_cppo_llm_risk_normalized_close, df_dji_normalized_close)
# df_assets_cppo_llm_risk_filtered_1, _, _ = filter_to_common_dates(
#     trade, df_dji, df_cppo_llm_risk_normalized_close_1, df_dji_normalized_close
# )

# df_assets_cppo_llm_risk_filtered_01, _, _ = filter_to_common_dates(
#     trade, df_dji, df_cppo_llm_risk_normalized_close_01, df_dji_normalized_close
# )


# # In[ ]:


# #df_assets_cppo_llama_risk_filtered, df_dji_normalized_close_filtered, common_dates = filter_to_common_dates(trade, df_dji, df_cppo_llama_risk_normalized_close, df_dji_normalized_close)


# # In[ ]:





# # In[ ]:


# df_dji_normalized_close_filtered[1]


# # In[ ]:


# result = pd.DataFrame(
#     {
#         "PPO 100 epochs": df_assets_ppo_filtered,
#         "CPPO 100 epochs": df_assets_cppo_filtered,
#         "PPO-DeepSeek 100 epochs": df_assets_ppo_llm_filtered,
#     #    "PPO-Llama 100 epochs": df_assets_ppo_llama_filtered,
#         "CPPO-DeepSeek 100 epochs": df_assets_cppo_llm_risk_filtered,
#     #    "CPPO-Llama 20 epochs": df_assets_cppo_llama_risk_filtered,
#         "Nasdaq-100 index": df_dji_normalized_close_filtered,
#     }
# )

# # Display the result
# print(result)


# # In[ ]:


# result_ppo = pd.DataFrame(
#     {
#         "PPO": df_assets_ppo_filtered,
#         "PPO-DeepSeek 10%": df_assets_ppo_llm_filtered,
#         "PPO-DeepSeek 1%": df_assets_ppo_llm_filtered_1,
#         "PPO-DeepSeek 0.1%": df_assets_ppo_llm_filtered_01,
#         "Nasdaq-100 index": df_dji_normalized_close_filtered,
#     }
# )


# # In[ ]:


# result_cppo = pd.DataFrame(
#     {
#         #"PPO 100 epochs": df_assets_ppo_filtered,
#         "CPPO": df_assets_cppo_filtered,
#         "CPPO-DeepSeek 10%": df_assets_cppo_llm_risk_filtered,
#         "CPPO-DeepSeek 1%": df_assets_cppo_llm_risk_filtered_1,
#         "CPPO-DeepSeek 0.1%": df_assets_cppo_llm_risk_filtered_01,
#         "Nasdaq-100 index": df_dji_normalized_close_filtered,
#     }
# )


# # In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

# Utility Functions
def calculate_metric(returns_strategy, returns_benchmark, confidence_level=0.05, upside_confidence=0.95):
    """Calculate performance metrics: IR, CVaR, and Rachev Ratio."""
    excess_return = returns_strategy - returns_benchmark
    ir = excess_return.mean() / excess_return.std()
    var = np.percentile(returns_strategy, confidence_level * 100)
    cvar = returns_strategy[returns_strategy <= var].mean()
    upside_var = np.percentile(returns_strategy, upside_confidence * 100)
    downside_var = var
    rachev_ratio = returns_strategy[returns_strategy >= upside_var].mean() / abs(returns_strategy[returns_strategy <= downside_var].mean())
    return {"Information Ratio": ir, "CVaR": cvar, "Rachev Ratio": rachev_ratio}

def align_returns(result, col_strategy, col_benchmark):
    """Align returns for strategy and benchmark."""
    returns_strategy = result[col_strategy].pct_change().dropna()
    returns_benchmark = result[col_benchmark].pct_change().dropna()
    return returns_strategy.align(returns_benchmark, join="inner")

# Metrics Calculation
def compute_metrics(result, strategies, benchmark, confidence_level=0.05, upside_confidence=0.95):
    """
    Compute metrics for multiple strategies compared to a benchmark.

    Parameters:
        result (pd.DataFrame): DataFrame with strategies and benchmark columns.
        strategies (list): List of strategy column names.
        benchmark (str): Benchmark column name.
        confidence_level (float): Confidence level for CVaR calculation.
        upside_confidence (float): Confidence level for upside in Rachev Ratio.

    Returns:
        dict: Performance metrics for each strategy.
    """
    metrics = {}
    for strategy in strategies:
        aligned_strategy, aligned_benchmark = align_returns(result, strategy, benchmark)
        metrics[strategy] = calculate_metric(
            aligned_strategy, aligned_benchmark, confidence_level, upside_confidence
        )
    return metrics

# Plotting
def plot_cumulative_returns(result, metrics, strategies, benchmark):
    """
    Plot cumulative returns for strategies and benchmark with annotated metrics.

    Parameters:
        result (pd.DataFrame): DataFrame with strategies and benchmark.
        metrics (dict): Performance metrics.
        strategies (list): List of strategy column names.
        benchmark (str): Benchmark column name.
    """
    plt.figure(figsize=(12, 6))
    for strategy in strategies:
        cumulative_returns = (1 + result[strategy].pct_change().dropna()).cumprod()
        plt.plot(cumulative_returns, label=f"{strategy}")
    cumulative_benchmark = (1 + result[benchmark].pct_change().dropna()).cumprod()
    plt.plot(cumulative_benchmark, label=f"{benchmark} (Benchmark)")
    plt.title("Cumulative Returns with Performance Metrics")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid()
    plt.show()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

# Utility Functions
def calculate_metric(returns_strategy, returns_benchmark, confidence_level=0.05, upside_confidence=0.95):
    """Calculate performance metrics: IR, CVaR, and Rachev Ratio."""
    excess_return = returns_strategy - returns_benchmark
    ir = excess_return.mean() / excess_return.std()
    var = np.percentile(returns_strategy, confidence_level * 100)
    cvar = returns_strategy[returns_strategy <= var].mean()
    upside_var = np.percentile(returns_strategy, upside_confidence * 100)
    downside_var = var
    rachev_ratio = returns_strategy[returns_strategy >= upside_var].mean() / abs(returns_strategy[returns_strategy <= downside_var].mean())
    return {"Information Ratio": ir, "CVaR": cvar, "Rachev Ratio": rachev_ratio}

def align_returns(result, col_strategy, col_benchmark):
    """Align returns for strategy and benchmark."""
    returns_strategy = result[col_strategy].pct_change().dropna()
    returns_benchmark = result[col_benchmark].pct_change().dropna()
    return returns_strategy.align(returns_benchmark, join="inner")

# Metrics Calculation
def compute_metrics(result, strategies, benchmark, confidence_level=0.05, upside_confidence=0.95):
    """
    Compute metrics for multiple strategies compared to a benchmark.

    Parameters:
        result (pd.DataFrame): DataFrame with strategies and benchmark columns.
        strategies (list): List of strategy column names.
        benchmark (str): Benchmark column name.
        confidence_level (float): Confidence level for CVaR calculation.
        upside_confidence (float): Confidence level for upside in Rachev Ratio.

    Returns:
        dict: Performance metrics for each strategy.
    """
    metrics = {}
    for strategy in strategies:
        aligned_strategy, aligned_benchmark = align_returns(result, strategy, benchmark)
        metrics[strategy] = calculate_metric(
            aligned_strategy, aligned_benchmark, confidence_level, upside_confidence
        )
    return metrics

# Plotting
def plot_cumulative_returns(result, metrics, strategies, benchmark):
    """
    Plot cumulative returns for strategies and benchmark with annotated metrics.

    Parameters:
        result (pd.DataFrame): DataFrame with strategies and benchmark.
        metrics (dict): Performance metrics.
        strategies (list): List of strategy column names.
        benchmark (str): Benchmark column name.
    """
    plt.figure(figsize=(12, 6))
    for strategy in strategies:
        cumulative_returns = (1 + result[strategy].pct_change().dropna()).cumprod()
        plt.plot(cumulative_returns, label=f"{strategy}")
    cumulative_benchmark = (1 + result[benchmark].pct_change().dropna()).cumprod()
    plt.plot(cumulative_benchmark, label=f"{benchmark} (Benchmark)")
    plt.title("Cumulative Returns with Performance Metrics")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid()
    plt.show()

# # Example Usage
# # Assuming `result` DataFrame is prepared with "PPO 25 epochs", "CPPO 25 epochs", and "Nasdaq-100 index"
# strategies = [
# "PPO 100 epochs",
# "CPPO 100 epochs",
# "PPO-DeepSeek 100 epochs",
# #"PPO-Llama 100 epochs",
# "CPPO-DeepSeek 100 epochs"
# #"CPPO-Llama 100 epochs",
# ]
# #strategies = ["PPO 20 epochs", "CPPO 20 epochs", "CPPO-DeepSeek 20 epochs"]
# benchmark = "Nasdaq-100 index"
# metrics = compute_metrics(result, strategies, benchmark)
# plot_cumulative_returns(result, metrics, strategies, benchmark)

# # Print metrics
# for strategy, strategy_metrics in metrics.items():
#     print(f"{strategy} Metrics:")
#     for metric_name, value in strategy_metrics.items():
#         print(f"  {metric_name}: {value:.4f}")


# # In[ ]:


# # Example Usage
# # Assuming `result` DataFrame is prepared with "PPO 25 epochs", "CPPO 25 epochs", and "Nasdaq-100 index"
# strategies = [
# "PPO",
# "PPO-DeepSeek 10%",
# "PPO-DeepSeek 1%",
# "PPO-DeepSeek 0.1%"
# #"CPPO-Llama 100 epochs",
# ]
# #strategies = ["PPO 20 epochs", "CPPO 20 epochs", "CPPO-DeepSeek 20 epochs"]
# benchmark = "Nasdaq-100 index"
# metrics = compute_metrics(result_ppo, strategies, benchmark)
# plot_cumulative_returns(result_ppo, metrics, strategies, benchmark)

# # Print metrics
# for strategy, strategy_metrics in metrics.items():
#     print(f"{strategy} Metrics:")
#     for metric_name, value in strategy_metrics.items():
#         print(f"  {metric_name}: {value:.4f}")


# # In[ ]:


# # Example Usage
# # Assuming `result` DataFrame is prepared with "PPO 25 epochs", "CPPO 25 epochs", and "Nasdaq-100 index"
# strategies = [
# "CPPO",
# "CPPO-DeepSeek 10%",
# "CPPO-DeepSeek 1%",
# "CPPO-DeepSeek 0.1%"
# #"CPPO-Llama 100 epochs",
# ]
# #strategies = ["PPO 20 epochs", "CPPO 20 epochs", "CPPO-DeepSeek 20 epochs"]
# benchmark = "Nasdaq-100 index"
# metrics = compute_metrics(result_cppo, strategies, benchmark)
# plot_cumulative_returns(result, metrics, strategies, benchmark)

# # Print metrics
# for strategy, strategy_metrics in metrics.items():
#     print(f"{strategy} Metrics:")
#     for metric_name, value in strategy_metrics.items():
#         print(f"  {metric_name}: {value:.4f}")


# # In[ ]:


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

# def calculate_information_ratio(returns_strategy, returns_benchmark):
#     """Calculate the Information Ratio (IR)."""
#     excess_return = returns_strategy - returns_benchmark
#     return excess_return.mean() / excess_return.std()

# def calculate_cvar(returns_strategy, confidence_level=0.05):
#     """Calculate the Conditional Value at Risk (CVaR)."""
#     var = np.percentile(returns_strategy, confidence_level * 100)
#     return returns_strategy[returns_strategy <= var].mean()

# def calculate_rachev_ratio(returns_strategy, upside_confidence=0.95, downside_confidence=0.05):
#     """Calculate the Rachev Ratio."""
#     upside_var = np.percentile(returns_strategy, upside_confidence * 100)
#     downside_var = np.percentile(returns_strategy, downside_confidence * 100)
#     upside_mean = returns_strategy[returns_strategy >= upside_var].mean()
#     downside_mean = abs(returns_strategy[returns_strategy <= downside_var].mean())
#     return upside_mean / downside_mean

# def align_and_compute_metrics(result, confidence_level=0.05, upside_confidence=0.95):
#     """
#     Align data for PPO, CPPO, and benchmark, and compute performance metrics.

#     Parameters:
#         result (pd.DataFrame): DataFrame with strategies and benchmark.
#         confidence_level (float): Confidence level for CVaR calculation.
#         upside_confidence (float): Confidence level for upside in Rachev Ratio.

#     Returns:
#         dict: Performance metrics for PPO and CPPO.
#     """
#     # Calculate returns
#     returns_ppo = result["PPO 25 epochs"].pct_change().dropna()
#     returns_cppo = result["CPPO 25 epochs"].pct_change().dropna()
#     returns_benchmark = result["Nasdaq-100 index"].pct_change().dropna()

#     # Align returns
#     returns_ppo, returns_benchmark_ppo = returns_ppo.align(returns_benchmark, join="inner")
#     returns_cppo, returns_benchmark_cppo = returns_cppo.align(returns_benchmark, join="inner")

#     # Compute metrics
#     metrics = {
#         "PPO": {
#             "Information Ratio": calculate_information_ratio(returns_ppo, returns_benchmark_ppo),
#             "CVaR": calculate_cvar(returns_ppo, confidence_level),
#             "Rachev Ratio": calculate_rachev_ratio(returns_ppo, upside_confidence, confidence_level),
#         },
#         "CPPO": {
#             "Information Ratio": calculate_information_ratio(returns_cppo, returns_benchmark_cppo),
#             "CVaR": calculate_cvar(returns_cppo, confidence_level),
#             "Rachev Ratio": calculate_rachev_ratio(returns_cppo, upside_confidence, confidence_level),
#         }
#     }
#     return metrics

# def plot_cumulative_returns(result, metrics):
#     """
#     Plot cumulative returns for PPO, CPPO, and benchmark with annotated metrics.

#     Parameters:
#         result (pd.DataFrame): DataFrame with strategies and benchmark.
#         metrics (dict): Performance metrics.
#     """
#     # Calculate cumulative returns
#     cumulative_ppo = (1 + result["PPO 25 epochs"].pct_change().dropna()).cumprod()
#     cumulative_cppo = (1 + result["CPPO 25 epochs"].pct_change().dropna()).cumprod()
#     cumulative_benchmark = (1 + result["Nasdaq-100 index"].pct_change().dropna()).cumprod()

#     # Plot
#     plt.figure(figsize=(12, 6))
#     plt.plot(cumulative_ppo, label=f"PPO 25 epochs (IR={metrics['PPO']['Information Ratio']:.4f})")
#     plt.plot(cumulative_cppo, label=f"CPPO 25 epochs (IR={metrics['CPPO']['Information Ratio']:.4f})")
#     plt.plot(cumulative_benchmark, label="Nasdaq-100 index (Benchmark)")
#     plt.title("Cumulative Returns with Performance Metrics")
#     plt.legend()
#     plt.xlabel("Date")
#     plt.ylabel("Cumulative Return")
#     plt.grid()
#     plt.show()

# # Example Usage
# # Assuming `result` DataFrame is prepared with "PPO 25 epochs", "CPPO 25 epochs", and "Nasdaq-100 index"
# metrics = align_and_compute_metrics(result)
# plot_cumulative_returns(result, metrics)

# # Print metrics
# for strategy, strategy_metrics in metrics.items():
#     print(f"{strategy} Metrics:")
#     for metric_name, value in strategy_metrics.items():
#         print(f"  {metric_name}: {value:.4f}")


# # In[ ]:


# import pandas as pd
# import matplotlib.pyplot as plt

# # ... (your existing code to load data and calculate strategies) ...

# # Get unique trading dates from your trade data
# trade_dates = pd.to_datetime(trade['date'].unique())

# first_date = trade_dates[0]
# date_before_first = first_date - pd.DateOffset(days=1)

# # Prepend the date before the first date to trade_dates
# trade_dates = pd.DatetimeIndex([date_before_first] + trade_dates.tolist())


# # Reindex your strategy results to match the trading dates
# df_assets_ppo_series = pd.Series(df_assets_ppo, index=trade_dates)
# #df_dji_normalized_close_series = pd.Series(df_dji_normalized_close, index=trade_dates) # Convert to Series


# # 1. Get dates from df_dji (Yahoo Finance data)

# dji_dates = pd.to_datetime(df_dji['date'])

# # 2. Find the intersection of trade_dates and dji_dates
# common_dates = trade_dates.intersection(dji_dates)

# # 3. Reindex df_assets_ppo to keep only common_dates
# df_assets_ppo_series = pd.Series(df_assets_ppo, index=trade_dates).reindex(common_dates)

# # Reindex df_dji to match common_dates, forward-filling missing values (if any)
# df_dji_normalized_close_series = pd.Series(df_dji_normalized_close, index=common_dates)

# # Create the DataFrame with trading dates as the index
# result = pd.DataFrame(
#     {
#         "PPO 70 epochs": df_assets_ppo_series,
#         "Nasdaq-100 index": df_dji_normalized_close_series,
#     }
# )


# # In[ ]:


# result.to_csv('result_ppo_qwen_25_epochs_20k_steps.csv')


# # In[ ]:


# # prompt: plot also the sharpe ratio and sortino ratio of nasdaq and ppo

# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# # Assuming 'result' DataFrame is already created as in your provided code

# # Calculate daily returns
# result['PPO 70 epochs_daily_return'] = result['PPO 70 epochs'].pct_change()
# result['Nasdaq-100 index_daily_return'] = result['Nasdaq-100 index'].pct_change()


# # Calculate Sharpe Ratio
# sharpe_ppo = np.sqrt(252) * (result['PPO 70 epochs_daily_return'].mean() / result['PPO 70 epochs_daily_return'].std())
# sharpe_nasdaq = np.sqrt(252) * (result['Nasdaq-100 index_daily_return'].mean() / result['Nasdaq-100 index_daily_return'].std())

# # Calculate Sortino Ratio (assuming a target return of 0)
# downside_returns_ppo = result['PPO 70 epochs_daily_return'].where(result['PPO 70 epochs_daily_return'] < 0, 0)
# downside_returns_nasdaq = result['Nasdaq-100 index_daily_return'].where(result['Nasdaq-100 index_daily_return'] < 0, 0)

# sortino_ppo = np.sqrt(252) * (result['PPO 70 epochs_daily_return'].mean() / downside_returns_ppo.std())
# sortino_nasdaq = np.sqrt(252) * (result['Nasdaq-100 index_daily_return'].mean() / downside_returns_nasdaq.std())


# #Plotting
# plt.figure(figsize=(12, 6))
# plt.plot(result.index, result['PPO 70 epochs'], label='PPO 70 epochs')
# plt.plot(result.index, result['Nasdaq-100 index'], label='Nasdaq-100 index')
# plt.title('Portfolio Value Comparison')
# plt.xlabel('Date')
# plt.ylabel('Portfolio Value')
# plt.legend()
# plt.grid(True)


# # Move legend to upper left
# plt.legend(loc='upper left')


# plt.show()


# # In[ ]:


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Assuming `df_assets_ppo_series` and `df_dji_normalized_close_series` are already created
# returns_strategy = df_assets_ppo_series.pct_change().dropna()
# returns_benchmark = df_dji_normalized_close_series.pct_change().dropna()

# # Align data
# returns_strategy, returns_benchmark = returns_strategy.align(returns_benchmark, join="inner")

# # Information Ratio
# excess_return = returns_strategy - returns_benchmark
# information_ratio = excess_return.mean() / excess_return.std()

# # CVaR
# confidence_level = 0.05
# var = np.percentile(returns_strategy, confidence_level * 100)
# cvar = returns_strategy[returns_strategy <= var].mean()

# # Rachev Ratio
# upside_confidence = 0.95
# downside_confidence = 0.05
# upside_var = np.percentile(returns_strategy, upside_confidence * 100)
# downside_var = np.percentile(returns_strategy, downside_confidence * 100)
# rachev_ratio = returns_strategy[returns_strategy >= upside_var].mean() / abs(returns_strategy[returns_strategy <= downside_var].mean())

# # Print metrics
# print(f"Information Ratio: {information_ratio:.4f}")
# print(f"CVaR (5%): {cvar:.4f}")
# print(f"Rachev Ratio: {rachev_ratio:.4f}")

# # Plot
# plt.figure(figsize=(12, 6))

# # Strategy and benchmark cumulative returns
# cumulative_strategy = (1 + returns_strategy).cumprod()
# cumulative_benchmark = (1 + returns_benchmark).cumprod()

# plt.plot(cumulative_strategy, label="PPO 25 epochs (Strategy)")
# plt.plot(cumulative_benchmark, label="Nasdaq-100 index (Baseline)")

# # Add metrics to the plot
# plt.title(f"Performance Metrics:\nIR={information_ratio:.4f}, CVaR (5%)={cvar:.4f}, Rachev={rachev_ratio:.4f}")
# plt.legend()
# plt.xlabel("Date")
# plt.ylabel("Cumulative Return")
# plt.grid()
# plt.show()


# # In[ ]:


# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# # ... (your existing code to load data and calculate strategies) ...

# # Calculate daily returns (same as before)
# # ...

# # Calculate Sharpe and Sortino Ratios (same as before)
# # ...

# # Create a figure with two subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

# # --- Subplot 1: Sharpe Ratio Comparison ---
# ax1.bar(['PPO 70 epochs', 'Nasdaq-100 index'], [sharpe_ppo, sharpe_nasdaq], color=['blue', 'orange'])
# ax1.set_title('Sharpe Ratio Comparison')
# ax1.set_ylabel('Sharpe Ratio')

# # Add Sharpe ratio values as text on top of the bars
# for i, v in enumerate([sharpe_ppo, sharpe_nasdaq]):
#     ax1.text(i, v + 0.05, f"{v:.2f}", ha='center', va='bottom')

# # --- Subplot 2: Sortino Ratio Comparison ---
# ax2.bar(['PPO 70 epochs', 'Nasdaq-100 index'], [sortino_ppo, sortino_nasdaq], color=['green', 'red'])
# ax2.set_title('Sortino Ratio Comparison')
# ax2.set_ylabel('Sortino Ratio')

# # Add Sortino ratio values as text on top of the bars
# for i, v in enumerate([sortino_ppo, sortino_nasdaq]):
#     ax2.text(i, v + 0.05, f"{v:.2f}", ha='center', va='bottom')

# plt.tight_layout()
# plt.show()


# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[ ]:


# # Plotting
# plt.figure(figsize=(16, 6))
# for column in result.columns:
#     plt.plot(result.index, result[column], label=column)

# plt.xlabel("Date")
# plt.ylabel("Portfolio Value")
# plt.title("Backtesting Results (Tradable Days Only)")


# # Get the first and last dates from the index
# first_day = result.index[0]
# last_day = result.index[-1]

# # Create a list of dates for ticks, including first, last, and every 15 days
# tick_dates = [first_day]  # Start with the first day
# current_date = first_day + pd.DateOffset(days=15)  # Add 15 days
# while current_date < last_day:
#     tick_dates.append(current_date)
#     current_date += pd.DateOffset(days=15)
# tick_dates.append(last_day)  # Add the last day


# # Remove December 13th if it's in the tick_dates list
# tick_dates = [d for d in tick_dates if d.strftime('%Y-%m-%d') != '2023-12-13']


# # Set x-axis ticks to the calculated tick_dates
# plt.xticks(tick_dates, [d.strftime('%Y-%m-%d') for d in tick_dates], rotation=45)


# #plt.xticks(rotation=45)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# # In[ ]:


# plt.rcParams["figure.figsize"] = (15,5)
# plt.figure()
# result.plot()


# # In[ ]:


# plt.rcParams["figure.figsize"] = (15,5)
# plt.figure()
# result.plot()

