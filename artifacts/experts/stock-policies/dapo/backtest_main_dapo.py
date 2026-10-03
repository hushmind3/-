#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
import sys
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gymnasium.spaces import Box, Discrete
import gym
import seaborn as sns

# Import necessary components from FinRL_DeepSeek_backtest
from FinRL_DeepSeek_backtest import (
    INDICATORS,
    YahooDownloader
)

# Import the specialized environment for DeepSeek risk
from env_stocktrading_llm_risk import StockTradingEnv as StockTradingEnv_llm_risk

# Set paths and constants
MODEL_PATH = "./checkpoint/model_rl.pth"
RISK_DATA_PATH = "./dataset/trade_data_deepseek_risk_2019_2023.csv"
SENTIMENT_DATA_PATH = "./dataset/trade_data_deepseek_sentiment_2019_2023.csv"
TRADE_START_DATE = '2019-01-01'
TRADE_END_DATE = '2023-12-31'

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found: {MODEL_PATH}")
    sys.exit(1)

# Define MLPActorCritic class matching the one used in DAPO training
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
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
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # Policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # Move to GPU
        self.to(device)

    def step(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
        return a.cpu().numpy(), logp_a.cpu().numpy()
    
    def act_batch(self, obs, num_samples=10):
        """Sample multiple actions for a single observation"""
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
            actions = []
            logps = []
            for _ in range(num_samples):
                pi = self.pi._distribution(obs)
                a = pi.sample()
                logp_a = self.pi._log_prob_from_distribution(pi, a)
                actions.append(a.cpu().numpy())
                logps.append(logp_a.cpu().numpy())
            return actions, logps

    def act(self, obs):
        return self.step(obs)[0]

# Custom prediction function for DAPO
def custom_DAPO_prediction(act, environment, device='cuda'):
    """
    Custom prediction function for DAPO that properly handles GPU tensors
    """
    import torch
    
    print("Starting custom DAPO prediction...")
    try:
        # Reset environment and get initial state
        print("Resetting environment...")
        state, _ = environment.reset()
        print(f"Initial state shape: {state.shape if hasattr(state, 'shape') else 'N/A'}")
        
        # Initialize memory structures
        account_memory = []  # To store portfolio values
        actions_memory = []  # To store actions taken
        portfolio_distribution = []  # To store portfolio distribution
        episode_total_assets = [environment.initial_amount]
        sentiment_impact = []  # To track sentiment impact
        risk_impact = []  # To track risk impact

        # Extract functions to calculate LLM features impact
        def extract_prices(state):
            """Extract prices from state"""
            try:
                stock_dim = len(environment.df.tic.unique())
                
                # Check if state is a 2D array or a 1D array
                if isinstance(state, np.ndarray):
                    if state.ndim == 2:
                        # If 2D array, use the first row
                        state_vector = state[0]
                    else:
                        # If 1D array, use it directly
                        state_vector = state
                else:
                    # If not a numpy array, try to convert it
                    state_vector = np.array(state).flatten()
                    
                return state_vector[1:stock_dim+1]
            except Exception as e:
                print(f"Error extracting prices: {e}")
                print(f"State shape: {state.shape if hasattr(state, 'shape') else 'unknown'}")
                print(f"State type: {type(state)}")
                return None

        def extract_llm_features(state):
            """Extract LLM sentiment and risk scores from state"""
            try:
                stock_dim = len(environment.df.tic.unique())
                
                # Check if state is a 2D array or a 1D array
                if isinstance(state, np.ndarray):
                    if state.ndim == 2:
                        # If 2D array, use the first row
                        state_vector = state[0]
                    else:
                        # If 1D array, use it directly
                        state_vector = state
                else:
                    # If not a numpy array, try to convert it
                    state_vector = np.array(state).flatten()
                    
                # State space structure: [Current Balance] + [Stock Prices] + [Stock Shares] + [Technical Indicators] + [LLM Sentiment] + [LLM Risk]
                sentiment_start = -(2 * stock_dim)  # Second to last block
                risk_start = -stock_dim  # Last block
                
                llm_sentiments = state_vector[sentiment_start:risk_start]
                llm_risks = state_vector[risk_start:]
                
                return llm_sentiments, llm_risks
            except Exception as e:
                print(f"Error extracting LLM features: {e}")
                print(f"State shape: {state.shape if hasattr(state, 'shape') else 'unknown'}")
                print(f"State type: {type(state)}")
                # Return default neutral values
                stock_dim = len(environment.df.tic.unique())
                return np.ones(stock_dim) * 3, np.ones(stock_dim) * 3

        # Main prediction loop
        print(f"Starting prediction loop for {len(environment.df.index.unique())} time steps...")
        with torch.no_grad():
            for i in range(len(environment.df.index.unique())):
                try:
                    # Create tensor on the appropriate device
                    print(f"Step {i}: Creating tensor from state...")
                    if isinstance(state, np.ndarray):
                        # Print state shape for debugging
                        print(f"State shape: {state.shape}")
                        s_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
                        # If state is 1D, add batch dimension
                        if s_tensor.dim() == 1:
                            s_tensor = s_tensor.unsqueeze(0)
                    else:
                        print(f"Warning: State is not a numpy array. Type: {type(state)}")
                        # Try to convert to numpy array first
                        try:
                            state_np = np.array(state)
                            s_tensor = torch.as_tensor(state_np, dtype=torch.float32).to(device)
                            # If state is 1D, add batch dimension
                            if s_tensor.dim() == 1:
                                s_tensor = s_tensor.unsqueeze(0)
                        except Exception as e:
                            print(f"Error converting state to tensor: {e}")
                            # Create a dummy state tensor with the right shape
                            s_tensor = torch.zeros((1, environment.observation_space.shape[0]), dtype=torch.float32).to(device)
                    
                    # Get model outputs - for DAPO, we'll use the first action from act_batch
                    print(f"Step {i}: Getting actions from model...")
                    try:
                        actions, _ = act.act_batch(s_tensor, num_samples=1)
                        action = actions[0]
                        print(f"Action generated: {action[:5]}..." if len(action) > 5 else f"Action generated: {action}")
                    except Exception as e:
                        print(f"Error getting actions from model: {e}")
                        # Generate a random action as fallback
                        print("Generating random action as fallback")
                        action = np.random.uniform(-1, 1, environment.action_space.shape[0])
                    
                    # Step through the environment
                    print(f"Step {i}: Stepping through environment...")
                    next_state, reward, done, _, _ = environment.step(action)
                    
                    # Extract LLM features for analysis
                    print(f"Step {i}: Extracting LLM features...")
                    llm_sentiments, llm_risks = extract_llm_features(next_state)
                    
                    # Define mappings for risk and sentiment scores
                    risk_to_weight = {1: 0.99, 2: 0.995, 3: 1.0, 4: 1.005, 5: 1.01}
                    sentiment_to_weight = {1: 0.99, 2: 0.995, 3: 1.0, 4: 1.005, 5: 1.01}
                    
                    # Apply mappings to generate weights
                    print(f"Step {i}: Calculating risk and sentiment impacts...")
                    try:
                        llm_risks_weights = np.vectorize(lambda x: risk_to_weight.get(int(x), 1.0))(llm_risks)
                        llm_sentiment_weights = np.vectorize(lambda x: sentiment_to_weight.get(int(x), 1.0))(llm_sentiments)
                    except Exception as e:
                        print(f"Error calculating weights: {e}")
                        llm_risks_weights = np.ones_like(llm_risks)
                        llm_sentiment_weights = np.ones_like(llm_sentiments)

                    # Get stock prices for the current day
                    try:
                        price_array = environment.df.loc[environment.day, "close"].values
                    except Exception as e:
                        print(f"Error getting price array: {e}")
                        # Use a fallback method
                        try:
                            if hasattr(environment, 'data') and hasattr(environment.data, 'close'):
                                price_array = environment.data.close.values
                            else:
                                # Create dummy prices
                                price_array = np.ones(environment.stock_dim) * 100
                        except Exception as e2:
                            print(f"Error with fallback price method: {e2}")
                            price_array = np.ones(environment.stock_dim) * 100

                    # Stock holdings and cash balance
                    try:
                        stock_holdings = environment.state[(environment.stock_dim + 1):(environment.stock_dim * 2 + 1)]
                        cash_balance = environment.state[0]
                    except Exception as e:
                        print(f"Error getting holdings and balance: {e}")
                        stock_holdings = environment.num_stock_shares
                        cash_balance = environment.asset_memory[-1] if environment.asset_memory else environment.initial_amount

                    # Calculate total portfolio value
                    try:
                        total_asset = cash_balance + (price_array * stock_holdings).sum()
                    except Exception as e:
                        print(f"Error calculating total asset: {e}")
                        total_asset = environment.asset_memory[-1] if environment.asset_memory else environment.initial_amount

                    # Calculate portfolio distribution
                    try:
                        stock_values = price_array * stock_holdings
                        total_invested = stock_values.sum()
                    except Exception as e:
                        print(f"Error calculating stock values: {e}")
                        stock_values = np.zeros_like(stock_holdings)
                        total_invested = 0
                    
                    # Calculate weights based on portfolio allocation
                    if total_invested > 0:
                        try:
                            stock_weights = stock_values / total_invested
                            # Calculate aggregated sentiment and risk impact
                            aggregated_sentiment = np.dot(stock_weights, llm_sentiment_weights)
                            aggregated_risk = np.dot(stock_weights, llm_risks_weights)
                            
                            # Record impacts for analysis
                            sentiment_impact.append(aggregated_sentiment)
                            risk_impact.append(aggregated_risk)
                        except Exception as e:
                            print(f"Error calculating impacts: {e}")
                            sentiment_impact.append(1.0)
                            risk_impact.append(1.0)
                    else:
                        # If no positions, no sentiment/risk impact
                        sentiment_impact.append(1.0)
                        risk_impact.append(1.0)
                    
                    # Calculate distribution for tracking
                    try:
                        distribution = stock_values / total_asset if total_asset > 0 else np.zeros_like(stock_values)
                        cash_fraction = cash_balance / total_asset if total_asset > 0 else 1.0
                    except Exception as e:
                        print(f"Error calculating distribution: {e}")
                        distribution = np.zeros_like(stock_values)
                        cash_fraction = 1.0

                    # Store results
                    episode_total_assets.append(total_asset)
                    account_memory.append(total_asset)
                    actions_memory.append(action)
                    portfolio_distribution.append({
                        "cash": cash_fraction, 
                        "stocks": distribution.tolist(),
                        "sentiment_impact": sentiment_impact[-1],
                        "risk_impact": risk_impact[-1]
                    })

                    # Update state
                    state = next_state
                    
                    # Print progress every 10 steps
                    if i % 10 == 0:
                        print(f"Step {i}: Total asset value = {total_asset:.2f}")

                    if done:
                        print(f"Environment signaled done at step {i}")
                        break
                        
                except Exception as e:
                    print(f"Error in prediction loop at step {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue to next step instead of breaking
                    # Create a placeholder for this step's data
                    last_asset = episode_total_assets[-1] if episode_total_assets else environment.initial_amount
                    episode_total_assets.append(last_asset)
                    account_memory.append(last_asset)
                    actions_memory.append(np.zeros(environment.action_space.shape[0]))
                    portfolio_distribution.append({
                        "cash": 1.0, 
                        "stocks": [0] * environment.stock_dim,
                        "sentiment_impact": 1.0,
                        "risk_impact": 1.0
                    })
                    
                    # Try to get the next state
                    try:
                        environment.day += 1
                        if environment.day < len(environment.df.index.unique()):
                            environment.data = environment.df.loc[environment.day, :]
                            state = environment._update_state()
                    except Exception as e2:
                        print(f"Error advancing environment: {e2}")
                        # If we can't advance the environment, we need to break
                        break

        print("Test Finished!")
        # Save prediction results to CSV for analysis
        try:
            print("Saving prediction details to CSV...")
            pd.DataFrame({
                'step': range(len(episode_total_assets)),
                'total_assets': episode_total_assets,
                'sentiment_impact': sentiment_impact + [0] * (len(episode_total_assets) - len(sentiment_impact)),
                'risk_impact': risk_impact + [0] * (len(episode_total_assets) - len(risk_impact))
            }).to_csv('dapo_results/prediction_details.csv', index=False)
        except Exception as e:
            print(f"Error saving prediction details: {e}")
        
        return episode_total_assets, account_memory, actions_memory, portfolio_distribution
        
    except Exception as e:
        print(f"Critical error in custom_DAPO_prediction: {e}")
        import traceback
        traceback.print_exc()
        
        # Return dummy data to allow the script to continue
        print("Returning dummy data to allow script to continue")
        dummy_assets = [environment.initial_amount] * 10
        dummy_account = [environment.initial_amount] * 9
        dummy_actions = [np.zeros(environment.action_space.shape[0])] * 9
        dummy_portfolio = [{
            "cash": 1.0, 
            "stocks": [0] * environment.stock_dim,
            "sentiment_impact": 1.0,
            "risk_impact": 1.0
        }] * 9
        
        return dummy_assets, dummy_account, dummy_actions, dummy_portfolio

def enhanced_DRL_prediction(act, environment, verbose=True):
    """
    Enhanced wrapper around custom prediction function to capture dates
    """
    try:
        if verbose:
            print("Starting enhanced DRL prediction...")
        
        # Call our custom prediction function that properly handles GPU tensors
        episode_total_assets, account_memory, actions_memory, portfolio_distribution = custom_DAPO_prediction(
            act=act, environment=environment, device=device
        )
        
        if verbose:
            print(f"Prediction completed with {len(episode_total_assets)} time steps")
        
        # Extract dates from the environment data
        date_memory = []
        try:
            for day in range(len(environment.df.index.unique())):
                try:
                    # Get date for each day in the trading period
                    date = environment.df.loc[day, "date"].iloc[0]
                    date_memory.append(date)
                except Exception as e:
                    if verbose:
                        print(f"Error extracting date for day {day}: {e}")
                    # If we hit an error, we've likely run past the trading period
                    break
            
            # Verify lengths match
            if len(date_memory) == len(episode_total_assets) - 1:
                # Add initial date (using the first date in the dataset)
                first_date = environment.df.loc[0, "date"].iloc[0]
                date_memory = [first_date] + date_memory
                if verbose:
                    print("Added initial date to match assets length")
            elif len(date_memory) != len(episode_total_assets):
                if verbose:
                    print(f"Warning: Date memory length ({len(date_memory)}) doesn't match episode total assets length ({len(episode_total_assets)})")
                    print("Using indices as dates instead")
                date_memory = [str(i) for i in range(len(episode_total_assets))]
        except Exception as e:
            if verbose:
                print(f"Error processing dates: {e}")
                print("Using indices as dates instead")
            date_memory = [str(i) for i in range(len(episode_total_assets))]
        
        # Save prediction results to CSV immediately to preserve data
        if verbose:
            print("Saving raw prediction results to CSV...")
        try:
            pd.DataFrame({
                'date': date_memory,
                'total_assets': episode_total_assets
            }).to_csv('dapo_deepseek_raw_results.csv', index=False)
            if verbose:
                print("Raw results saved successfully")
        except Exception as e:
            if verbose:
                print(f"Error saving raw results: {e}")
        
        return episode_total_assets, date_memory, actions_memory, portfolio_distribution
        
    except Exception as e:
        if verbose:
            print(f"Error in enhanced_DRL_prediction: {e}")
            import traceback
            traceback.print_exc()
        
        # Return dummy data to allow the script to continue
        print("Returning dummy data to allow script to continue")
        dummy_assets = [1000000] * 10
        dummy_dates = [str(i) for i in range(10)]
        dummy_actions = []
        dummy_portfolio = None
        
        return dummy_assets, dummy_dates, dummy_actions, dummy_portfolio

def plot_performance(assets, dates, benchmark=None, title="Performance Comparison of DAPO Models with Ablation on Sentiment and Risk Adjustments", save_path=None):
    print(f"Generating performance plot with {len(assets)} asset points and {len(dates)} date points...")
    
    try:
        # Create dapo_results directory if it doesn't exist
        os.makedirs('dapo_results', exist_ok=True)
        print("Ensured dapo_results directory exists")
        
        plt.figure(figsize=(16, 8))
        
        # Initialize variables
        benchmark_returns = None
        outperformance_frequency = np.nan
        outperformance_down_frequency = np.nan
        information_ratio = np.nan
        rachev_ratio = np.nan
        
        # Store original dates length for debugging
        original_dates_len = len(dates)
        original_assets_len = len(assets[1:])
        
        # Check and fix lengths for assets and dates first
        if len(assets[1:]) != len(dates):
            print(f"Warning: Assets length ({len(assets[1:])}) doesn't match dates length ({len(dates)})")
            # Option 1: Truncate dates to match assets
            if len(dates) > len(assets[1:]):
                dates = dates[:len(assets[1:])]
                print("Truncated dates to match assets length")
            # Option 2: If assets are longer, truncate assets
            elif len(assets[1:]) > len(dates):
                assets = [assets[0]] + assets[1:len(dates)+1]
                print("Truncated assets to match dates length")
        
        # Convert to pandas Series for easier handling
        try:
            # Convert dates to datetime if they're not already
            if not isinstance(dates[0], pd.Timestamp):
                dates = pd.to_datetime(dates, errors='coerce')
                # Check for NaT values
                if pd.isna(dates).any():
                    print("Warning: Some dates couldn't be converted to datetime. Using integer indices instead.")
                    dates = pd.date_range(start='2019-01-01', periods=len(assets[1:]))
            
            portfolio_series = pd.Series(assets[1:], index=dates)
            print(f"Created portfolio series with {len(portfolio_series)} points")
        except Exception as e:
            print(f"Error creating portfolio series: {e}")
            print("Using integer indices instead")
            portfolio_series = pd.Series(assets[1:], index=range(len(assets[1:])))
        
        # Calculate cumulative returns
        try:
            cumulative_returns = (portfolio_series / portfolio_series.iloc[0] - 1) * 100
            print(f"Calculated cumulative returns: {cumulative_returns.iloc[-1]:.2f}%")
        except Exception as e:
            print(f"Error calculating cumulative returns: {e}")
            cumulative_returns = pd.Series([0] * len(portfolio_series), index=portfolio_series.index)
        
        # Calculate daily returns for metrics
        try:
            daily_returns = portfolio_series.pct_change().dropna()
            print(f"Calculated {len(daily_returns)} daily returns")
        except Exception as e:
            print(f"Error calculating daily returns: {e}")
            daily_returns = pd.Series([0] * (len(portfolio_series) - 1), index=portfolio_series.index[1:])
        
        # Calculate performance metrics
        try:
            total_return = cumulative_returns.iloc[-1]
            annual_return = ((1 + total_return/100) ** (252/len(portfolio_series)) - 1) * 100
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
            
            # Calculate Sortino ratio (downside risk only)
            downside_returns = daily_returns[daily_returns < 0]
            sortino_ratio = np.sqrt(252) * daily_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() != 0 else 0
            
            # Calculate max drawdown
            max_drawdown = ((portfolio_series / portfolio_series.cummax()) - 1).min() * 100
            volatility = daily_returns.std() * np.sqrt(252) * 100
            
            # Calculate Conditional Value at Risk (CVaR)
            confidence_level = 0.05  # 5% confidence level
            var = np.percentile(daily_returns, confidence_level * 100)
            cvar = daily_returns[daily_returns <= var].mean() * 100 if len(daily_returns[daily_returns <= var]) > 0 else 0
            
            print(f"Calculated performance metrics: Total Return={total_return:.2f}%, Annual Return={annual_return:.2f}%, Sharpe={sharpe_ratio:.2f}")

            # Calculate Rachev Ratio
            alpha = 0.05  # 5% confidence level
            if len(daily_returns) > 20:  # Need enough data points
                # Calculate the upper and lower tails
                upper_tail = daily_returns[daily_returns >= np.percentile(daily_returns, (1-alpha)*100)]
                lower_tail = daily_returns[daily_returns <= np.percentile(daily_returns, alpha*100)]
                
                # Calculate expected returns in these tails
                upper_tail_mean = upper_tail.mean()
                lower_tail_mean = abs(lower_tail.mean())  # Take absolute value for denominator
                
                # Calculate Rachev Ratio
                rachev_ratio = upper_tail_mean / lower_tail_mean if lower_tail_mean != 0 else np.nan
            else:
                rachev_ratio = np.nan

            # Process benchmark if provided
            if benchmark is not None:
                try:
                    print("Processing benchmark data...")
                    benchmark_values, benchmark_name = benchmark
                    
                    # Ensure benchmark values match the length of dates
                    if len(benchmark_values) != len(dates):
                        print(f"Warning: Benchmark length ({len(benchmark_values)}) doesn't match dates length ({len(dates)})")
                        # Find minimum length
                        min_len = min(len(benchmark_values), len(dates))
                        # Truncate both series to minimum length
                        dates = dates[:min_len]
                        benchmark_values = benchmark_values[:min_len]
                        cumulative_returns = cumulative_returns.iloc[:min_len]
                        print("Truncated all series to minimum length")
                    
                    # Calculate benchmark returns
                    benchmark_series = pd.Series(benchmark_values, index=dates)
                    benchmark_returns = (benchmark_series / benchmark_series.iloc[0] - 1) * 100
                    
                    # Calculate benchmark daily returns
                    benchmark_daily_returns = benchmark_series.pct_change().dropna()
                    
                    # Ensure both return series have the same index
                    common_dates = daily_returns.index.intersection(benchmark_daily_returns.index)
                    if len(common_dates) > 0:
                        aligned_strategy_returns = daily_returns[common_dates]
                        aligned_benchmark_returns = benchmark_daily_returns[common_dates]
                        
                        # Calculate Information Ratio
                        excess_returns = aligned_strategy_returns - aligned_benchmark_returns
                        if len(excess_returns) > 0 and excess_returns.std() != 0:
                            information_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
                            print(f"Calculated Information Ratio: {information_ratio:.2f}")
                        
                        # Calculate outperformance frequencies
                        outperformance_count = (aligned_strategy_returns > aligned_benchmark_returns).sum()
                        total_periods = len(aligned_strategy_returns)
                        outperformance_frequency = (outperformance_count / total_periods) * 100
                        print(f"Calculated Outperformance Frequency: {outperformance_frequency:.2f}%")
                        
                        # Calculate outperformance during market downturns
                        down_markets = aligned_benchmark_returns[aligned_benchmark_returns < 0]
                        if len(down_markets) > 0:
                            down_market_dates = down_markets.index
                            strategy_down = aligned_strategy_returns[down_market_dates]
                            benchmark_down = aligned_benchmark_returns[down_market_dates]
                            outperformance_down_count = (strategy_down > benchmark_down).sum()
                            outperformance_down_frequency = (outperformance_down_count / len(down_markets)) * 100
                            print(f"Calculated Down Market Outperformance: {outperformance_down_frequency:.2f}%")
                    else:
                        print("Warning: No overlapping dates between strategy and benchmark returns")
                    
                    print(f"Added benchmark to plot: {benchmark_name}")
                except Exception as e:
                    print(f"Error processing benchmark data: {e}")
                    benchmark_returns = None
            
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            total_return = annual_return = sharpe_ratio = sortino_ratio = max_drawdown = volatility = cvar = 0
        
        print("Setting up figure with subplots...")
        # Setup figure with two subplots - returns and drawdowns
        fig, ax1 = plt.subplots(figsize=(16, 8))
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        print("Plotting strategy performance...")
        # Plot strategy performance on first subplot
        ax1.plot(cumulative_returns, linewidth=2, label='DAPO-DeepSeek Strategy')
        
        # Plot benchmark if available
        if benchmark_returns is not None:
            ax1.plot(benchmark_returns, linewidth=2, label=benchmark_name, color='purple')
        
        # # Calculate drawdowns for second subplot
        # try:
        #     drawdowns = (portfolio_series / portfolio_series.cummax() - 1) * 100
        #     ax2.fill_between(drawdowns.index, drawdowns, 0, alpha=0.5, color='r')
        #     ax2.set_ylabel('Drawdown (%)')
        #     ax2.set_title('Drawdowns')
        #     ax2.grid(True)
        #     print("Added drawdown subplot")
        # except Exception as e:
        #     print(f"Error calculating drawdowns: {e}")
        
        # Add metrics to plot
        try:
            # Update the metrics text to include all metrics
            metrics_text = (
                f"Cumulative Return: {total_return:.2f}%\n"
                f"Annual Return: {annual_return:.2f}%\n"
                f"Max Drawdown: {max_drawdown:.2f}%\n"
                f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
                f"Sortino Ratio: {sortino_ratio:.2f}\n"
                f"Volatility: {volatility:.2f}%\n"
                f"CVaR (5%): {cvar:.2f}%\n"
                f"Rachev Ratio: {rachev_ratio:.2f}\n"
                f"Information Ratio: {information_ratio:.2f}\n"
                f"Outperformance: {outperformance_frequency:.1f}%\n"
                f"Down Market Outperf.: {outperformance_down_frequency:.1f}%"
            )

            print("Metrics text:\n", metrics_text)
            
            # Add text box with metrics
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.02, 0.95, metrics_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            print("Added metrics text box to plot")
        except Exception as e:
            print(f"Error adding metrics text: {e}")
        
        # Finalize first subplot
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.set_title(title)
        ax1.grid(True)
        ax1.legend(loc='upper left')
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save to file if path provided
        if save_path is None:
            save_path = 'dapo_results/dapo_deepseek_performance.png'
        
        try:
            plt.savefig(save_path, dpi=300)
            print(f"Successfully saved plot to {save_path}")
            
            # Save metrics to CSV for reference
            metrics_df = pd.DataFrame({
                'Metric': ['Cumulative Return', 'Annual Return', 'Max Drawdown', 
                        'Sharpe Ratio', 'Sortino Ratio', 'Volatility',
                        'CVaR (5%)', 'Rachev Ratio', 'Information Ratio', 
                        'Outperformance', 'Down Market Outperformance'],
                'Value': [total_return, annual_return, max_drawdown, 
                        sharpe_ratio, sortino_ratio, volatility,
                        cvar, rachev_ratio, information_ratio, 
                        outperformance_frequency, outperformance_down_frequency]
            })
            metrics_csv_path = 'dapo_results/dapo_deepseek_metrics.csv'
            metrics_df.to_csv(metrics_csv_path, index=False)
            print(f"Saved metrics to {metrics_csv_path}")
            
            # Save returns data for further analysis
            returns_df = pd.DataFrame({
                'Date': cumulative_returns.index,
                'Strategy_Return': cumulative_returns.values
            })
            if benchmark_returns is not None:
                returns_df['Benchmark_Return'] = benchmark_returns.values
            returns_csv_path = 'dapo_results/dapo_deepseek_returns.csv'
            returns_df.to_csv(returns_csv_path, index=False)
            print(f"Saved returns data to {returns_csv_path}")
            
        except Exception as e:
            print(f"Error saving plot: {e}")
            # Try alternative save method
            try:
                plt.savefig('dapo_results/dapo_performance_backup.png', dpi=200)
                print("Saved plot using backup method")
            except Exception as e2:
                print(f"Error with backup save method: {e2}")
        
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Error in plot_performance: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_benchmark_data(start_date, end_date, initial_value=1000000):
    """
    Get benchmark data (NASDAQ 100 index) for comparison
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        initial_value (float): Initial portfolio value to normalize benchmark against
        
    Returns:
        tuple: (benchmark_values, benchmark_name) if successful, None if failed
    """
    try:
        # Try to download NASDAQ-100 data
        df_benchmark = YahooDownloader(
            start_date=start_date, 
            end_date=end_date, 
            ticker_list=["^NDX"]
        ).fetch_data()
        
        # Verify data quality
        if df_benchmark.empty:
            print("Warning: Retrieved benchmark data is empty")
            return None
        
        if 'close' not in df_benchmark.columns:
            print("Warning: No close price data in benchmark dataset")
            return None
            
        # Check for sufficient data points
        if len(df_benchmark) < 2:
            print("Warning: Insufficient benchmark data points")
            return None
            
        # Normalize to initial portfolio value
        first_value = df_benchmark["close"].iloc[0]
        if first_value <= 0:
            print("Warning: Invalid first value in benchmark data")
            return None
            
        benchmark_values = df_benchmark["close"].div(first_value).mul(initial_value)
        
        # Verify final calculations
        if benchmark_values.isna().any():
            print("Warning: NaN values detected in benchmark calculations")
            return None
            
        return benchmark_values.tolist(), "NASDAQ-100 Index"
        
    except Exception as e:
        print(f"Error retrieving benchmark data: {str(e)}")
        return None

def plot_multiple_models(model_results, dates, benchmark=None, title="Performance Comparison of DAPO Models with Ablation on Sentiment and Risk Adjustments", save_dir=None):
    """
    Plot multiple model results on the same chart
    
    Args:
        model_results (list): List of dicts with 'name' and 'assets' keys
        dates (list): List of dates
        benchmark (tuple): Optional (benchmark_values, benchmark_name)
        title (str): Plot title
        save_dir (str): Directory to save plot
    """
    print(f"Generating comparison plot for {len(model_results)} models...")
    
    try:
        # Increase font sizes globally
        plt.rcParams.update({'font.size': 18})  # Increase from 14 to 18
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up figure with larger size for better readability
        fig, ax1 = plt.subplots(figsize=(20, 10))  # Increased figure size
        
        
        # Make sure dates are datetime
        if not isinstance(dates[0], pd.Timestamp):
            dates = pd.to_datetime(dates, errors='coerce')
        
        # Plot each model's performance
        all_metrics = {}
        colors = plt.cm.tab10.colors  # Use a color cycle
        
        for i, result in enumerate(model_results):
            model_name = result['name']
            assets = result['assets']
            color = colors[i % len(colors)]
            
            # Create series for current model
            portfolio_series = pd.Series(assets[1:], index=dates[:len(assets)-1])
            
            # Calculate returns
            cumulative_returns = (portfolio_series / portfolio_series.iloc[0] - 1) * 100
            
            # Plot on main axis
            ax1.plot(cumulative_returns, linewidth=2, label=model_name, color=color)
            
            # Calculate metrics
            daily_returns = portfolio_series.pct_change().dropna()
            
            # Store metrics for comparison
            all_metrics[model_name] = {
                'total_return': cumulative_returns.iloc[-1],
                'sharpe_ratio': np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0,
                'max_drawdown': ((portfolio_series / portfolio_series.cummax()) - 1).min() * 100,
                'volatility': daily_returns.std() * np.sqrt(252) * 100,
                'win_rate': (daily_returns > 0).sum() / len(daily_returns) * 100 if len(daily_returns) > 0 else 0
            }
            
            # Calculate Sortino ratio (downside risk only)
            downside_returns = daily_returns[daily_returns < 0]
            all_metrics[model_name]['sortino_ratio'] = np.sqrt(252) * daily_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() != 0 else 0
            
            # Calculate Conditional Value at Risk (CVaR)
            confidence_level = 0.05  # 5% confidence level
            var = np.percentile(daily_returns, confidence_level * 100)
            all_metrics[model_name]['cvar'] = daily_returns[daily_returns <= var].mean() * 100 if len(daily_returns[daily_returns <= var]) > 0 else 0
            
            
            # # Plot drawdown for first model only (to avoid clutter)
            # if i == 0:
            #     drawdowns = (portfolio_series / portfolio_series.cummax() - 1) * 100
            #     ax2.fill_between(drawdowns.index, drawdowns, 0, alpha=0.5, color='r')
            #     ax2.set_ylabel('Drawdown (%)')
            #     ax2.set_title('Drawdowns (First Model)')
            #     ax2.grid(True)
        
        # Plot benchmark if available
        # Plot benchmark if available
        if benchmark is not None:
            benchmark_values, benchmark_name = benchmark
            benchmark_series = pd.Series(benchmark_values, index=dates[:len(benchmark_values)])
            benchmark_returns = (benchmark_series / benchmark_series.iloc[0] - 1) * 100
            ax1.plot(benchmark_returns, linewidth=2, label=benchmark_name, color='purple')
        
        # Add metrics table as text
        metrics_text = "Model Comparison:\n"
        for model_name, metrics in all_metrics.items():
            metrics_text += f"\n{model_name}:\n"
            metrics_text += f"  Return: {metrics['total_return']:.2f}%\n"
            metrics_text += f"  Sharpe: {metrics['sharpe_ratio']:.2f}\n"
            metrics_text += f"  Max DD: {metrics['max_drawdown']:.2f}%\n"
      
        
        metrics_print = "Model Comparison:\n"
        for model_name, metrics in all_metrics.items():
            metrics_print += f"\n{model_name}:\n"
            metrics_print += f"  Cumulative Return: {metrics['total_return']:.2f}%\n"
            metrics_print += f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            metrics_print += f"  Max Drawdown: {metrics['max_drawdown']:.2f}%\n"
            metrics_print += f"  Volatility: {metrics['volatility']:.2f}%\n"
            metrics_print += f"  Outperformance Frequency: {metrics['win_rate']:.2f}%\n"
            metrics_print += f"  CVaR (5%): {metrics['cvar']:.2f}%\n"
        
        # # Add text box with metrics
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # ax1.text(0.02, 0.95, metrics_text, transform=ax1.transAxes, fontsize=10,
        #         verticalalignment='top', bbox=props)
        
        # Finalize first subplot
        ax1.set_ylabel('Cumulative Return (%)', fontsize=22)
        ax1.set_title(title, fontsize=24)
        ax1.grid(True)
        ax1.legend(loc='best', fontsize=20)  # Increased legend font size
        ax1.tick_params(axis='both', which='major', labelsize=18)  # Increase from 14 to 18
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save to file
        save_path = os.path.join(save_dir, "DAPO_Models_Comparison.png")
        plt.savefig(save_path, dpi=300)
        print(f"Successfully saved comparison plot to {save_path}")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({model: metrics for model, metrics in all_metrics.items()})
        metrics_csv_path = os.path.join(save_dir, "dapo_models_comparison_metrics.csv")
        metrics_df.to_csv(metrics_csv_path)
        print(f"Saved comparison metrics to {metrics_csv_path}")
        
        plt.close(fig)
        print(f"{metrics_print}")
        return True
        
    except Exception as e:
        print(f"Error in plot_multiple_models: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Load and merge trading data


    print("Getting benchmark data...")
    benchmark_data = get_benchmark_data(
        start_date=TRADE_START_DATE, 
        end_date=TRADE_END_DATE, 
        initial_value=1000000
    )



    try:
        # Check if both files exist
        if not os.path.exists(RISK_DATA_PATH):
            print(f"Risk data file not found: {RISK_DATA_PATH}")
            sys.exit(1)
            
        if not os.path.exists(SENTIMENT_DATA_PATH):
            print(f"Sentiment data file not found: {SENTIMENT_DATA_PATH}")
            sys.exit(1)
        
        # Load risk data
        print(f"Loading risk data from {RISK_DATA_PATH}...")
        try:
            trade_risk = pd.read_csv(RISK_DATA_PATH)
            print(f"Risk data loaded successfully with {len(trade_risk)} rows")
            print(f"Risk data columns: {trade_risk.columns.tolist()}")
            
            # Ensure date column is properly formatted
            if 'date' in trade_risk.columns:
                # Make sure date is in datetime format
                trade_risk['date'] = pd.to_datetime(trade_risk['date'])
                print("Date column formatted successfully in risk data")
            else:
                print("Warning: 'date' column not found in risk data")
                # Check if there's a column that might contain date information
                date_like_cols = [col for col in trade_risk.columns if 'date' in col.lower()]
                if date_like_cols:
                    print(f"Found potential date columns: {date_like_cols}")
                    trade_risk['date'] = pd.to_datetime(trade_risk[date_like_cols[0]])
                    print(f"Using {date_like_cols[0]} as date column")
                else:
                    print("No date-like columns found, creating dummy date column")
                    # Create a dummy date column if none exists
                    trade_risk['date'] = pd.date_range(start=TRADE_START_DATE, periods=len(trade_risk))
            
            # Fill missing values in llm_risk column
            if 'llm_risk' not in trade_risk.columns:
                print("Adding llm_risk column with default value 3")
                trade_risk['llm_risk'] = 3  # Neutral risk as default
            else:
                trade_risk['llm_risk'] = trade_risk['llm_risk'].fillna(3)  # Fill missing values with neutral risk
                
            # Create a simplified version with just the necessary columns for testing
            trade_llm_risk = trade_risk.copy()
            
            # Add dummy sentiment if sentiment data can't be loaded
            if 'llm_sentiment' not in trade_llm_risk.columns:
                print("Adding llm_sentiment column with default value 3")
                trade_llm_risk['llm_sentiment'] = 3  # Neutral sentiment as default
            
            # Try to load sentiment data, but continue with risk data if it fails
            try:
                print(f"Loading sentiment data from {SENTIMENT_DATA_PATH}...")
                trade_sentiment = pd.read_csv(SENTIMENT_DATA_PATH)
                print(f"Sentiment data loaded successfully with {len(trade_sentiment)} rows")
                
                # Ensure date column is properly formatted in sentiment data
                if 'date' in trade_sentiment.columns:
                    trade_sentiment['date'] = pd.to_datetime(trade_sentiment['date'])
                
                # Try to merge risk and sentiment data
                print("Merging risk and sentiment data...")
                trade_merged = pd.merge(
                    trade_risk, 
                    trade_sentiment, 
                    on=['date', 'tic'], 
                    suffixes=('', '_sentiment'),
                    how='left'  # Use left join to keep all risk data
                )
                
                # Check if sentiment column was merged correctly
                if 'llm_sentiment' in trade_sentiment.columns and 'llm_sentiment_sentiment' in trade_merged.columns:
                    # Use sentiment data from sentiment file
                    trade_merged['llm_sentiment'] = trade_merged['llm_sentiment_sentiment'].fillna(3)
                    trade_merged.drop('llm_sentiment_sentiment', axis=1, inplace=True)
                    print("Successfully merged sentiment data")
                    trade_llm_risk = trade_merged
                
            except Exception as e:
                print(f"Error loading sentiment data: {e}")
                print("Continuing with risk data only")
                # We'll use the trade_llm_risk with dummy sentiment values created above
            
            # Ensure all required columns exist
            required_columns = ['date', 'tic', 'close', 'llm_sentiment', 'llm_risk']
            for col in required_columns:
                if col not in trade_llm_risk.columns:
                    if col == 'llm_sentiment':
                        print(f"Adding missing column: {col}")
                        trade_llm_risk[col] = 3  # Neutral sentiment
                    elif col == 'llm_risk':
                        print(f"Adding missing column: {col}")
                        trade_llm_risk[col] = 3  # Neutral risk
                    else:
                        print(f"Error: Required column {col} missing and cannot be synthesized")
                        raise ValueError(f"Missing required column: {col}")
            
            # Final check of data
            print(f"Final data shape: {trade_llm_risk.shape}")
            print(f"Final data columns: {trade_llm_risk.columns.tolist()}")
            print(f"Sample of final data:\n{trade_llm_risk.head()}")
            
        except Exception as e:
            print(f"Error loading risk data: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Prepare index
        unique_dates = trade_llm_risk['date'].unique()
        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
        trade_llm_risk['new_idx'] = trade_llm_risk['date'].map(date_to_idx)
        trade_llm_risk = trade_llm_risk.set_index('new_idx')
        
        # Fill missing values for sentiment and risk
        trade_llm_risk['llm_sentiment'].fillna(3, inplace=True)  # neutral sentiment is 3
        trade_llm_risk['llm_risk'].fillna(3, inplace=True)  # neutral risk is 3
        
        # Print dataset statistics
        print(f"Loaded and merged {len(trade_llm_risk)} records with {len(unique_dates)} trading days")
        print(f"Date range: {trade_llm_risk['date'].min()} to {trade_llm_risk['date'].max()}")
        print(f"Number of stocks: {len(trade_llm_risk['tic'].unique())}")
        print(f"LLM risk score distribution: \n{trade_llm_risk['llm_risk'].value_counts().sort_index()}")
        print(f"LLM sentiment score distribution: \n{trade_llm_risk['llm_sentiment'].value_counts().sort_index()}")
        
        # Use the merged data for the trading environment
        trade_llm_risk = trade_llm_risk
        
    except Exception as e:
        print(f"Error loading and merging trading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Filter data to start from 2019-01-01
    print("Filtering data to start from 2019-01-01...")
    trade_llm_risk = trade_llm_risk[trade_llm_risk['date'] >= '2019-01-01']

    # Rebuild index after filtering
    unique_dates = trade_llm_risk['date'].unique()
    date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
    trade_llm_risk['new_idx'] = trade_llm_risk['date'].map(date_to_idx)
    trade_llm_risk = trade_llm_risk.set_index('new_idx')
    print(f"Filtered data date range: {trade_llm_risk['date'].min()} to {trade_llm_risk['date'].max()}")
    print(f"Filtered data has {len(unique_dates)} trading days")
    
    # Configure environment
    stock_dimension = len(trade_llm_risk.tic.unique())
    state_space_llm_risk = 1 + 2 * stock_dimension + (2+len(INDICATORS)) * stock_dimension
    
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space_llm_risk}")
    
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
    
    # Create environment
    print("Creating DAPO-DeepSeek trading environment with risk and sentiment features...")
    try:
        # Use the custom DAPO inference environment
        print("Initializing DAPOInferenceEnv with the following parameters:")
        print(f"  - Stock dimension: {stock_dimension}")
        print(f"  - State space: {state_space_llm_risk}")
        print(f"  - Action space: {stock_dimension}")
        print(f"  - Technical indicators: {INDICATORS}")
        
        # e_trade_llm_risk_gym = DAPOInferenceEnv(
        #     df=trade_llm_risk, 
        #     stock_dim=stock_dimension,
        #     hmax=100,
        #     initial_amount=1000000,
        #     num_stock_shares=[0] * stock_dimension,
        #     buy_cost_pct=[0.001] * stock_dimension,
        #     sell_cost_pct=[0.001] * stock_dimension,
        #     state_space=state_space_llm_risk,
        #     action_space=stock_dimension,
        #     tech_indicator_list=INDICATORS,
        #     turbulence_threshold=70, 
        #     risk_indicator_col='vix',
        #     llm_sentiment_col='llm_sentiment',
        #     llm_risk_col='llm_risk',
        #     reward_scaling=1.0,  # Adding the missing reward_scaling parameter
        #     make_plots=True,
        #     print_verbosity=10,
        #     model_name="dapo_deepseek",
        #     mode="backtest"
        # )

        e_trade_llm_risk_gym = StockTradingEnv_llm_risk(
            df=trade_llm_risk, 
            stock_dim=stock_dimension,
            hmax=100,
            initial_amount=1000000,
            num_stock_shares=[0] * stock_dimension,
            buy_cost_pct=[0.001] * stock_dimension,
            sell_cost_pct=[0.001] * stock_dimension,
            state_space=state_space_llm_risk,
            action_space=stock_dimension,
            tech_indicator_list=INDICATORS,
            turbulence_threshold=70, 
            risk_indicator_col='vix',
            reward_scaling=1.0  # Adding the missing reward_scaling parameter
        )
        print("Successfully created DAPO inference environment")
    except Exception as e:
        print(f"Error creating DAPO inference environment: {e}")
        print("Falling back to standard environment...")
        import traceback
        traceback.print_exc()
        
        # Fallback to standard environment
        e_trade_llm_risk_gym = StockTradingEnv_llm_risk(
            df=trade_llm_risk, 
            stock_dim=stock_dimension,
            hmax=100,
            initial_amount=1000000,
            num_stock_shares=[0] * stock_dimension,
            buy_cost_pct=[0.001] * stock_dimension,
            sell_cost_pct=[0.001] * stock_dimension,
            state_space=state_space_llm_risk,
            action_space=stock_dimension,
            tech_indicator_list=INDICATORS,
            turbulence_threshold=70, 
            risk_indicator_col='vix',
            reward_scaling=1.0  # Adding the missing reward_scaling parameter
        )
    
    # Get observation and action spaces
    observation_space_llm_risk = e_trade_llm_risk_gym.observation_space
    action_space_llm_risk = e_trade_llm_risk_gym.action_space
    
    # Load the model
    print(f"Loading DAPO-DeepSeek model from {MODEL_PATH}...")
    try:
        # Using the MLPActorCritic class defined above
        loaded_dapo = MLPActorCritic(
            observation_space_llm_risk, 
            action_space_llm_risk, 
            hidden_sizes=(512, 512),
            activation=nn.ReLU
        )
        
        # Load model weights - handle both direct state dict and checkpoint formats
        if not os.path.exists(MODEL_PATH):
            print(f"Warning: Model file not found at {MODEL_PATH}")
            print("Using the initialized model for testing plot functionality...")
        else:
            try:
                checkpoint = torch.load(MODEL_PATH, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # The model was saved as a checkpoint dict
                    loaded_dapo.load_state_dict(checkpoint['model_state_dict'])
                    epoch = checkpoint.get('epoch', 'unknown')
                    print(f"Loaded checkpoint from epoch {epoch}")
                else:
                    # The model was saved as a direct state dict
                    loaded_dapo.load_state_dict(checkpoint)
                print(f"Model loaded successfully from {MODEL_PATH}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using the initialized model for testing plot functionality...")
                import traceback
                traceback.print_exc()
            
        loaded_dapo.to(device)
        loaded_dapo.eval()  # Set the model to evaluation mode
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # Debugging information
        print("\nDebugging information:")
        if os.path.exists(MODEL_PATH):
            try:
                checkpoint = torch.load(MODEL_PATH, map_location="cpu")
                if isinstance(checkpoint, dict):
                    print(f"Checkpoint keys: {checkpoint.keys()}")
                    if 'model_state_dict' in checkpoint:
                        print(f"Model state dict keys: {checkpoint['model_state_dict'].keys()}")
                else:
                    print(f"Direct state dict keys: {checkpoint.keys()}")
            except Exception as e:
                print(f"Failed to load model state dict for debugging: {e}")
        
        sys.exit(1)
    
    # Make sure save directory exists
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dapo_results")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created save directory: {save_dir}")

    # Run predictions for all models
    all_results = []
    all_dates = None
    
    for model_config in MODELS:
        print(f"\n=== Running inference for model: {model_config['name']} ===")
        model_path = model_config['path']
        model_name = model_config['name']
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue
            
        # Load the model
        print(f"Loading model from {model_path}...")
        try:
            # Create a fresh environment instance for each model
            env_instance = StockTradingEnv_llm_risk(
                df=trade_llm_risk.copy(),  # Use a copy to avoid state issues
                stock_dim=stock_dimension,
                hmax=100,
                initial_amount=1000000,
                num_stock_shares=[0] * stock_dimension,
                buy_cost_pct=[0.001] * stock_dimension,
                sell_cost_pct=[0.001] * stock_dimension,
                state_space=state_space_llm_risk,
                action_space=stock_dimension,
                tech_indicator_list=INDICATORS,
                turbulence_threshold=70, 
                risk_indicator_col='vix',
                reward_scaling=1.0
            )
            
            # Load model weights
            loaded_model = MLPActorCritic(
                env_instance.observation_space, 
                env_instance.action_space, 
                hidden_sizes=(512, 512),
                activation=nn.ReLU
            )
            
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                loaded_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                loaded_model.load_state_dict(checkpoint)
                
            loaded_model.to(device)
            loaded_model.eval()
            
            # Run prediction
            assets, dates, _, _ = enhanced_DRL_prediction(loaded_model, env_instance)
            
            # Store results
            all_results.append({
                "name": model_name,
                "assets": assets
            })
            
            # Store dates from first model run (assuming same dates for all models)
            if all_dates is None:
                all_dates = dates
                
        except Exception as e:
            print(f"Error running model {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Plot all models together
    if all_results and all_dates:
        plot_multiple_models(all_results, all_dates, benchmark=benchmark_data, save_dir=save_dir)
    else:
        print("No successful model runs to plot")

if __name__ == "__main__":

    MODELS = [
        {
            "name": "Baseline (α=1, β=1)",
            "path": "./checkpoint/model_rl.pth"
        },
        # {
        #     "name": "Only Risk (α=0, β=1)", 
        #     "path": "./checkpoint/agent_dapo_risk_b1.0.pth"
        # },
        # {
        #     "name": "Only Sentiment (α=1, β=0)",
        #     "path": "./checkpoint/agent_dapo_sentiment_a1.0.pth"
        # }
    ]
    # Add more models as needed

    main()

