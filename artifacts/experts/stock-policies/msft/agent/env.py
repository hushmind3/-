import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd 

class TradingEnv(gym.Env):
    def __init__(self, df, window_size=30, initial_balance=10_000.0, fee=0.001):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.fee = fee

        self.action_space = spaces.Discrete(3) # 0 = CELL, 1 = HOLD, 2 = BUY

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size * 6,),
            dtype=np.float32
        )

        self.reset()
    
    # return features for past 30 days
    def _get_obsercation(self):
        frame = self.df.loc[self.current_step - self.window_size:self.current_step - 1,
                            ["log_return", "sma20", "sma50", "rsi14", "macd", "volume_change"]]
        obs = frame.values.flatten()
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
            super().reset(seed=seed)

            self.current_step = self.window_size
            self.balance = self.initial_balance
            self.shares = 0
            self.last_price = self.df.loc[self.current_step, "Close"]
            self.equity = self.balance
            self.reward = 0
    

            obs = self._get_obsercation()
            info = {"balance": self.balance, "shares": self.shares, "equity": self.equity}
            return obs, info

    def step(self, action):
        price = self.df.loc[self.current_step, "Close"]
        prev_equity = self.balance + self.shares * price
        log_r = self.df.loc[self.current_step, "log_return"]
        position = 1 if self.shares > 0 else 0
        # reward = position * log_r
        reward = 0.0

        if action == 2:
            if(self.balance > 0):
                # buying with all money
                reward += 0.3 # incentive for action
                self.shares = (self.balance * (1 - self.fee)) / price
                self.balance = 0.0
                self.last_price = price
            else:
                reward -= 0.8 # tuff barrier for buying without money
        

        # selling all shares
        if action == 0:
            if(self.shares > 0):
                # reward = (price - self.last_price) / self.last_price * 100
                trade_profit = (price - self.last_price) / (self.last_price + 1e-9)
                reward += 10.0 * trade_profit

                # reward = ((self.shares * price * (1 - self.fee)) - (self.shares * self.last_price)) / (self.last_price * self.shares)
                self.balance = self.shares * price * (1 - self.fee)
                self.shares = 0

                reward += -0.02 # toll for action
                reward += 0.1
            else:
                reward -= 0.8 # tuff barrier for selling without shares

        elif action == 1:  # HOLD
            if position == 1:
                reward += 3.0 * log_r
            else:
                reward -= 0.8

        self.equity = self.balance + self.shares * price
        equity_change = (self.equity - prev_equity) / (prev_equity + 1e-9)
        reward += 10.0 * equity_change # reward for equity growth

        # pnl_step = (self.equity - prev_equity) / (prev_equity + 1e-9)
        # reward += 10.0 * pnl_step # reward for PnL

        reward += -0.2 * abs(log_r) # small penalty for volatility

        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        obs = self._get_obsercation()
        info = {"balance": self.balance, "shares": self.shares, "equity": self.equity, "price": price, "reward": reward}

        return obs, reward, terminated, truncated, info
    
    def render(self):
        print(f"Step: {self.current_step}, Equity: {self.equity:.2f}")

# testing with random operations
if __name__ == "__main__":
    from features import load_data

    df = pd.read_csv("data/MSFT_train.csv")

    env = TradingEnv(df)

    obs, info = env.reset()
    print("Initial obs shape:", obs.shape)
    print("Initial info:", info)

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Equity={info['equity']:.2f}")
        if terminated:
            print("Episode finished!")
            break