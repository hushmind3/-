import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
from env import TradingEnv
from stable_baselines3.common.monitor import Monitor

if __name__ == "__main__":
    df = pd.read_csv("data/MSFT_train.csv")
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = DummyVecEnv([
    lambda: Monitor(TradingEnv(df, window_size=15), log_dir)
])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        tensorboard_log=log_dir
    )

    model.learn(total_timesteps=100_000)

    model.save("ppo_trader")

    print("Learning completed! Result is in ppo_trader.zip")
