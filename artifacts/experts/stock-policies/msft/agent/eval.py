import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from env import TradingEnv
import collections
from tensorboard import program

def evaluate_model(model, df, window_size=15):
    env = TradingEnv(df, window_size=window_size)
    obs, info = env.reset()

    agent_equity = [info["equity"]]
    actions = []

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True) 
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        agent_equity.append(info["equity"])

    return np.array(agent_equity), actions


if __name__ == "__main__":
    df_test = pd.read_csv("data/MSFT_test.csv")

    model = PPO.load("ppo_trader")

    agent_equity, actions = evaluate_model(model, df_test, window_size=15)

    action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    decoded_actions = [action_map[int(a)] for a in actions]


    print(decoded_actions[:50])

    print(collections.Counter([int(a) for a in actions]))


    start_price = df_test["Close"].iloc[0]
    bh_growth = df_test["Close"] / start_price * agent_equity[0]

    plt.plot(agent_equity, label="RL Agent")
    plt.plot(bh_growth.values, label="Buy & Hold")
    plt.legend()
    plt.title("RL Agent vs Buy&Hold")
    plt.show()

    print("\n🚀 Launching TensorBoard...")
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', './logs'])
    url = tb.launch()
    print(f"✅ TensorBoard running at: {url}")
    input("\nPress ENTER to close TensorBoard...\n")
