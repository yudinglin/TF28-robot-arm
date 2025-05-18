import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("logs/.monitor.csv", skiprows=1)


df["rolling_reward"] = df["r"].rolling(window=100).mean()


plt.figure(figsize=(10,5))
plt.plot(df["t"], df["rolling_reward"], label="Smoothed Reward")
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("Reward Curve During PPO Training")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("ppo_reward_curve.png")
plt.show()
