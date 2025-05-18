import gym
import pybullet_envs
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

def evaluate_agent(agent, agent_type, episodes=100):
    env = gym.make("KukaBulletEnv-v0")
    success_count = 0
    steps_list = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        steps = 0

        while not done:
            if agent_type == "ppo":
                action, _ = agent.predict(obs)
            elif agent_type == "random":
                action = env.action_space.sample()
            else:
                raise ValueError("Unknown agent type")

            obs, reward, done, _ = env.step(action)
            steps += 1

        steps_list.append(steps)
        if reward > 0:  
            success_count += 1

    env.close()
    return success_count, steps_list

# Load trained PPO model
ppo_model = PPO.load("ppo_kuka_model")

# Evaluate both agents
ppo_success, ppo_steps = evaluate_agent(ppo_model, "ppo")
rand_success, rand_steps = evaluate_agent(None, "random")

# Calculate metrics
ppo_rate = ppo_success / 100
rand_rate = rand_success / 100
ppo_avg_steps = sum(ppo_steps) / 100
rand_avg_steps = sum(rand_steps) / 100

print(f"PPO Success Rate: {ppo_rate:.2f}, Avg Steps: {ppo_avg_steps:.2f}")
print(f"Random Success Rate: {rand_rate:.2f}, Avg Steps: {rand_avg_steps:.2f}")

# Plot success rate comparison
plt.figure(figsize=(6, 5))
plt.bar(["PPO", "Random"], [ppo_rate, rand_rate], color=["skyblue", "salmon"])
plt.title("Success Rate Comparison")
plt.ylabel("Success Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("success_rate_real.png")

# Plot average steps comparison
plt.figure(figsize=(6, 5))
plt.bar(["PPO", "Random"], [ppo_avg_steps, rand_avg_steps], color=["skyblue", "salmon"])
plt.title("Average Steps to Complete Task")
plt.ylabel("Average Steps")
plt.tight_layout()
plt.savefig("avg_steps_real.png")

print(" Plots saved: success_rate_real.png and avg_steps_real.png")
