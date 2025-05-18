import gym
import pybullet_envs
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3 import PPO

env = gym.make("KukaBulletEnv-v0")
env = Monitor(env, "./logs/", info_keywords=("success",))
policy_kwargs = dict(net_arch=[256, 128])
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    clip_range=0.2
)
model.learn(total_timesteps=1_000_000)
model.save("ppo_kuka_model")
env.close()
