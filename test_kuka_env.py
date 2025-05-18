import gym
import pybullet_envs
import pybullet  
import time

pybullet.connect(pybullet.GUI)  

env = gym.make("KukaBulletEnv-v0")
obs = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.01)  # time stop
    if done:
        obs = env.reset()
env.close()
