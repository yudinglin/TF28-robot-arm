import gym
import pybullet_envs
from stable_baselines3 import PPO
import pybullet
import time

pybullet.connect(pybullet.GUI)
model = PPO.load("ppo_kuka_model")
env = gym.make("KukaBulletEnv-v0")

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()

    if _ == 200:
        width, height, rgbImg, depthImg, segImg = pybullet.getCameraImage(320, 240)
        from PIL import Image
        import numpy as np
        rgb_array = np.array(rgbImg, dtype=np.uint8).reshape((240, 320, 4))
        img = Image.fromarray(rgb_array[:, :, :3])
        img.save("ppo_simulation_result.png")
        print(" Screenshot saved")

    time.sleep(0.01)
    if done:
        obs = env.reset()

env.close()
