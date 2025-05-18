🤖 Environment
We use a customized Gym environment KukaBulletEnv-v0, based on PyBullet’s KUKA simulation. It supports continuous action space and includes reward shaping, success detection, and logging

install Microsoft Visual Studio with C++ workload

Make sure the following key libraries are installed:
*pybullet
*gym
*pandas
*matplotlib
*stable-baselines3

⚠️ Note: Compatible Python version 3.10. Other versions may cause compatibility issues.



🛠️ Project Structure
File	Purpose
train_robot.py	          Train the PPO agent with KukaBulletEnv and log rewards
test_model.py	            Load the trained model and test its behavior visually
plot_reward_curve.py	    Plot the reward learning curve from training logs
evaluate_agents.py	      Compare performance of PPO agent vs random agent
test_kuka_env.py	        Visual test: run environment with random actions
logs/monitor.csv	        Stores training rewards, success status, etc.
ppo_kuka_model.zip	      Trained model weights
