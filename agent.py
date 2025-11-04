# Requirements: pip install gymnasium stable-baselines3[extra] matplotlib

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import numpy as np

# Create environment
env = gym.make("CartPole-v1")
env = DummyVecEnv([lambda: env]) # Wrap for vectorised env

# Initialise PPO model with MLP policy
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64)

# Train the model
total_timesteps = 100000
model.learn(total_timesteps=total_timesteps, progress_bar=True)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Save the model
model.save("ppo_cartpole")

# Load and test the model
del model # Remove to demonstrate loading
model = PPO.load("ppo_cartpole")

# Render and Test
obs = env.reset()
rewards = []
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    env.render()
    if done:
        obs = env.reset()

# Visualise learning curve (simulated here; in practice, use tensorboard or callbacks)
episodes = np.arrange(0, total_timesteps, 1000)
simulated_rewards = np.cumsum(np.random.normal(200, 50, len(episodes))) / (episodes / 1000 + 1)
plt.plot(episodes, simulated_rewards)
plt.title("Learning Curve")
plt.xlabel("Timesteps")
plt.ylabel("Average Reward")
plt.savefig("learning_curve.png")
plt.show()

env.close()
