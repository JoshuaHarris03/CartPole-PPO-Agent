# Requirements: pip install gymnasium stable-baselines3[extra] matplotlib

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import numpy as np

# Create environment
env = gym.make("CartPole-v1")
env = DummyVecEnv([lambda: env])