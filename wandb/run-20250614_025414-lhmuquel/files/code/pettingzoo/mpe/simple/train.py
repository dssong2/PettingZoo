import sys
import os
import numpy as np
import imageio
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env

# Ensure the UAMToyEnvironment module is accessible.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple import raw_env

total_timesteps = 5e5  # Define the total timesteps for training.
# Initialize WandB for logging.
wandb.init(
    project="uam-toy-project",  # Change this to your project name
    name="uam-toy-run",  # Change this to your run name
    config={"total_timesteps": total_timesteps, "algo": "PPO"},
    sync_tensorboard=True,  # Automatically sync TensorBoard logs
    monitor_gym=True,       # Monitor Gym environments
    save_code=True,
)

from pettingzoo.mpe import simple_v3

env = simple_v3.aec_env()      # Use the AEC API
env.reset()
gym_env = env.gymnasium_env()  # Now this works

# Now you can use gym_env with SB3
from stable_baselines3 import PPO
model = PPO("MlpPolicy", gym_env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_simple")