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
import supersuit as ss
from stable_baselines3 import PPO

# Create the PettingZoo environment
env = simple_v3.env()
env.reset()

# Convert to a single-agent Gymnasium environment
gym_env = ss.pettingzoo_env_to_vec_env_v1(env)
gym_env = ss.concat_vec_envs_v1(gym_env, 1, num_cpus=0, base_class="stable_baselines3")

# Train with SB3
model = PPO("MlpPolicy", gym_env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_simple")