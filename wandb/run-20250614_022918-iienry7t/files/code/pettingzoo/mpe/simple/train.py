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
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe.simple.simple import raw_env

total_timesteps = 5e5  # Define the total timesteps for training.
# Initialize WandB for logging.
wandb.init(
    project="simple",  # Change this to your project name
    name="simple-test",  # Change this to your run name
    config={"total_timesteps": total_timesteps, "algo": "PPO"},
    sync_tensorboard=True,  # Automatically sync TensorBoard logs
    monitor_gym=True,       # Monitor Gym environments
    save_code=True,
)

env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)