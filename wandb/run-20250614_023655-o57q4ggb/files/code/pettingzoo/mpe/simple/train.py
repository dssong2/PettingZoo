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

# Register the UAMToyEnvironment.
gym.register(
    id="simple",
    entry_point="pettingzoo/mpe/simple/simple.py",
    max_episode_steps=200,
)
# Create the environment instance.
print("Here 1!")
def make_env():
    env = gym.make("simple", render_mode="rgb_array")
    env = Monitor(env)  # Wrap the environment with Monitor for logging.
    env = gym.wrappers.FlattenObservation(env)
    return env

env = make_env()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_simple")