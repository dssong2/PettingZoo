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

# Ensure the UAMToyEnvironment module is accessible.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Register the UAMToyEnvironment.
gym.register(
    id="simple",
    entry_point="pettingzoo.mpe.simple.simple:raw_env",
    max_episode_steps=200,
)

def make_env():
    env = gym.make("simple")
    env = Monitor(env)  # Wrap the environment with Monitor for logging.
    env = gym.wrappers.FlattenObservation(env)
    return env

env = make_env()

model_PPO1 = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
)

model_PPO1.learn(
    total_timesteps=int(total_timesteps),
    tb_log_name="test run",
    progress_bar=True,
    callback=WandbCallback()
)

model_PPO1.save("simple_test")