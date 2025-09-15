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
from uam_toy_environment.environ.debug_env_v2 import DebugEnv

total_timesteps = 2.5e5  # Define the total timesteps for training.
# Initialize WandB for logging.
wandb.init(
    project="debug",  # Change this to your project name
    name="debug-run",  # Change this to your run name
    config={"total_timesteps": total_timesteps, "algo": "PPO"},
    sync_tensorboard=True,  # Automatically sync TensorBoard logs
    monitor_gym=True,       # Monitor Gym environments
    save_code=True,
)

# Register the DebugEnv.
gym.register(
    id="debug-v0",
    entry_point="uam_toy_environment.environ.debug_env:DebugEnv",
    max_episode_steps=200,
)
# Create the environment instance.
print("Here 1!")
def make_env():
    env = gym.make("debug-v0", render_mode="rgb_array", num_obstacles=1)
    env = Monitor(env)  # Wrap the environment with Monitor for logging.
    env = gym.wrappers.FlattenObservation(env)
    return env

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# env = make_env()

# model_A2C1 = A2C(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     tensorboard_log="./a2c_tensorboard/",
# )
model_PPO1 = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
)
# model_SAC = SAC( # try this
#     "MlpPolicy",
#     env,
#     verbose=1,
#     tensorboard_log="./sac_tensorboard/",
# )
# Train the model for a specified number of timesteps.
model_PPO1.learn(
    total_timesteps=int(total_timesteps),
    tb_log_name="test run",
    progress_bar=True,
    callback=WandbCallback()
)
# model.learn(total_timesteps=100000, tb_log_name="second_run", reset_num_timesteps=False, progress_bar=True)
# model.learn(total_timesteps=100000, tb_log_name="third_run", reset_num_timesteps=False, progress_bar=True)
# Save the model.
model_PPO1.save("uam_toy")

# env.save("vec_normalize.pkl")