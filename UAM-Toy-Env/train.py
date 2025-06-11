import sys
import os
import numpy as np
import imageio
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure the UAMToyEnvironment module is accessible.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uam_toy_environment.environ.uam_toy_environment import UAMToyEnvironment

"""Test the learning process of the UAMToyEnvironment."""
print("Here 0!")
gym.register(
    id="UAMToyEnvironment-v0",
    entry_point="uam_toy_environment.environ.uam_toy_environment:UAMToyEnvironment",
    max_episode_steps=200,
)
# Create the environment instance.
print("Here 1!")
def make_env():
    env = gym.make("UAMToyEnvironment-v0", render_mode="rgb_array", num_obstacles=1)
    env = gym.wrappers.FlattenObservation(env)
    return env

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# env = make_env()

model_A2C1 = A2C(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./a2c_tensorboard/",
)
model_PPO1 = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
)
# model_SAC = SAC(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     tensorboard_log="./sac_tensorboard/",
# )
# Train the model for a specified number of timesteps.
model_PPO1.learn(total_timesteps=100000, tb_log_name="test run", progress_bar=True)
# model.learn(total_timesteps=100000, tb_log_name="second_run", reset_num_timesteps=False, progress_bar=True)
# model.learn(total_timesteps=100000, tb_log_name="third_run", reset_num_timesteps=False, progress_bar=True)
# Save the model.
model_PPO1.save("uam_toy")

# env.save("vec_normalize.pkl")