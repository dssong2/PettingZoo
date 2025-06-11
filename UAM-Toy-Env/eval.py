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

gym.register(
    id="UAMToyEnvironment-v0",
    entry_point="uam_toy_environment.environ.uam_toy_environment:UAMToyEnvironment",
    max_episode_steps=200,
)
def make_env():
    env = gym.make("UAMToyEnvironment-v0", render_mode="rgb_array", num_obstacles=1)
    env = gym.wrappers.FlattenObservation(env)
    return env

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# env = make_env()
env.training = False
env.norm_reward = False

# # Load the model.
# model_SAC = SAC.load("uam_toy")
# # Manually set the environment after loading.
# model_SAC.set_env(env)
# print("Model trained and saved successfully.")

# Load the model.
model_PPO1 = PPO.load("uam_toy")
# Manually set the environment after loading.
model_PPO1.set_env(env)
print("Model trained and saved successfully.")

vec_env = model_PPO1.get_env()
if vec_env is None:
    print("vec_env is still None; check that the environment was properly attached using set_env().")
obs = vec_env.reset()
for i in range(1000):
    action, _states = model_PPO1.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
print("DONE!!")

