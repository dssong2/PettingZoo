import sys
import os
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
from uam_toy_environment.environ.env import UAMToy

total_timesteps = 7.5e5  # Define the total timesteps for training.

# Initialize WandB for logging.
wandb.init(
    project="UAM Toy",  # Change this to your project name
    name="uam-run",  # Change this to your run name
    config={"total_timesteps": total_timesteps, "algo": "PPO"},
    sync_tensorboard=True,  # Automatically sync TensorBoard logs
    monitor_gym=True,       # Monitor Gym environments
    save_code=True,
)

# Register the UAMToy environment.
gym.register(
    id="uam-toy-v0",
    entry_point="uam_toy_environment.environ.env:UAMToy",
    max_episode_steps=200,
)
# Create the environment instance.
print("Here 1!")
def make_env():
    env = gym.make("uam-toy-v0", render_mode="rgb_array", num_obstacles=1)
    env = Monitor(env)  # Wrap the environment with Monitor for logging.
    env = gym.wrappers.FlattenObservation(env)
    return env

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

model_PPO = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
)

# Train the model for a specified number of timesteps.
model_PPO.learn(
    total_timesteps=int(total_timesteps),
    tb_log_name="test run",
    progress_bar=True,
    callback=WandbCallback()
)

# Save the model.
model_PPO.save("uam_toy")
env.save("vec_normalize.pkl")