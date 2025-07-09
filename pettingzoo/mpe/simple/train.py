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

import wandb
from stable_baselines3.common.callbacks import BaseCallback

class WandbMeanRewardCallback(BaseCallback):
    def __init__(self, verbose=0, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Check if a new episode has started
        if len(self.locals["infos"]) > 0:
            for info in self.locals["infos"]:
                if "episode" in info.keys():
                    self.episode_rewards.append(info["episode"]["r"])
        # Log mean reward every log_freq steps
        if self.n_calls % self.log_freq == 0 and self.episode_rewards:
            mean_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            wandb.log({"mean_reward": mean_reward, "timesteps": self.num_timesteps})
            self.episode_rewards = []
        return True
    


total_timesteps = 5e5  # Define the total timesteps for training.
# Initialize WandB for logging.
wandb.init(
    project="simple",  # Change this to your project name
    name="simple0",  # Change this to your run name
    config={"total_timesteps": total_timesteps, "algo": "PPO"},
    sync_tensorboard=True,  # Automatically sync TensorBoard logs
    monitor_gym=True,       # Monitor Gym environments
    save_code=True,
)

from pettingzoo.mpe import simple_v3
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback, EvalCallback

# Create the ParallelEnv
env = simple_v3.parallel_env()
env.reset()

# Wrap with SuperSuit
gym_env = ss.pettingzoo_env_to_vec_env_v1(env)
gym_env = ss.concat_vec_envs_v1(gym_env, 1, num_cpus=0, base_class="stable_baselines3")

# Create a separate evaluation environment
eval_env = simple_v3.parallel_env()
eval_env.reset()
eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
eval_env = ss.concat_vec_envs_v1(eval_env, 1, num_cpus=0, base_class="stable_baselines3")

eval_callback = EvalCallback(
    eval_env,
    eval_freq=5000,
    best_model_save_path="./logs/",
    log_path="./logs/",
    deterministic=True,
    render=False,
    verbose=1,  # This will print mean reward to terminal
)

# Train with SB3, showing progress bar and mean reward
model = PPO("MlpPolicy", gym_env, verbose=1, tensorboard_log="./ppo_tensorboard/")
model.learn(total_timesteps=500_000, callback=eval_callback, progress_bar=True, tb_log_name="simple_run", reset_num_timesteps=False)
model.save("ppo_simple")