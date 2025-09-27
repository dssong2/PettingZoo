import os
import sys
import imageio
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Paths (adjust as needed)
MODEL_PATH = "My-UAM-Toy-Env/Data/uam_toy"             
VECNORM_PATH = "My-UAM-Toy-Env/Data/vec_normalize.pkl"  
GIF_PATH = "My-UAM-Toy-Env/Data/UAM Videos and Images/result.gif"

# Ensure the UAMToyEnvironment module is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uam_toy_environment.environ.env import UAMToy 

# Register the same env id used in training
gym.register(
    id="uam-toy-v0",
    entry_point="uam_toy_environment.environ.env:UAMToy",
    max_episode_steps=200,
)

def make_env():
    # IMPORTANT: kwargs and wrappers must match training
    env = gym.make("uam-toy-v0", render_mode="rgb_array", num_obstacles=1)
    env = gym.wrappers.FlattenObservation(env)
    return env

# Build bare VecEnv (same as training factory)
venv = DummyVecEnv([make_env])

# Load VecNormalize stats collected during training
if os.path.exists(VECNORM_PATH):
    venv = VecNormalize.load(VECNORM_PATH, venv)
    print(f"Loaded VecNormalize stats from {VECNORM_PATH}")
else:
    # Fallback: fresh normalizer (won't match training distribution)
    print(f"[WARNING] {VECNORM_PATH} not found. Using a fresh VecNormalize wrapper — "
          "evaluation returns may not match training logs.")
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True)

# Freeze stats and report unnormalized rewards (compare to W&B ep_rew_mean)
venv.training = False
venv.norm_reward = False

# Load model and attach normalized env
model = PPO.load(MODEL_PATH)
model.set_env(venv)
print("Model loaded and environment attached!")

# Rollout deterministically and collect episode returns
obs = venv.reset()
frames = []
raw_env = venv.envs[0]  # for frame grabbing
ep_ret = 0.0
ep_returns = []
n_steps = 1000

for t in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = venv.step(action)

    # Render a frame (rgb_array) and store it
    frame = raw_env.render()
    if frame is not None:
        frames.append(frame)

    ep_ret += float(rewards[0])

    if dones[0]:
        ep_returns.append(ep_ret)
        print(f"Episode {len(ep_returns)-1} reward: {ep_ret:.2f}")
        ep_ret = 0.0

# Save GIF
os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)
if frames:
    imageio.mimsave(GIF_PATH, frames, fps=30)
    print(f"Saved GIF to {GIF_PATH}")

if ep_returns:
    mean = np.mean(ep_returns)
    std = np.std(ep_returns)
    print(f"Mean rewards over {len(ep_returns)} episodes: {mean:.2f} ± {std:.2f}")
print("DONE!")
