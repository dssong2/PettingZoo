from pettingzoo.mpe import simple_v3
from stable_baselines3 import PPO
import time

model = PPO.load("ppo_simple")
env = simple_v3.env(render_mode="human")

num_episodes = 10

for episode in range(num_episodes):
    env.reset()
    while env.agents:
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            action, _ = model.predict(observation, deterministic=True)
        env.step(action)
        env.render()
        time.sleep(0.02)
env.close()
