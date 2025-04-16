import sys
import os
import numpy as np
import imageio
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO

# Ensure the UAMToyEnvironment module is accessible.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uam_toy_environment.environ.uam_toy_environment import UAMToyEnvironment

def test_uam() -> None:
    """Test the UAMToyEnvironment by running an episode with random actions and saving the renders as a GIF."""
    # Create the environment instance.
    env = UAMToyEnvironment()

    # Reset the environment.
    obs, info = env.reset()
    frames = []
    frames.append(env.render())

    num_episodes = 1

    for ep in range(num_episodes):
        done = False
        while not done:
            # Create a dictionary of random actions for each drone.
            actions = {}
            if env.drones is None:
                raise ValueError("The 'drones' attribute in the environment is None. Ensure it is properly initialized.")
            for i in env.drones:
                # Sample a random action from the continuous action space (Box).
                act = np.random.uniform(low=-env.max_accel, high=env.max_accel, size=2)
                actions[i] = act

            obs, rewards, terminated, truncated, info = env.step(actions)
            frames.append(env.render())

            # Termination if episode is done.
            if terminated or truncated:
                done = True

    # Save the collected frames as a GIF.
    imageio.mimsave("test.gif", frames, duration=0.5)
    print("Test GIF saved as test.gif")

def test_learn():
    """Test the learning process of the UAMToyEnvironment."""
    print("Here 0!")
    gym.register(
        id="UAMToyEnvironment-v0",
        entry_point="uam_toy_environment.environ.uam_toy_environment:UAMToyEnvironment",
        max_episode_steps=100,
    )
    # Create the environment instance.
    print("Here 1!")
    env = gym.make("UAMToyEnvironment-v0", render_mode="rgb_array", num_obstacles=1)
    env = gym.wrappers.FlattenObservation(env)
    obs, info = env.reset()
    print("Here 2!")
    model_A2C = A2C(
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
    # Train the model for a specified number of timesteps.
    model_A2C.learn(total_timesteps=100000, tb_log_name="test run", progress_bar=True)
    # model.learn(total_timesteps=100000, tb_log_name="second_run", reset_num_timesteps=False, progress_bar=True)
    # model.learn(total_timesteps=100000, tb_log_name="third_run", reset_num_timesteps=False, progress_bar=True)
    # Save the model.
    model_A2C.save("uam_toy")
    del model_A2C  # delete trained model to demonstrate loading

    # Load the model.
    model_A2C = PPO.load("uam_toy")
    # Manually set the environment after loading.
    model_A2C.set_env(env)
    print("Model trained and saved successfully.")

    vec_env = model_A2C.get_env()
    if vec_env is None:
        print("vec_env is still None; check that the environment was properly attached using set_env().")
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model_A2C.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
    print("DONE!!")
test_learn()
