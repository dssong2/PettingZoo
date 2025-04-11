import sys
import os
import numpy as np
import imageio

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

test_uam()
