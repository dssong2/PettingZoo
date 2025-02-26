from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
import numpy as np
import random

class UAMToyEnvironment(ParallelEnv):
    """
    Environment in which drones begin at one site and must travel to the oppositing site without crashing into each other
    """
    metadata = {"render_modes": ["human"], "name": "uam_toy_environment_v0"}
    def __init__(self, grid_size=100, N=10, K=5, R_v=5, R_s=2, max_accel=1.0, dt=0.1, time_limit=100):
        self.grid_size = grid_size
        self.N = N
        self.K = K
        self.R_v = R_v
        self.R_s = R_s
        self.max_accel = max_accel
        self.dt = dt
        self.time_limit = time_limit
        self.current_step = 0

        super().__init__()

    def reset(self, seed = None, options = None):
        return super().reset(seed, options)
    
    def step(self, actions):
        return super().step(actions)
    
    def render(self):
        return super().render()
    
    def observation_space(self, agent):
        return super().observation_space(agent)
    
    def action_space(self, agent):
        return super().action_space(agent)