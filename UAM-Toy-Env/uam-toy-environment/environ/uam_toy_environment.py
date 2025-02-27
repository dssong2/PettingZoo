from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
from collections import OrderedDict
import numpy as np
import random

class UAMToyEnvironment(ParallelEnv):
    """
    Environment in which drones begin at one site and must travel to the oppositing site without crashing into each other
    """
    metadata = {"render_modes": ["human"], "name": "uam_toy_environment_v0"}
    def __init__(self,
                 grid_size:int = 100,
                 max_drones: int =20,
                 R_v: int = 5,
                 R_d: int = 2,
                 R_o: int = 1,
                 max_accel: float = 5.0,
                 max_steps: int = 100,
                 max_obstacles=10):
        """Create a UAMToyEnvironment environment

        Parameters
        ----------
        grid_size : int, optional
            size of the grid elements will be placed/moved on, by default 100
        max_drones : int, optional
            max number of drones, by default 20
        R_v : int, optional
            radius of vertiport, by default 5
        R_d : int, optional
            radius of drone, by default 2
        R_o : int, optional
            radius of obstacle, by default 1
        max_accel : float, optional
            max drone acceleration, by default 5.0
        max_steps : int, optional
            max number of steps, by default 100
        max_obstacles : int, optional
            max number of obstacles, by default 10
        """
        self.grid_size = grid_size
        self.max_drones = max_drones
        self.R_v = R_v
        self.R_d = R_d
        self.R_o = R_o
        self.max_accel = max_accel
        self.max_steps = max_steps
        self.max_obstacles = max_obstacles

        self.num_drones = None
        self.drones = None
        self.num_obstacles = None

        self.time_step = None
        self.vertiport1 = None
        self.vertiport2 = None
        self.obstacles_pos = None
        self.drone_pos = None
        self.drone_vel = None

        self.dt = 0.1

        super().__init__()

    def _get_obs(self):
        drone_obs = [np.zeros(80)] ## Implement properly
        return drone_obs

    def reset(self, seed = None, options = None):
        self.num_drones = random.randint(2, self.max_drones)

        self.drones = ["drone_" + str(i) for i in range(self.num_drones)]

        self.num_obstacles = random.randint(0, self.max_obstacles)

        self.time_step = 0

        # Randomized vertiports or no? Or randomize vertiports in a certain area of the grid (quad 1, quad 3)
        # Currently set to random places on the graph
        # More sophisticated version, set requried distance between two vertiports is at least some distance as a function of grid size
        mid_graph = (int) (self.grid_size / 2)
        self.vertiport1 = np.array([random.randint(0, mid_graph), random.randint(0, mid_graph)])
        self.vertiport2 = np.array([random.randint(mid_graph, self.grid_size), random.randint(mid_graph, self.grid_size)])

        self.obstacles_pos = np.empty(self.num_obstacles)
        invalid_range1_hi = self.vertiport1 + np.array([self.R_v + self.R_o, self.R_v + self.R_o])
        invalid_range1_lo = self.vertiport1 - np.array([self.R_v + self.R_o, self.R_v + self.R_o])
        invalid_range2_hi = self.vertiport2 + np.array([self.R_v + self.R_o, self.R_v + self.R_o])
        invalid_range2_lo = self.vertiport2 - np.array([self.R_v + self.R_o, self.R_v + self.R_o])
        for i in range(self.num_obstacles):
            self.obstacles_pos[i] = np.array([random.randint(0, self.grid_size), random.randint(0, self.grid_size)])
            count = 0
            # Keep running until a valid obstacle location found outside the vertiport area (square box around it)
            # Pretty sure this accounts for edge cases where vertiports are at the corners of the grid
            while True:
                if ((invalid_range1_lo[0] < self.obstacles_pos[i][0] < invalid_range1_hi[0]) and
                    (invalid_range1_lo[1] < self.obstacles_pos[i][1] < invalid_range1_hi[1]) and
                    (invalid_range2_lo[0] < self.obstacles_pos[i][0] < invalid_range2_hi[0]) and
                    (invalid_range2_lo[1] < self.obstacles_pos[i][1] < invalid_range2_hi[1])):
                    self.obstacles_pos[i] = np.array([random.randint(0, self.grid_size), random.randint(0, self.grid_size)])
                    count += 1
                    if count == 100:
                        raise Exception("Failed to find valid obstacle location, likely because the grid size is too small or by chance")
                else:
                    break
        # Allow for obstacle overlap? If not, additional logic will be needed
            # Will need to consider obstacle radius and eliminate all grid spaces taken up by an obstacle 

        self.drone_pos = np.empty(self.num_drones)
        vertiports = np.array([self.vertiport1, self.vertiport2])
        for i in range(self.num_drones):
            loc = vertiports[random.randint(0,1)]     # Changes upon how many vertiports we want, for scaling up leave as numerical
            self.drone_pos[i] = np.array([loc[0], loc[1]])
        
        # Implement _get_obs()
        drone_obs = self._get_obs()
        obs = OrderedDict({f"{a.index}": drone_obs[a.index] for a in self.drones})

        return obs
    
    def step(self, actions):
        return super().step(actions)
    
    def render(self):
        return super().render()
    
    def observation_space(self, agent):
        return super().observation_space(agent)
    
    def action_space(self, agent):
        return super().action_space(agent)