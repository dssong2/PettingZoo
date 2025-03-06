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
                 num_drones: int =20,
                 S_v: int = 5,
                 S_d: int = 2,
                 S_o: int = 1,
                 max_accel: float = 5.0,
                 max_steps: int = 100,
                 max_obstacles=10,
                 num_obstacles=10,
                 num_vertiports=2,
                 vertiports_loc=np.array([(1, 2), (3, 4)])):
                
        """Create a UAMToyEnvironment environment

        Parameters
        ----------
        grid_size : int optional
            size of the grid, by default 100
        num_drones : int optional
            number of drones, by default 20
        S_v : int, optional
            side length of vertiport, by default 5
        S_d : int, optional
            side length of drone, by default 2
        S_o : int, optional
            side length of obstacle, by default 1
        max_accel : float, optional
            maximum acceleration applied to drone, by default 5.0
        max_steps : int, optional
            maximum number of steps for an episode, by default 100
        max_obstacles : int, optional
            maximum number of obstacles spawned, by default 10
        num_vertiports : int, optional
            total number of vertiports spawned, by default 2
        vertiports_loc : NDArray containing tuples, optional
            (x,y) location of vertiports on grid, by default np.array([(1, 2), (3, 4)])
        """
        self.grid_size = grid_size
        self.num_drones = num_drones
        self.S_v = S_v
        self.S_d = S_d
        self.S_o = S_o
        self.max_accel = max_accel
        self.max_steps = max_steps
        self.max_obstacles = max_obstacles
        self.num_obstacles = num_obstacles # Added this parameter so I can get properly define get_state with num obstacles
        self.num_vertiports = num_vertiports
        self.vertiports_loc = vertiports_loc # np.array of tuples (x, y)
        assert vertiports_loc.shape == num_vertiports, "Number of vertiport coordinates do not correspond to number of vertiports"

        self.drones = None
        self.time_step = None
        self.vertiport1 = None
        self.vertiport2 = None
        self.obstacles_pos = None # np.array of tuples (x, y)
        self.drone_pos = None # np.array of tuples (x, y)
        self.drone_vel = None # np.array of tuples (x, y)

        self.dt = 0.1

        super().__init__()

    def _get_state(self): # add a boolean crashed into state? can use in _get_reward()
        if self.drone_pos is None or self.drone_vel is None or self.obstacles_pos is None:
            raise RuntimeError("Environment not initialized; call reset() first.")
        state_list = []
        state_list.extend(self.drone_pos)
        state_list.extend(self.drone_vel)
        state_list.extend(self.obstacles_pos)
        state_list.extend([np.array(v) for v in self.vertiports_loc])
        return state_list


    def _get_obs(self): 
        if self.drone_pos is None or self.drone_vel is None or self.obstacles_pos is None:
            raise RuntimeError("Environment not initialized; call reset() first.")
        
        obs = {}

        for i in range(self.num_drones):
            drone_pos : tuple = self.drone_pos[i]
            drone_vel : tuple = self.drone_vel[i]

            rel_drone_positions = []
            rel_drone_velocities = []
            for j in range(self.num_drones):
                if j != i:
                    rel_pos = np.array(self.drone_pos[j]) - np.array(drone_pos)
                    rel_vel = np.array(self.drone_vel[j]) - np.array(drone_vel)
                    rel_drone_positions.append(rel_pos)
                    rel_drone_velocities.append(rel_vel)

            rel_obst_positions = [np.array(obst) - np.array(drone_pos) for obst in self.obstacles_pos]
            rel_obst_velocities = [-np.array(drone_vel) for _ in self.obstacles_pos]

            rel_vert_positions = [np.array(vp) - np.array(drone_pos) for vp in self.vertiports_loc]
            rel_vert_velocities = [-np.array(drone_vel) for _ in self.vertiports_loc]

            obs[i] = { # Name of drone is simply the index, same as in drones[] defined in reset()
                "rel_drone_positions": np.array(rel_drone_positions),
                "rel_drone_velocities": np.array(rel_drone_velocities),
                "rel_obst_positions": np.array(rel_obst_positions),
                "rel_obst_velocities" : np.array(rel_obst_velocities),
                "rel_vert_positions": np.array(rel_vert_positions),
                "rel_vert_velocities" : np.array(rel_vert_velocities)
            }


        # relative position, rel, locations of vertipoints, location of obstalces
        # obs stored as dictionary, each drone is a key value, other part is the array e.g drone1 : rel pos, rel vel...
        return obs
    
    def _get_reward(self, agent, state, action):
        # state = self._get_state() when called in step, use the crashed boolean to help determine reward
        return 1

    def _is_overlapping(self, obj1_loc, obj1_side, obj2_loc, obj2_side):
        invalid_range_hi = obj1_loc + np.array([obj1_side + obj2_side, obj1_side + obj2_side])
        invalid_range_lo = obj1_loc - np.array([obj1_side + obj2_side, obj1_side + obj2_side])
        if ((invalid_range_lo[0] < obj2_loc[0] < invalid_range_hi[0]) and
            (invalid_range_lo[1] < obj2_loc[1] < invalid_range_hi[1])): # if obj2 overlaps with obj1
            return True
        return False
    
    def reset(self, seed = None, options = None):
        super().reset(seed, options)

        self.drones = [i for i in range(self.num_drones)]

        self.num_obstacles = random.randint(0, self.max_obstacles)

        self.time_step = 0

        # Randomized vertiports or no? Or randomize vertiports in a certain area of the grid (quad 1, quad 3)
        # Currently set to random places on the graph
        # More sophisticated version, set requried distance between two vertiports is at least some distance as a function of grid size
        mid_graph = (int) (self.grid_size / 2)
        self.vertiport1 = np.array([random.randint(0, mid_graph), random.randint(0, mid_graph)])
        self.vertiport2 = np.array([random.randint(mid_graph, self.grid_size), random.randint(mid_graph, self.grid_size)])

        self.obstacles_pos = np.empty(self.num_obstacles)
        for i in range(self.num_obstacles):
            self.obstacles_pos[i] = np.array([random.randint(0, self.grid_size), random.randint(0, self.grid_size)])
            count = 0
            # Keep running until a valid obstacle location found outside the vertiport area (square box around it)
            # Pretty sure this accounts for edge cases where vertiports are at the corners of the grid
            while True:
                if (self._is_overlapping(self.vertiport1, self.S_v, self.obstacles_pos[i], self.S_o)):
                    self.obstacles_pos[i] = np.array([random.randint(0, self.grid_size), random.randint(0, self.grid_size)])
                    count += 1
                    if count >= 100:
                        raise Exception("Failed to find valid obstacle location, likely because the grid size is too small or by chance")
                else:
                    break
        # Allow for obstacle overlap? If not, additional logic will be needed
            # Will need to consider obstacle radius and eliminate all grid spaces taken up by an obstacle 
        # Create a function that determines if two objects are overlapping, takes in two radii, for drones, obstacles, or vertipoints

        self.drone_pos = np.empty(self.num_drones)
        for i in range(self.num_drones):
            loc = self.vertiports_loc[random.randint(0,self.num_vertiports - 1)]     # Changes upon how many vertiports we want, for scaling up leave as numerical
            self.drone_pos[i] = (loc[0], loc[1])

        self.drone_vel = np.empty(self.num_drones)
        for i in range(self.num_drones):
            self.drone_vel[i] = (0.0, 0.0) # Set initial velocity to 0
        
        # Implement _get_obs()
        drone_obs = self._get_obs()
        # obs = OrderedDict({f"{a}": drone_obs[a] for a in self.drones}) do I need this line if my key is already defined in the dictionary drone_obs?

        return drone_obs
    
    def step(self, actions):
        super().step(actions)
        obs = self._get_obs()
        reward = ...
        return obs, reward
    
    def render(self):
        return super().render()
    
    def observation_space(self, agent : int): # agent is int because we define the names of drones by numbers
        super().observation_space(agent)
        obs = self._get_obs() # obs type 2D dictionary, where each drone has a dictionary of individual observations
        return obs[agent] # returns a 1D dictionary, with {"rel_pos": x, "rel_vel": y, ...}
    
    def action_space(self, agent):
        super().action_space(agent)
        action = Discrete(4)
        return action