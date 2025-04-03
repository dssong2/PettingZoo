from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box, Dict
from collections import OrderedDict
import numpy as np
import random


class UAMToyEnvironment(ParallelEnv):
    """
    Environment in which drones begin at one site and must travel to the oppositing site without crashing into each other
    """

    metadata = {"render_modes": ["human"], "name": "uam_toy_environment_v0"}

    def __init__(
        self,
        grid_size: int = 100,
        num_drones: int = 20,
        S_v: int = 5,
        S_d: int = 2,
        S_o: int = 1,
        max_accel: float = 5.0,
        max_vel : float = 500.0,
        max_steps: int = 100,
        max_obstacles=10,
        num_obstacles=10,
        num_vertiports=2,
        vertiports_loc=np.array([(1, 2), (3, 4)]),
        safety_radius=5,
    ):
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
        max_vel : float, optional
            maximum velocity of drone, by default 500.0
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
        self.max_vel = max_vel
        self.max_steps = max_steps
        self.max_obstacles = max_obstacles
        self.num_obstacles = num_obstacles  # Added this parameter so I can get properly define get_state with num obstacles
        self.num_vertiports = num_vertiports
        self.vertiports_loc = vertiports_loc  # np.array of tuples (x, y)
        assert (
            vertiports_loc.shape == num_vertiports
        ), "Number of vertiport coordinates do not correspond to number of vertiports"
        self.safety_radius = safety_radius

        self.drones = None # can initialize in innit
        self.destinations_loc = None  # np.array of tuples (x, y), randomized for more than 2 vertiports, implement after testing 2 vertiports
        self.time_step = None
        self.vertiport1 = None
        self.vertiport2 = None
        self.obstacles_pos = None  # np.array of tuples (x, y)
        self.drone_pos = None  # np.array of tuples (x, y)
        self.drone_vel = None  # np.array of tuples (x, y)
        self.drone_vertiport = None  # np.array of each drone's starting vertiport

        self.dt = 0.1

        super().__init__()

    def _get_state(self):
        """Get the current state of the environment.

        Returns
        -------
        list
            List containing the drone positions, drone velocities, obstacle positions, and vertiport locations.

        Raises
        ------
        RuntimeError
            If the environment has not been initialized; call reset() first.
        """
        if (
            self.drone_pos is None
            or self.drone_vel is None
            or self.obstacles_pos is None
        ):
            raise RuntimeError("Environment not initialized; call reset() first.")
        state_list = []
        state_list.append(self.drone_pos)
        state_list.append(self.drone_vel)
        state_list.append(self.obstacles_pos)
        state_list.append(self.vertiports_loc)
        return state_list  # change state_list to a dictionary, or vice-versa for _get_obs (keep the output type the same)

    def _get_obs(self):
        """Get the current observations for all drones.

        Returns
        -------
        dict
            Dictionary containing the observations for all drones. The keys are the drone indices and the values are dictionaries containing the observations for each drone.

        Raises
        ------
        RuntimeError
            If the environment has not been initialized; call reset() first.
        """
        if (
            self.drone_pos is None
            or self.drone_vel is None
            or self.obstacles_pos is None
        ):
            raise RuntimeError("Environment not initialized; call reset() first.")

        obs = {}

        for i in range(self.num_drones):
            drone_pos: tuple = self.drone_pos[i]
            drone_vel: tuple = self.drone_vel[i]

            rel_drone_positions = []
            rel_drone_velocities = []
            for j in range(self.num_drones):
                if j != i:
                    rel_pos = np.array(self.drone_pos[j]) - np.array(drone_pos)
                    rel_vel = np.array(self.drone_vel[j]) - np.array(drone_vel)
                    rel_drone_positions.append(rel_pos)
                    rel_drone_velocities.append(rel_vel)

            rel_obst_positions = [
                np.array(obst) - np.array(drone_pos) for obst in self.obstacles_pos
            ]
            rel_obst_velocities = [-np.array(drone_vel) for _ in self.obstacles_pos]

            rel_vert_positions = [
                np.array(vp) - np.array(drone_pos) for vp in self.vertiports_loc
            ]
            rel_vert_velocities = [-np.array(drone_vel) for _ in self.vertiports_loc]

            obs[i] = (
                {  # Name of drone is simply the index, same as in drones[] defined in reset()
                    "rel_drone_positions": np.array(rel_drone_positions),
                    "rel_drone_velocities": np.array(rel_drone_velocities),
                    "rel_obst_positions": np.array(rel_obst_positions),
                    "rel_obst_velocities": np.array(rel_obst_velocities),
                    "rel_vert_positions": np.array(rel_vert_positions),
                    "rel_vert_velocities": np.array(rel_vert_velocities),
                }
            )

        # relative position, rel, locations of vertipoints, location of obstalces
        # obs stored as dictionary, each drone is a key value, other part is the array e.g drone1 : rel pos, rel vel...
        return obs

    def _get_reward(self, agent: int, obs_initial: dict, obs_next: dict, action):
        """Compute the reward for a single drone (agent) based on the initial and next observations and the action taken.

        Parameters
        ----------
        agent : int
            Index of the drone for which to compute the reward.
        obs_initial : dict
            Dictionary containing the initial observations for all drones.
        obs_next : dict
            Dictionary containing the next observations for all drones.
        action : np.array
            Action taken by the agent.

        Returns
        -------
        float
            Reward for the agent.

        Raises
        ------
        RuntimeError
            If the environment has not been initialized; call reset() first.
        """
        if (
            self.drone_pos is None
            or self.drone_vel is None
            or self.obstacles_pos is None
            or self.num_drones is None
            or self.drone_vertiport is None
        ):
            raise RuntimeError("Environment not initialized; call reset() first.")
        ## vv WHY IS IT AN ARRAY OF TUPLE?? does this actually just work??
        drone_starting_vertiport: int = self.drone_vertiport[agent]
        ## For more than two vertiports, which vertiport does each drone go to??

        # Do +1 to get the correct vertiport index, since the vertiport index is 0-indexed
        initial_pos = obs_initial[agent]["rel_vert_positions"][
            self.num_vertiports - (drone_starting_vertiport + 1)
        ]
        next_pos = obs_next[agent]["rel_vert_positions"][
            self.num_vertiports - (drone_starting_vertiport + 1)
        ]
        # if moving closer to vertiport, positive reward, negative if moving farther away
        reward_goal = -np.linalg.norm(next_pos) + np.linalg.norm(initial_pos)
        agent_collision = 0.0
        obstacle_collision = 0.0
        for i in range(self.num_drones):
            if i != agent:
                if self._is_overlapping(
                    self.drone_pos[agent],
                    self.S_d + self.safety_radius,
                    self.drone_pos[i],
                    self.S_d,
                ):
                    agent_collision += -100.0
        for i in range(self.num_obstacles):
            if self._is_overlapping(  ## implement safety radius here
                self.drone_pos[agent],
                self.S_d + self.safety_radius,
                self.obstacles_pos[i],
                self.S_o,
            ):
                obstacle_collision += -50.0
        reward = reward_goal + agent_collision + obstacle_collision

        return reward

    def _is_overlapping(
        self, obj1_loc: tuple[int, int], obj1_side: int, obj2_loc: tuple, obj2_side
    ):
        invalid_range_hi = np.array(obj1_loc) + np.array(
            [obj1_side + obj2_side, obj1_side + obj2_side]
        )  # Convert tuples to np.array so arithmetic works,
        invalid_range_lo = np.array(obj1_loc) - np.array(
            [obj1_side + obj2_side, obj1_side + obj2_side]
        )
        if (invalid_range_lo[0] < obj2_loc[0] < invalid_range_hi[0]) and (
            invalid_range_lo[1] < obj2_loc[1] < invalid_range_hi[1]
        ):  # if obj2 overlaps with obj1
            return True
        return False

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state.

        Parameters
        ----------
        seed : _type_, optional
            _description_, by default None
        options : _type_, optional
            _description_, by default None

        Returns
        -------
        dict
            Dictionary containing the initial observations for all drones.

        Raises
        ------
        RuntimeError
            If a valid obstacle location cannot be found.
        """
        super().reset(seed, options)

        self.drones = [i for i in range(self.num_drones)]

        self.num_obstacles = random.randint(0, self.max_obstacles)

        self.time_step = 0

        ## Randomized vertiports or no? Or randomize vertiports in a certain area of the grid (quad 1, quad 3)
        # Currently set to random places on the graph
        # More sophisticated version, set required distance between two vertiports is at least some distance as a function of grid size
        mid_graph = (int)(self.grid_size / 2)
        self.vertiport1 = (random.randint(0, mid_graph), random.randint(0, mid_graph))
        self.vertiport2 = (
            random.randint(mid_graph, self.grid_size),
            random.randint(mid_graph, self.grid_size),
        )
        self.vertiports_loc = np.array(
            [self.vertiport1, self.vertiport2]
        )  ## Adjust for scaling up to more than two vertiports

        self.obstacles_pos = np.empty(self.num_obstacles)
        for i in range(self.num_obstacles):
            self.obstacles_pos[i] = np.array(
                [random.randint(0, self.grid_size), random.randint(0, self.grid_size)]
            )
            count = 0
            # Keep running until a valid obstacle location found outside the vertiport area (square box around it)
            ## Pretty sure this accounts for edge cases where vertiports are at the corners of the grid
            while True:
                if self._is_overlapping(
                    self.vertiport1, self.S_v, self.obstacles_pos[i], self.S_o
                ):
                    self.obstacles_pos[i] = np.array(
                        [
                            random.randint(0, self.grid_size),
                            random.randint(0, self.grid_size),
                        ]
                    )
                    count += 1
                    if count >= 100:
                        raise RuntimeError(
                            "Failed to find valid obstacle location, likely because the grid size is too small or by chance"
                        )
                else:
                    break
        # Allow for obstacle overlap? If not, additional logic will be needed
        # Will need to consider obstacle radius and eliminate all grid spaces taken up by an obstacle
        # Create a function that determines if two objects are overlapping, takes in two radii, for drones, obstacles, or vertipoints

        self.drone_pos = np.empty(self.num_drones)
        self.drone_vertiport = np.empty(self.num_drones, dtype=int)
        for i in range(self.num_drones):
            num_vertiport: int = random.randint(0, self.num_vertiports - 1)
            loc = self.vertiports_loc[num_vertiport]
            # Changes upon how many vertiports we want, for scaling up leave as numerical
            self.drone_pos[i] = (loc[0], loc[1])
            self.drone_vertiport[i] = num_vertiport
        self.drone_vel = np.empty(self.num_drones)
        for i in range(self.num_drones):
            self.drone_vel[i] = (0.0, 0.0)  # Set initial velocity to 0

        # Implement _get_obs()
        drone_obs = self._get_obs()
        # obs = OrderedDict({f"{a}": drone_obs[a] for a in self.drones}) do I need this line if my key is already defined in the dictionary drone_obs?

        return drone_obs

    def step(self, actions):
        """Take a step in the environment.

        Parameters
        ----------
        actions : Box
            Actions taken by the agents.

        Returns
        -------
        dict
            Dictionary containing the next observations for all drones.
        dict
            Dictionary containing the rewards for all drones.
        dict
            Dictionary containing whether each drone has terminated.
        dict
            Dictionary containing whether each drone has been truncated.
        dict
            Dictionary containing additional information for each drone.

        Raises
        ------
        RuntimeError
            If the environment has not been initialized; call reset() first.
        """
        if (
            self.drones is None
            or self.drone_pos is None
            or self.drone_vel is None
            or self.obstacles_pos is None
            or self.num_drones is None
            or self.time_step is None
            or self.drone_vertiport is None
        ):
            raise RuntimeError("Environment not initialized; call reset() first.")

        super().step(actions)
        current_obs = self._get_obs()
        # take an action
        # update state with kinematics equations for each drone
        for i in range(self.num_drones):
            action = self.action_space(i).sample()
            # i not needed as param bc agent param not used?
            # action is np.array([x accel, y accel])
            vx, vy = self.drone_vel[i]
            ## ^^ use _get_state() to get the current drone vel? or can i just access the global variable
            # np.random.normal(0, 1) adds Gaussian noise to the velocity and position
            vx += action[0] * self.dt + vx + np.random.normal(0, 1)
            vy += action[1] * self.dt + vy + np.random.normal(0, 1)
            self.drone_vel[i] = (vx, vy)

            px, py = self.drone_pos[i]  # same concern as above for vel
            px += (
                0.5 * action[0] * self.dt**2
                + vx * self.dt
                + px
                + np.random.normal(0, 1)
            )
            py += (
                0.5 * action[1] * self.dt**2
                + vy * self.dt
                + py
                + np.random.normal(0, 1)
            )
            self.drone_pos[i] = (px, py)
        # get next_obs
        next_obs = self._get_obs()
        self.time_step += 1  # increment time step

        # compute reward using obs and next_obs
        rewards = {}
        for drone in self.drones:
            reward = self._get_reward(drone, current_obs, next_obs, action)
            rewards[i] = reward
        # check term, trunc, info
        terminated = False # Check if all drones have reached final destination, or if all collided 
        #FIXME
        for drone in self.drones:
            if (
                (self.time_step >= self.max_steps)
                or (
                    self.drone_pos[drone]
                    == self.vertiports_loc[self.drone_vertiport[drone]]
                )
                or (self.collided())
            ):
                terminated = True
        # Default implementation for truncated and info (can implement later)
        truncated = {i: False for i in self.drones} # if self.time_step >= self.max_steps else False, just a boolean
        info = {i: {} for i in self.drones}
        return next_obs, rewards, terminated, truncated, info

    ## How to account for when drones spawn at the vertiports and must be overlapping?
    def collided(self):
        """Check if any drones or obstacles have collided.

        Returns
        -------
        bool
            True if any drones or obstacles have collided, False otherwise.

        Raises
        ------
        RuntimeError
            If the environment has not been initialized; call reset() first.
        """
        if (
            self.drone_pos is None
            or self.num_drones is None
            or self.obstacles_pos is None
        ):
            raise RuntimeError("Environment not initialized; call reset() first.")
        # Is ok for drones to be within safety radius of each other, just negatively rewarded
        for i in range(self.num_drones):
            for j in range(self.num_drones):
                if i != j:
                    if self._is_overlapping(
                        self.drone_pos[i],
                        self.S_d,
                        self.drone_pos[j],
                        self.S_d,
                    ):
                        return True
        for i in range(self.num_drones):
            for j in range(self.num_obstacles):
                if self._is_overlapping(
                    self.drone_pos[i],
                    self.S_d,
                    self.obstacles_pos[j],
                    self.S_o,
                ):
                    return True
        return False

    def render(self): # image numpy array, returns an image of the environment for each time step, 
        # piece together all the images to create a video
        return super().render()

    ## observation_space is a template of observations for each drone, right?
    ## drone parameter not used because each drone has the same set of observations?
    def observation_space(
        self, drone: int
    ):  # agent is int because we define the names of drones by numbers
        """Get the observation space for a single drone.

        Returns
        -------
        Dict
            Dictionary containing the observation space for a single drone.
        """
        assert self.drones is not None, "Environment not initialized; call reset() first."
        max_rel_drone_vel = float(2. * self.max_vel)

        obs_space = Dict({ i: Dict(
            {
                "rel_drone_positions": Box(
                    low=0,
                    high=self.grid_size - 1,
                    shape=(self.num_drones - 1, 2),
                    dtype=np.float32,
                ),
                "rel_drone_velocities": Box(
                    low=-max_rel_drone_vel,
                    high=max_rel_drone_vel,
                    shape=(self.num_drones - 1, 2),
                    dtype=np.float32,
                ),
                "rel_obst_positions": Box(
                    low=0,
                    high=self.grid_size - 1,
                    shape=(self.num_obstacles, 2),
                    dtype=np.float32,
                ),
                "rel_obst_velocities": Box(
                    low=-self.max_vel,
                    high=self.max_vel,
                    shape=(self.num_obstacles, 2),
                    dtype=np.float32,
                ),
                "rel_vert_positions": Box(
                    low=0,
                    high=self.grid_size - 1,
                    shape=(self.num_vertiports, 2),
                    dtype=np.float32,
                ),
                "rel_vert_velocities": Box(
                    low=-self.max_vel,
                    high=self.max_vel,
                    shape=(self.num_vertiports, 2),
                    dtype=np.float32,
                ),
            }
        ) for i in self.drones})
        return obs_space

    def action_space(self, agent: int):
        """Get the action space for a single drone.

        Parameters
        ----------
        agent : int
            _description_

        Returns
        -------
        Box
            the action for a single drone
        """
        action = Box(
            low=-self.max_accel, high=self.max_accel, shape=(2,), dtype=np.float32
        )
        return action
