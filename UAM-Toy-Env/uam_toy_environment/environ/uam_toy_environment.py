from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box, Dict
from collections import OrderedDict
import numpy as np
import random
import gymnasium as gym


class UAMToyEnvironment(gym.Env):
    """
    Environment in which drones begin at one site and must travel to the oppositing site without crashing into each other
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "uam_toy_environment_v0"}

    def __init__(
        self,
        grid_size: int = 100,
        num_drones: int = 1,
        S_v: int = 5,
        S_d: int = 2,
        S_o: int = 1,
        max_accel: float = 5.0,
        max_vel : float = 500.0,
        max_steps: int = 100,
        max_obstacles=10,
        num_obstacles=10,
        num_vertiports=2,
        vertiports_loc=np.array([(1, 2), (2, 4)]),
        safety_radius=5,
        render_mode=None,
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
            vertiports_loc.shape[0] == num_vertiports
        ), "Number of vertiport coordinates do not correspond to number of vertiports"
        self.safety_radius = safety_radius
        self.render_mode = render_mode

        self.drones = None # can initialize in innit
        self.destinations_loc = None  # np.array of tuples (x, y), randomized for more than 2 vertiports, implement after testing 2 vertiports
        self.time_step = None
        self.vertiport1 = vertiports_loc[0]
        self.vertiport2 = vertiports_loc[1]
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
        dict
            Dictionary containing the state information with keys: 'drone_pos', 'drone_vel', 
            'obstacles_pos', and 'vertiports_loc'.

        Raises
        ------
        RuntimeError
            If the environment has not been initialized; call reset() first.
        """
        if self.drone_pos is None or self.drone_vel is None or self.obstacles_pos is None:
            raise RuntimeError("Environment not initialized; call reset() first.")
        
        state = {
            "drone_pos": self.drone_pos,
            "drone_vel": self.drone_vel,
            "obstacles_pos": self.obstacles_pos,
            "vertiports_loc": self.vertiports_loc,
        }
        return state

    def _get_obs(self):
        """Get the current observations.
        
        For multi-agent, returns a dict mapping agent keys to observations.
        For single-agent, returns just the observation dict.
        """
        if self.drone_pos is None or self.drone_vel is None or self.obstacles_pos is None:
            raise RuntimeError("Environment not initialized; call reset() first.")

        # Build the inner observation dictionary for one agent.
        def build_obs_for_agent(i):
            if self.drone_pos is None or self.drone_vel is None or self.obstacles_pos is None:
                raise RuntimeError("Environment not initialized; call reset() first.")
            # Ensure the drone's position and velocity are float32
            drone_pos = np.array(self.drone_pos[i], dtype=np.float32)
            drone_vel = np.array(self.drone_vel[i], dtype=np.float32)

            # Compute relative positions and velocities for all other drones
            rel_drone_positions = []
            rel_drone_velocities = []
            for j in range(self.num_drones):
                if j != i:
                    pos_diff = np.array(self.drone_pos[j], dtype=np.float32) - drone_pos
                    vel_diff = np.array(self.drone_vel[j], dtype=np.float32) - drone_vel
                    rel_drone_positions.append(pos_diff)
                    rel_drone_velocities.append(vel_diff)
            rel_drone_positions = np.array(rel_drone_positions, dtype=np.float32)
            rel_drone_velocities = np.array(rel_drone_velocities, dtype=np.float32)

            # Compute relative positions and velocities for obstacles.
            rel_obst_positions = []
            rel_obst_velocities = []
            for obst in self.obstacles_pos:
                pos_diff = np.array(obst, dtype=np.float32) - drone_pos
                rel_obst_positions.append(pos_diff)
                rel_obst_velocities.append(-drone_vel)
            rel_obst_positions = np.array(rel_obst_positions, dtype=np.float32)
            rel_obst_velocities = np.array(rel_obst_velocities, dtype=np.float32)

            # Compute relative positions and velocities for vertiports.
            rel_vert_positions = []
            rel_vert_velocities = []
            for vp in self.vertiports_loc:
                pos_diff = np.array(vp, dtype=np.float32) - drone_pos
                rel_vert_positions.append(pos_diff)
                rel_vert_velocities.append(-drone_vel)
            rel_vert_positions = np.array(rel_vert_positions, dtype=np.float32)
            rel_vert_velocities = np.array(rel_vert_velocities, dtype=np.float32)

            return {
                "rel_drone_positions": rel_drone_positions,
                "rel_drone_velocities": rel_drone_velocities,
                "rel_obst_positions": rel_obst_positions,
                "rel_obst_velocities": rel_obst_velocities,
                "rel_vert_positions": rel_vert_positions,
                "rel_vert_velocities": rel_vert_velocities,
            }

        # If only one drone exists, return its observation directly.
        if self.num_drones == 1:
            return build_obs_for_agent(0)
        else:
            # For multi-agent, return a dict mapping agent keys (as strings) to the corresponding observation.
            obs = {}
            for i in range(self.num_drones):
                obs[str(i)] = build_obs_for_agent(i)
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
            or self.drones is None
            or self.time_step is None
        ):
            raise RuntimeError("Environment not initialized; call reset() first.")
        ## vv WHY IS IT AN ARRAY OF TUPLE?? does this actually just work??
        drone_starting_vertiport: int = self.drone_vertiport[agent]
        ## For more than two vertiports, which vertiport does each drone go to??

        # Do +1 to get the correct vertiport index, since the vertiport index is 0-indexed
        obs_initial_single_agent = obs_initial
        obs_next_single_agent = obs_next
        # obs_initial_multi_agent = obs_initial[str(agent)] # use if more than one drone
        # obs_next_multi_agent = obs_next[str(agent)] # use if more than one drone

        initial_pos = obs_initial_single_agent["rel_vert_positions"][ # substitute single_agent for multi_agent if more than one drone
            self.num_vertiports - (drone_starting_vertiport + 1)
        ]
        next_pos = obs_next_single_agent["rel_vert_positions"][ # substitute single_agent for multi_agent if more than one drone
            self.num_vertiports - (drone_starting_vertiport + 1)
        ]
        # if moving closer to vertiport, positive reward, negative if moving farther away
        reward_goal = 1. * (-np.linalg.norm(next_pos) + np.linalg.norm(initial_pos))**3

        agent_collision = 0.0
        obstacle_collision = 0.0
        out_of_bounds = 0.0
        vertiport_reached = 0.0
        time = 0.0
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
                obstacle_collision += -100.0
        for drone in self.drones:
            any_out_of_bounds = (
                self.drone_pos[drone][0] < 0
                or self.drone_pos[drone][0] > self.grid_size
                or self.drone_pos[drone][1] < 0
                or self.drone_pos[drone][1] > self.grid_size)
            if any_out_of_bounds:
                out_of_bounds += -100.0
        for drone in self.drones:
            if self._is_overlapping(
                self.drone_pos[drone],
                self.S_d,
                self.vertiports_loc[1 if self.drone_vertiport[drone] == 0 else 0],
                0,
            ):
            # if all(self.drone_pos[drone] == self.vertiports_loc[1 if self.drone_vertiport[drone] == 0 else 0]):
                vertiport_reached += 1.0e3
                time += (self.max_steps - self.time_step) * 1.0e4
                print("Reached vertiport!")
        if self.time_step >= self.max_steps:
            time += -1000.0
        reward = reward_goal + agent_collision + obstacle_collision + out_of_bounds + vertiport_reached + time

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
        super().reset(seed=seed)

        self.drones = [i for i in range(self.num_drones)]

        self.time_step = 0

        ## Randomized vertiports or no? Or randomize vertiports in a certain area of the grid (quad 1, quad 3)
        # Currently set to random places on the graph
        # More sophisticated version, set required distance between two vertiports is at least some distance as a function of grid size
        mid_graph = (int)(self.grid_size / 2)
        self.vertiport1 = (random.randint(0 + self.S_v, mid_graph), random.randint(0 + self.S_v, mid_graph))
        self.vertiport2 = (
            random.randint(mid_graph, self.grid_size - self.S_v),
            random.randint(mid_graph, self.grid_size - self.S_v),
        )
        self.vertiports_loc = np.array(
            [self.vertiport1, self.vertiport2]
        )  ## Adjust for scaling up to more than two vertiports

        self.obstacles_pos = np.empty(self.num_obstacles, dtype=object)
        for i in range(self.num_obstacles):
            self.obstacles_pos[i] = np.array(
                (random.randint(0, self.grid_size), random.randint(0, self.grid_size))
            )
            count = 0
            # Keep running until a valid obstacle location found outside the vertiport area (square box around it)
            ## Pretty sure this accounts for edge cases where vertiports are at the corners of the grid
            while True:
                if self._is_overlapping(
                    self.vertiport1, self.S_v, self.obstacles_pos[i], self.S_o
                ):
                    self.obstacles_pos[i] = np.array(
                            (random.randint(0, self.grid_size),
                            random.randint(0, self.grid_size))
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

        self.drone_pos = np.empty(self.num_drones, dtype=object)
        self.drone_vertiport = np.empty(self.num_drones, dtype=int)
        for i in range(self.num_drones):
            num_vertiport: int = random.randint(0, self.num_vertiports - 1)
            loc = self.vertiports_loc[num_vertiport]
            # Changes upon how many vertiports we want, for scaling up leave as numerical
            self.drone_pos[i] = (loc[0], loc[1])
            self.drone_vertiport[i] = num_vertiport
        self.drone_vel = np.empty(self.num_drones, dtype=object)
        for i in range(self.num_drones):
            self.drone_vel[i] = (0.0, 0.0)  # Set initial velocity to 0

        # Implement _get_obs()
        drone_obs = self._get_obs()
        # obs = OrderedDict({f"{a}": drone_obs[a] for a in self.drones}) do I need this line if my key is already defined in the dictionary drone_obs?
        info = {}
        return drone_obs, info

    def step(self, actions):
        """Take a step in the environment.

        Parameters
        ----------
        actions : Dict
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

        current_obs = self._get_obs()
        # take an action
        # update state with kinematics equations for each drone
        for i in range(self.num_drones):

            # i not needed as param bc agent param not used?
            # action is np.array([x accel, y accel])
            vx, vy = self.drone_vel[i]
            ## ^^ use _get_state() to get the current drone vel? or can i just access the global variable
            # np.random.normal(0, 1) adds Gaussian noise to the velocity and position
            action_single_agent = actions
            # action_multi_agent = actions[i] # use if more than one drone
            vx += (action_single_agent[0] * self.dt
                #    + np.random.normal(0, 1) # no noise for initial training (single agent)
                   )
            vy += (action_single_agent[1] * self.dt
                #    + np.random.normal(0, 1) # no noise for initial training (single agent)
                   )
            self.drone_vel[i] = (vx, vy)

            px, py = self.drone_pos[i]  # same concern as above for vel
            if self._is_overlapping(
                self.drone_pos[i],
                self.S_d,
                self.obstacles_pos[i],
                self.S_o,
            ):
                # stop the drone when it hits an obstacle
                self.drone_pos[i] = (px, py)
                self.drone_vel[i] = (0.0, 0.0)
                continue  # skip to the next drone

            px += (
                0.5 * action_single_agent[0] * self.dt**2
                + vx * self.dt
                # + np.random.normal(0, 1) # no noise for initial training (single agent)
            )
            py += (
                0.5 * action_single_agent[1] * self.dt**2
                + vy * self.dt
                # + np.random.normal(0, 1) # no noise for initial training (single agent)
            )
            self.drone_pos[i] = (px, py)
            
        # get next_obs
        next_obs = self._get_obs()
        self.time_step += 1  # increment time step

        # compute reward using obs and next_obs
        rewards = {}
        for i in range(self.num_drones):
            reward = self._get_reward(self.drones[i], current_obs, next_obs, actions[i])
            rewards[i] = reward
        # check term, trunc, info
        ## Check if all drones have reached final destination, or if all collided 

        all_collided = all(self.collided(drone) for drone in self.drones)
        # all_reached = all(
        #     self._is_overlapping(
        #         self.drone_pos[drone],
        #         self.S_d,
        #         self.vertiports_loc[1 if self.drone_vertiport[drone] == 0 else 0],
        #         self.S_v,
        #     )
        #     for drone in self.drones
        # )
        all_reached = False
        if self._is_overlapping(
                self.drone_pos[0],
                self.S_d,
                self.vertiports_loc[1 if self.drone_vertiport[0] == 0 else 0],
                0,
            ):
            all_reached = True
        any_out_of_bounds = any(
            self.drone_pos[i][0] < 0
            or self.drone_pos[i][0] > self.grid_size
            or self.drone_pos[i][1] < 0
            or self.drone_pos[i][1] > self.grid_size
            for i in range(self.num_drones)
        )
        terminated = (all_collided or all_reached or any_out_of_bounds) 

        truncated = (self.time_step >= self.max_steps)
        info = {i: {} for i in self.drones}

        rewards_single_agent = rewards[0]
        return next_obs, rewards_single_agent, terminated, truncated, info

    ## How to account for when drones spawn at the vertiports and must be overlapping?
    def collided(self, drone: int) -> bool:
        """Check if a drone has collided with another drone or an obstacle.

        Args:
            drone (int): The index of the drone to check for collisions.

        Returns:
            bool: True if the drone has collided with another drone or an obstacle, False otherwise.
        """
        # Check against other drones.
        if self.drones is None or self.drone_pos is None or self.obstacles_pos is None or self.vertiport1 is None or self.vertiport2 is None:
            raise RuntimeError("Environment not initialized; call reset() first.")
        for other in self.drones:
            if drone != other:
                if (self._is_overlapping(self.drone_pos[drone], self.S_d, self.drone_pos[other], self.S_d)
                    and not self._is_overlapping(self.drone_pos[drone], self.S_d, self.vertiport1, self.S_v)
                    and not self._is_overlapping(self.drone_pos[drone], self.S_d, self.vertiport2, self.S_v)):
                    return True
        # Check against obstacles.
        for obst in self.obstacles_pos:
            if self._is_overlapping(self.drone_pos[drone], self.S_d, obst, self.S_o):
                return True
        return False

    def render(self, mode="human"):
        """
        Render the current state of the environment as an RGB image.
        
        Parameters:
            mode (str): If "rgb_array" (default), returns an image as a NumPy array.
                        If "human", you might choose to display the image using an image viewer.
                        
        Returns:
            np.ndarray: An RGB image representing the environment at the current time step.
        """
        assert self.drone_vel is not None, "Environment not initialized; call reset() first."
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        
        # Create a figure and axis for drawing the environment.
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect("equal")
        ax.set_title(f"UAMToyEnvironment - Time Step: {self.time_step}")

        # Plot vertiport positions first with a lower zorder.
        if self.vertiports_loc is not None:
            for i, vp in enumerate(self.vertiports_loc):
                color = "green" if i == 0 else "purple"
                ax.scatter(vp[0], vp[1], c=color, marker="s", s=150, label=f"Vertiport {i+1}", zorder=1)

        # Plot obstacle positions.
        if self.obstacles_pos is not None and self.num_obstacles > 0:
            obstacles_positions = np.array([np.array(o) for o in self.obstacles_pos])
            if obstacles_positions.size > 0:
                ax.scatter(obstacles_positions[:, 0], obstacles_positions[:, 1],
                        c="red", marker="x", s=100, label="Obstacles")

        # Plot drone positions with a higher zorder so that they appear above the vertiports.
        if self.drone_pos is not None and len(self.drone_pos) > 0:
            drone_positions = np.array([np.array(p) for p in self.drone_pos])
            ax.scatter(drone_positions[:, 0], drone_positions[:, 1],
                    c="blue", marker="o", label="Drones", zorder=3)
            # Plot velocity arrows for each drone.
            ## if want to include arrows for each drone, uncomment the following lines
            # for i, pos in enumerate(drone_positions):
            #     vx, vy = self.drone_vel[i]
            #     ax.arrow(pos[0], pos[1], vx * 0.01, vy * 0.01,
            #             head_width=2, head_length=2, fc="blue", ec="blue", zorder=4)

        # Move legend off to the side (outside the plot) 
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

        # Draw the canvas and convert it to an RGB NumPy array.
        fig.canvas = FigureCanvas(fig)
        fig.canvas.draw()
        img_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_rgba = img_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        rgb_img = img_rgba[:, :, :3]

        plt.close(fig)

        # If mode is 'human', you can display the image using an image viewer.
        if mode == self.render_mode:
            import cv2
            cv2.imshow("UAMToyEnvironment", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        return rgb_img

    ## observation_space is a template of observations for each drone, right?
    ## drone parameter not used because each drone has the same set of observations?
    @property
    def observation_space(self):
        max_rel_drone_vel = float(2. * self.max_vel)
        inner_space = Dict({
            "rel_drone_positions": Box(
                low=-self.grid_size,
                high=self.grid_size,
                shape=(self.num_drones - 1, 2),  # For one drone, this would be shape (0, 2)
                dtype=np.float32,
            ),
            "rel_drone_velocities": Box(
                low=-max_rel_drone_vel,
                high=max_rel_drone_vel,
                shape=(self.num_drones - 1, 2),
                dtype=np.float32,
            ),
            "rel_obst_positions": Box(
                low=-self.grid_size,
                high=self.grid_size,
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
                low=-self.grid_size,
                high=self.grid_size,
                shape=(self.num_vertiports, 2),
                dtype=np.float32,
            ),
            "rel_vert_velocities": Box(
                low=-self.max_vel,
                high=self.max_vel,
                shape=(self.num_vertiports, 2),
                dtype=np.float32,
            ),
        })
        
        if self.num_drones == 1:
            # For a single agent environment, just return the inner space.
            return inner_space
        else:
            # For multiple agents, return a dictionary mapping agent keys to the inner space.
            return Dict({str(i): inner_space for i in range(self.num_drones)})
    
    @property
    def action_space(self):
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
