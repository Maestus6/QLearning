import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import time


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=(12,4)):
        self.xsize, self.ysize = size  # The size of the square grid
        self.window_xsize = 3*256  # The size of the square grid (512 doesnt fits the screen)
        self.window_ysize = 256  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
		 "agent": spaces.Box(low=np.array([0,0]), high=np.array([self.xsize-1, self.ysize-1]), shape=(2,), dtype=int),
           	 "target": spaces.Box(low=np.array([0,0]), high=np.array([self.xsize-1, self.ysize-1]), shape=(2,), dtype=int),
               # "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
               # "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = np.array([0, self.ysize-1])

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = np.array([self.xsize-1, self.ysize-1])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

       # # We need the following line to seed self.np_random
       # super().reset(seed=seed)
       # # Choose the agent's location uniformly at random
 #       # self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
#	#Making it start at top left corner
#	self._agent_location = np.array([0, self.ysize-1]) #Making it start at top left corner
  #      # We will sample the target's location randomly until it does not coincide with the agent's location
   #     self._target_location = self._agent_location
    #    while np.array_equal(self._target_location, self._agent_location):
#		 self._target_location = np.array([self.xsize-1, self.ysize-1]) #target's location fixed to the bottom-right corner of the grid
 #           #self._target_location = self.np_random.integers(
  #          #    0, self.size, size=2, dtype=int
   #        # )

    #    observation = self._get_obs()
     #   info = self._get_info()

      #  if self.render_mode == "human":
       #     self._render_frame()

       # return observation, info
    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            #self._agent_location + direction, 0, self.size - 1
	    self._agent_location + direction, (0, 0), (self.xsize - 1, self.ysize - 1) #changed because of the xsize, ysize
        )
        # An episode is done iff the agent has reached the target
        if ( 1 < self._agent_location[0] < self.xsize-1) and self._agent_location[1] == self.ysize-1:   
             print(self._agent_location)
             time.sleep(0.8)   #to see the fall, not needed
             self._agent_location = np.array([0, self.ysize-1]) #resetting the position of agent after it falls down
             self.cliff=True
        else :
             self.cliff=False
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = -100 if self.cliff else 1 if terminated else -1  # Binary sparse rewards
        observation = self._get_obs()
#        terminated = np.array_equal(self._agent_location, self._target_location)
#	reward = -100 if self.cliff else 1 if terminated else -1 #going down the cliff is not desirable, need -1 to encourage the process to reach to the goal asap
       # reward = 1 if terminated else 0  # Binary sparse rewards 
#        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
           # self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.window = pygame.display.set_mode((self.window_xsize, self.window_ysize)) #Changed due to x and y size changes
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

       # canvas = pygame.Surface((self.window_size, self.window_size))
        canvas = pygame.Surface((self.window_xsize, self.window_ysize))
        canvas.fill((255, 255, 255))
       # pix_square_size = (
        #    self.window_size / self.size
       # )  # The size of a single grid square in pixels
        pix_square_size_x = self.window_xsize / self.xsize
        pix_square_size_y = self.window_ysize / self.ysize

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
               # pix_square_size * self._target_location,
               # (pix_square_size, pix_square_size),
                pix_square_size_x * self._target_location[0], #calculates the X position of the target rectangle by multiplying the target's X coordinate by the width of each grid cell
                pix_square_size_y * self._target_location[1], #calculates the Y position of the target rectangle by multiplying the target's Y coordinate by the height of each grid cell
                pix_square_size_x, #is the width of the rectangle
                pix_square_size_y, #is the height of the rectangle
            ),
        )

        # Now we draw the agent
	# Whole point here is to make agent stay within the cell, centered
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            #(self._agent_location + 0.5) * pix_square_size,
            #pix_square_size / 3,
            (
                (self._agent_location[0] + 0.5) * pix_square_size_x,
                (self._agent_location[1] + 0.5) * pix_square_size_y,
            ),
            min(pix_square_size_x, pix_square_size_y) / 3,
        )

        # Finally, add some gridlines
        for x in range(self.ysize + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size_y * x),
                (self.window_xsize, pix_square_size_y * x),
                width=3,
            )
        for x in range(self.xsize + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size_x * x, 0),
                (pix_square_size_x * x, self.window_ysize),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
