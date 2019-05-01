from ast import literal_eval
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

import matplotlib.pyplot as plt

# Defining actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Define colors
COLOURS = {0: [1, 1, 1], 1: [0.6, 0.3, 0.0], 3: [0.0, 1.0, 0.0], 10: [0.6, 0, 1]}

MAP = "1 1 1 1 1 1 1 1 1 1 1 1 1\n" \
      "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
      "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
      "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
      "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
      "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
      "1 1 0 1 1 1 1 0 0 0 0 0 1\n" \
      "1 0 0 0 0 0 1 1 1 0 1 1 1\n" \
      "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
      "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
      "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
      "1 0 0 0 0 0 1 0 0 0 0 0 1\n" \
      "1 1 1 1 1 1 1 1 1 1 1 1 1"


class FourRooms(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, goal_reward=10.0, step_reward=-1.0, goal='G1'):
        self.n = None
        self.m = None

        self.grid = None
        self.hallwayStates = None
        self.possibleStates = []
        self.walls = []

        self._map_init()

        self.start_state_coord = (1, 1)
        self.state = self.start_state_coord

        self.done = False

        self.goal = (7, 9)  # default goal

        if goal.upper() == "G1":
            self.goal = (7, 9)
        elif goal.upper() == "G2":
            self.goal = (8, 9)

        # Rewards
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.hallway_reward = step_reward
        self.hit_wall_reward = -10.0

        # Gym spaces for observation and action space
        self.observation_space = spaces.Discrete(len(self.possibleStates))
        self.action_space = spaces.Discrete(4)

    def step(self, action):
        assert self.action_space.contains(action)

        if self.state == self.goal:
            self.done = True
            return [self._getRoomNumber(), self.state], self._get_reward(self.state), self.done, None

        x, y = self.state
        if action == UP:
            x = x - 1
        elif action == DOWN:
            x = x + 1
        elif action == RIGHT:
            y = y + 1
        elif action == LEFT:
            y = y - 1
        new_state = (x, y)

        reward = self._get_reward(new_state)
        if self._get_grid_value(new_state) == 1:  # new_state in walls list
            # stay at old state if new coord is wall
            new_state = self.state
        else:
            self.state = new_state

        return [self._getRoomNumber(), self.state], reward, self.done, None

    def reset(self):
        self.done = False
        self.state = self.start_state_coord
        return [self._getRoomNumber(), self.state]

    def render(self, mode='human', drawArrows=False, policy=None, name_prefix='FourRooms-v1 (G1)'):

        img = self._gridmap_to_img()
        fig = plt.figure(1, figsize=(10, 8), dpi=60,
                         facecolor='w', edgecolor='k')

        plt.clf()
        plt.xticks(np.arange(0, 14, 1))
        plt.yticks(np.arange(0, 14, 1))
        plt.grid(True)
        plt.title(name_prefix + "\nAgent:Purple, Goal:Green", fontsize=20)

        plt.imshow(img, origin="upper", extent=[0, 13, 0, 13])
        fig.canvas.draw()

        if drawArrows & (type(policy) is not None):  # For drawing arrows of optimal policy
            fig = plt.gcf()
            ax = fig.gca()
            for state, action in policy.items():
                y, x = literal_eval(state)[1]
                y = 12 - y
                if action > 3:
                    #  style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
                    if action == 4:
                        ax.text(x + 0.3, y + 0.3, 'O1', fontweight='bold')
                    else:
                        ax.text(x + 0.3, y + 0.3, 'O2', fontweight='bold')
                elif action == -1:
                    ax.text(x + 0.3, y + 0.3, 'NE', fontweight='bold')
                else:
                    self._draw_arrows(x, y, action)
        plt.title(name_prefix + " learned Policy"
                  + "\nO1: Option 1 (clockwise) O2: Option 2",
                  fontsize=15)

        plt.pause(0.00001)  # 0.01
        return

    def _map_init(self):
        self.grid = []
        lines = MAP.split('\n')

        for i, row in enumerate(lines):
            row = row.split(' ')
            if self.n is not None and len(row) != self.n:
                raise ValueError(
                    "Map's rows are not of the same dimension...")
            self.n = len(row)
            rowArray = []
            for j, col in enumerate(row):
                rowArray.append(int(col))
                if col == "1":
                    self.walls.append((i, j))
                # possible states
                else:
                    self.possibleStates.append((i, j))
            self.grid.append(rowArray)
        self.m = i + 1

        self._find_hallWays()

    def _find_hallWays(self):
        self.hallwayStates = []
        for x, y in self.possibleStates:
            if ((self.grid[x - 1][y] == 1) and (self.grid[x + 1][y] == 1)) or \
                    ((self.grid[x][y - 1] == 1) and (self.grid[x][y + 1] == 1)):
                self.hallwayStates.append((x, y))

    def _get_grid_value(self, state):
        return self.grid[state[0]][state[1]]

    # specific for MAP
    def _getRoomNumber(self, state=None):
        if state == None:
            state = self.state
        # if state isn't at hall way point
        xCount = self._greaterThanCounter(state, 0)
        yCount = self._greaterThanCounter(state, 1)
        room = 0
        if yCount >= 2:
            if xCount >= 2:
                room = 2
            else:
                room = 1
        else:
            if xCount >= 2:
                room = 3
            else:
                room = 0

        return room

    def _greaterThanCounter(self, state, index):
        count = 0
        for h in self.hallwayStates:
            if state[index] > h[index]:
                count = count + 1
        return count

    def _get_reward(self, state):
        if state == self.goal:
            return self.goal_reward
        elif self._get_grid_value(state) == 1:
            return self.hit_wall_reward
        elif state in self.hallwayStates:
            return self.hallway_reward
        else:
            return self.step_reward

    def _draw_arrows(self, x, y, direction):
        if direction == UP:
            x += 0.5
            dx = 0
            dy = 0.4
        if direction == DOWN:
            x += 0.5
            y += 1
            dx = 0
            dy = -0.4
        if direction == RIGHT:
            y += 0.5
            dx = 0.4
            dy = 0
        if direction == LEFT:
            x += 1
            y += 0.5
            dx = -0.4
            dy = 0

        plt.arrow(x,  # x1
                  y,  # y1
                  dx,  # x2 - x1
                  dy,  # y2 - y1
                  facecolor='k',
                  edgecolor='k',
                  width=0.005,
                  head_width=0.4,
                  )

    def _gridmap_to_img(self):
        row_size = len(self.grid)
        col_size = len(self.grid[0])

        obs_shape = [row_size, col_size, 3]

        img = np.random.randn(*obs_shape) * 0.0

        gs0 = int(img.shape[0] / row_size)
        gs1 = int(img.shape[1] / col_size)
        for i in range(row_size):
            for j in range(col_size):
                for k in range(3):
                    if (i, j) == self.state:
                        this_value = COLOURS[10][k]
                    elif (i, j) == self.goal:
                        this_value = COLOURS[3][k]
                    else:

                        colour_number = int(self.grid[i][j])
                        this_value = COLOURS[colour_number][k]
                    img[i * gs0:(i + 1) * gs0, j * gs1:(j + 1)
                                                       * gs1, k] = this_value
        return img
