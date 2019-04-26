import os

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

import matplotlib.pyplot as plt
import dill

# Defining actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Define colors
COLORS = {0: [1, 1, 1], 4: [0.6, 0, 1],
          -1: [0.4, 0.3, 1.0], -2: [0.0, 1.0, 0.0],
          -3: [1.0, 0.0, 0.0], 3: [0.0, 1.0, 0.0],
          -13: [0.5, 0.0, 0.5], 1: [0.6, 0.3, 0.0], 10: [0.6, 0, 1]}


# COLORS = {0: [1, 1, 1], 4: [1, 1, 1],
#           -1: [0.4, 0.3, 1.0], -2: [0.0, 1.0, 0.0],
#           -3: [1.0, 0.0, 0.0], 3: [1.0, 1.0, 1.0],
#           -13: [0.5, 0.0, 0.5], 1: [0.6, 0.3, 0.0], 10: [1, 1, 1]}


class FourRooms(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, terminal_reward=10.0, step_reward=-1.0, mapFileName='map1.txt'):
        self.n = None
        self.m = None

        self.done = False

        # Rewards
        self.step_reward = step_reward
        self.terminal_reward = terminal_reward
        self.hallway_reward = 1.0
        self.hit_wall_reward = -1.0

        self.options_policy = None

        # Grid world from file
        self.grid = None
        self.hallwayStates = None
        self.possibleStates = []
        self.walls = []
        self.goals = []  # termination states
        self.mapFile = mapFileName
        self._map_init()

        self.start_state_coord = (1, 1)  # self.possibleStates[int(np.random.choice(len(self.possibleStates) - 1, 1))]
        self.state = self.start_state_coord
        self.observation_space = spaces.Discrete(len(self.possibleStates))
        self.action_space = spaces.Discrete(6)

    # Calling to load Map
    def _map_init(self):
        self.grid = []
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.file_name = os.path.join(this_file_path, self.mapFile)
        with open(self.file_name, "r") as f:

            for i, row in enumerate(f):
                row = row.rstrip('\r\n')
                row = row.split(' ')
                if self.n is not None and len(row) != self.n:
                    raise ValueError(
                        "Map's rows are not of the same dimension...")
                self.n = len(row)
                rowArray = []
                for j, col in enumerate(row):
                    rowArray.append(col)
                    # if col == "4":
                    #     self.start.append([i, j])  # self.n * i + j
                    if col == "3":
                        self.goals.append((i, j))
                    elif col == "1":
                        self.walls.append((i, j))

                    # log possible states
                    if col != "1":
                        # TODO: Check if hallway or not
                        self.possibleStates.append((i, j))
                self.grid.append(rowArray)
            self.m = i + 1

        if len(self.goals) == 0:
            raise ValueError("At least one goal needs to be specified...")

        self._find_hallWays()

    def _find_hallWays(self):
        self.hallwayStates = []
        for x, y in self.possibleStates:
            if ((self.grid[x - 1][y] == "1") and (self.grid[x + 1][y] == "1")) or \
                    ((self.grid[x][y - 1] == "1") and (self.grid[x][y + 1] == "1")):
                self.hallwayStates.append((x, y))

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

    def step(self, action):
        assert self.action_space.contains(action)

        if self.state in self.goals:
            self.done = True
            return self.state, self._get_reward(), self.done, None

        if action < 4:  # if primitive action

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

            if self.grid[x][y] == "1":  # new_state in walls list
                # stay at old state if new coord is wall
                new_state = self.state
            else:
                # if not possible state then move
                self.state = new_state
            reward = self._get_reward(new_state=new_state)
            return self.state, reward, self.done, None

        else:  # if option then calling execute option
            return self._execute_options(action)

    def _execute_options(self, action):
        assert self.action_space.contains(action)

        options_termiantion_indexs = [[(3, 6), (6, 2)],
                                      [(7, 9), (3, 6)],
                                      [(10, 6), (7, 9)],
                                      [(6, 2), (10, 6)]]

        in_hallway = self.state in self.hallwayStates
        room = self._getRoomNumber(self.state)
        start_room = room

        if in_hallway:
            options_termiantion_indexs = [[(6, 2), (10, 6)],
                                          [(7, 9), (3, 6)],
                                          [(6, 2), (10, 6)],
                                          [(3, 6), (7, 9)]]

        option_terminal_state_index = options_termiantion_indexs[start_room][action - 4]
        total_reward = 0
        total_steps = 0
        new_room = start_room
        while (self.state != option_terminal_state_index) and (self.state != self.goals) and (
                in_hallway or start_room == new_room):

            a = self.options_policy[self.state]  # finding action to take according to option policy
            next_state, reward, self.done, _ = self.step(a)  # taking action
            if total_steps > 0:  # updating option total reward
                total_reward += reward * (0.9 ** total_steps)
            else:
                total_reward += reward
            total_steps += 1

            if self.done:
                break

            room = self._getRoomNumber()
            new_room = room

        self.options_length = total_steps
        return self.state, total_reward, self.done, None

    def _get_reward(self, new_state=None):
        if self.done:
            return self.terminal_reward

        if self.state == new_state:
            return self.hit_wall_reward

        reward = self.step_reward

        if new_state in self.hallwayStates:
            reward = self.hallway_reward

        return reward

    def at_border(self):
        [row, col] = self.ind2coord(self.state)
        return (row == 0 or row == self.n - 1 or col == 0 or col == self.n - 1)

    def reset(self):
        self.state = self.start_state_coord
        self.done = False
        return self.state

    def render(self, mode='human', drawArrows=False, policy=None):

        self.observation = self._gridmap_to_observation()
        img = self.observation
        fig = plt.figure(1, figsize=(10, 8), dpi=60,
                         facecolor='w', edgecolor='k')

        plt.clf()
        plt.xticks(np.arange(0, 14, 1))
        plt.yticks(np.arange(0, 14, 1))
        plt.grid(True)
        plt.title("Four room grid world\nAgent:Purple, Goal:Green", fontsize=20)

        plt.imshow(img, origin="upper", extent=[0, 13, 0, 13])
        fig.canvas.draw()

        if drawArrows & (type(policy) is not None):  # For drawing arrows of optimal policy
            fig = plt.gcf()
            ax = fig.gca()
            for state, action in policy.items():
                y, x = state
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
            plt.title(
                "Four room Grid World learned Optimal Policy\nO1: Hallway Option 1 (clockwise) O2: Hallway Option 2",
                fontsize=15)
            # plt.savefig(self.figtitle + "_Arrows.png")

        plt.pause(0.00001)  # 0.01
        return

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

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _gridmap_to_observation(self):
        grid_map = self._read_grid_map()
        obs_shape = [13, 13, 3]
        observation = np.random.randn(*obs_shape) * 0.0
        gs0 = int(observation.shape[0] / grid_map.shape[0])
        gs1 = int(observation.shape[1] / grid_map.shape[1])
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                for k in range(3):
                    if (i, j) == self.state:
                        this_value = COLORS[10][k]
                    else:
                        this_value = COLORS[grid_map[i, j]][k]
                    observation[i * gs0:(i + 1) * gs0, j * gs1:(j + 1)
                                                               * gs1, k] = this_value
        return observation

    def _read_grid_map(self):
        grid_map = open(self.file_name, 'r').readlines()
        grid_map_array = []
        for k1 in grid_map:
            k1s = k1.split(' ')
            tmp_arr = []
            for k2 in k1s:
                try:
                    tmp_arr.append(int(k2))
                except:
                    pass
            grid_map_array.append(tmp_arr)
        grid_map_array = np.array(grid_map_array)
        return grid_map_array
