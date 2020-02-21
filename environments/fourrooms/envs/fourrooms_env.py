from collections import defaultdict

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
NUMBER_OF_ACTIONS = 4

# Define colors
# COLOURS = {0: [1, 1, 1], 1: [0.0, 0.0, 0.0], 3: [0.0, 0.0, 0.0], 10: [0.0, 0, 0]}
COLOURS = {0: [1, 1, 1], 1: [0.6, 0.3, 0.0],
           3: [0.0, 1.0, 0.0], 10: [0.6, 0, 1]}
# Define Map
map1 = "1 1 1 1 1 1 1 1 1 1 1 1 1\n" \
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

map2 = "1 1 1 1 1 1 1 1 1 1 1 1 1\n" \
       "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
       "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
       "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
       "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
       "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
       "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
       "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
       "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
       "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
       "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
       "1 0 0 0 0 0 0 0 0 0 0 0 1\n" \
       "1 1 1 1 1 1 1 1 1 1 1 1 1"

MAP = map1


class FourRoomsEnv(gym.Env):
    """
    Four rooms environment from Options Paper (TODO: ADD REF).
    grid world env which is divided into 4 rooms and the goal is to reach the terminal state.
    Available actions are each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge/wall leave you in your current state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, terminal_states: list = None, terminal_reward: float = 0.0,
                 step_reward: float = -1.0):

        self.__grid = None
        self.__hallwayStates = None
        self.__possibleStates = []
        self.__possibleStatesToIndex = defaultdict(lambda: -1)
        self.__walls = []
        self.__number_of_rows = 0
        self.__number_of_columns = 0

        self.__map_init()

        self.observation_space = spaces.Discrete(len(self.__possibleStates))
        for t in terminal_states:
            assert 0 <= t < self.observation_space.n

        self.action_space = spaces.Discrete(NUMBER_OF_ACTIONS)
        self.__terminal_states = terminal_states
        self.__terminal_reward = terminal_reward
        self.__step_reward = step_reward

        self.__initial_state_distribution = np.ones(
            self.observation_space.n) / self.observation_space.n

        self.P = {}
        self.__probs_init()

        # current state is int - use as index in possible states array
        self.__current_state = self.__get_random_state()

    def get_coordinates(self, state_index):
        return self.__possibleStates[state_index]

    def get_params(self):
        return {'terminal_states': self.__terminal_states,
                'terminal_reward': self.__terminal_reward,
                'step_reward': self.__step_reward}

    def __get_random_state(self):
        state = np.random.choice(
            self.observation_space.n, p=self.__initial_state_distribution)
        while state in self.__terminal_states:
            state = np.random.choice(
                self.observation_space.n, p=self.__initial_state_distribution)
        return state

    def __is_done(self, state: int) -> bool:
        return state in self.__terminal_states

    def __probs_init(self):
        for s in range(self.observation_space.n):
            self.P[s] = {a: [] for a in range(self.action_space.n)}
            if self.__is_done(s):
                self.P[s][UP] = [(1.0, s, self.__terminal_reward, True)]
                self.P[s][RIGHT] = [(1.0, s, self.__terminal_reward, True)]
                self.P[s][DOWN] = [(1.0, s, self.__terminal_reward, True)]
                self.P[s][LEFT] = [(1.0, s, self.__terminal_reward, True)]
            else:
                ns_up = self.__get_next_state(s, UP)
                ns_right = self.__get_next_state(s, RIGHT)
                ns_down = self.__get_next_state(s, DOWN)
                ns_left = self.__get_next_state(s, LEFT)
                self.P[s][UP] = [
                    (1.0, ns_up, self.__step_reward, self.__is_done(ns_up))]
                self.P[s][RIGHT] = [
                    (1.0, ns_right, self.__step_reward, self.__is_done(ns_right))]
                self.P[s][DOWN] = [
                    (1.0, ns_down, self.__step_reward, self.__is_done(ns_down))]
                self.P[s][LEFT] = [
                    (1.0, ns_left, self.__step_reward, self.__is_done(ns_left))]

    def __get_next_state(self, state: int, action: int):
        x, y = self.__possibleStates[state]
        if action == UP:
            x = x - 1
        elif action == DOWN:
            x = x + 1
        elif action == RIGHT:
            y = y + 1
        elif action == LEFT:
            y = y - 1
        new_state_coord = (x, y)
        next_state = self.__possibleStatesToIndex[str(new_state_coord)]
        if next_state != -1:
            return next_state
        else:
            return state

    def __map_init(self):

        self.__grid = []
        lines = MAP.split('\n')
        index = 0
        n = None
        m = None
        for i, row in enumerate(lines):
            row = row.split(' ')
            if n is not None and len(row) != n:
                raise ValueError(
                    "Map's rows are not of the same dimension...")
            n = len(row)
            rowArray = []
            for j, col in enumerate(row):
                rowArray.append(int(col))
                if col == "1":
                    self.__walls.append((i, j))
                # possible states
                else:
                    self.__possibleStates.append((i, j))
                    self.__possibleStatesToIndex[str((i, j))] = index
                    index = index + 1
            self.__grid.append(rowArray)
        m = i + 1
        self.__number_of_rows = n
        self.__number_of_columns = m

        self.__find_hallWays()

    def __find_hallWays(self):
        self.__hallwayStates = []
        for x, y in self.__possibleStates:
            if ((self.__grid[x - 1][y] == 1) and (self.__grid[x + 1][y] == 1)) or \
                    ((self.__grid[x][y - 1] == 1) and (self.__grid[x][y + 1] == 1)):
                self.__hallwayStates.append((x, y))

    def __get_reward(self, state):
        if self.__is_done(self.__current_state):
            return self.__terminal_reward
        else:
            return self.__step_reward

    def step(self, action):
        assert self.action_space.contains(action)

        if self.__is_done(self.__current_state):
            return self.__current_state, self.__get_reward(self.__current_state), True, None
        else:
            next_state = self.__get_next_state(self.__current_state, action)
            self.__current_state = next_state

            return self.__current_state, self.__get_reward(self.__current_state), self.__is_done(
                self.__current_state), None

    def reset(self, state=None):
        if state is None:
            self.__current_state = self.__get_random_state()
        else:
            self.__current_state = state
        return self.__current_state

    def seed(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

    def close(self):
        pass

    def render(self, mode='human', draw_arrows=False, draw_values=False, draw_rewards=False, V=None, policy=None,
               R=None, title=None, grid=False, cmap='RdYlGn', draw_numbers=False):
        size_n = self.__number_of_rows
        size_m = self.__number_of_columns
        extent = [0, size_n, 0, size_m]
        img = self.__gridmap_to_img()

        if mode == 'human':
            fig = plt.figure(1, figsize=(10, 8), dpi=60,
                             facecolor='w', edgecolor='k')

            plt.clf()
            plt.xticks(np.arange(0, size_n, 1))
            plt.yticks(np.arange(0, size_m, 1))

            if title:
                plt.title(title)

            plt.imshow(img, origin="upper", extent=extent)
            plt.grid(grid, color='k', linewidth=2)
            fig.canvas.draw()

        if draw_rewards & (type(R) is not None):  # For showing optimal values
            r = np.zeros((size_n, size_m)) + float("-inf")
            for state in range(self.observation_space.n):
                state_coord = self.__possibleStates[state]
                y, x = state_coord
                r[y, x] = R[state]
            c = plt.imshow(r, origin="upper", cmap=cmap, extent=extent)
            fig.colorbar(c, ax=fig.gca())

        if draw_values & (type(V) is not None):  # For showing optimal values
            v = np.zeros((size_n, size_m)) + float("-inf")
            for state in range(self.observation_space.n):
                state_coord = self.__possibleStates[state]
                y, x = state_coord
                v[y, x] = V[state]
            c = plt.imshow(v, origin="upper", cmap=cmap, extent=extent)
            fig.colorbar(c, ax=fig.gca())

        # For drawing arrows of optimal policy
        if draw_arrows & (type(policy) is not None):
            fig = plt.gcf()
            ax = fig.gca()
            for state in range(self.observation_space.n):
                state_coord = self.__possibleStates[state]
                y, x = state_coord
                y = len(self.__grid) - 1 - y
                action = np.argmax(policy[state])
                if all(policy[state][0] == policy[state]):
                    continue
                else:
                    if action > 3:
                        ax.text(x + 0.3, y + 0.3, 'O' +
                                str(action - 3), fontweight='bold')
                    elif action == -1:
                        ax.text(x + 0.3, y + 0.3, 'NE', fontweight='bold')
                    else:
                        self.__draw_arrows(x, y, action)
        if draw_numbers:
            fig = plt.gcf()
            ax = fig.gca()
            for state in range(self.observation_space.n):
                state_coord = self.__possibleStates[state]
                y, x = state_coord
                y = len(self.__grid) - 1 - y
                ax.text(x + 0.3, y + 0.3, str(state), fontweight='bold')

        # plt.pause(0.00001)  # 0.01
        if mode == 'human':
            plt.show()
            return
        elif mode == 'rgb_array':
            return img
        return

    def __draw_arrows(self, x, y, direction):
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

    def __gridmap_to_img(self):
        row_size = len(self.__grid)
        col_size = len(self.__grid[0])

        obs_shape = [row_size, col_size, 3]

        img = np.zeros(obs_shape)

        gs0 = int(img.shape[0] / row_size)
        gs1 = int(img.shape[1] / col_size)
        for i in range(row_size):
            for j in range(col_size):
                for k in range(3):
                    state_coord_string = str((i, j))
                    if self.__possibleStatesToIndex[state_coord_string] == self.__current_state:
                        this_value = COLOURS[10][k]
                    elif self.__possibleStatesToIndex[state_coord_string] in self.__terminal_states:
                        this_value = COLOURS[3][k]
                    else:
                        colour_number = int(self.__grid[i][j])
                        this_value = COLOURS[colour_number][k]
                    img[i * gs0:(i + 1) * gs0, j * gs1:(j + 1)
                                                       * gs1, k] = this_value
        return img
