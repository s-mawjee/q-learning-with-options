from utils.options import load_option, save_option
import gym
from option import QOption

import envs.gridworld

# Defining actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


def create_options(env):
    policy_option_1 = {

        # ROOM 0
        '[0, (1, 1)]': RIGHT, '[0, (1, 2)]': RIGHT, '[0, (1, 3)]': RIGHT, '[0, (1, 4)]': RIGHT, '[0, (1, 5)]': DOWN,
        '[0, (2, 1)]': DOWN, '[0, (2, 2)]': LEFT, '[0, (2, 3)]': LEFT, '[0, (2, 4)]': LEFT, '[0, (2, 5)]': LEFT,
        '[0, (3, 1)]': RIGHT, '[0, (3, 2)]': RIGHT, '[0, (3, 3)]': RIGHT, '[0, (3, 4)]': RIGHT, '[0, (3, 5)]': RIGHT,
        '[0, (4, 1)]': UP, '[0, (4, 2)]': LEFT, '[0, (4, 3)]': LEFT, '[0, (4, 4)]': LEFT, '[0, (4, 5)]': LEFT,
        '[0, (5, 1)]': RIGHT, '[0, (5, 2)]': RIGHT, '[0, (5, 3)]': RIGHT, '[0, (5, 4)]': RIGHT, '[0, (5, 5)]': UP,

        # ROOM 1
        '[1, (1, 7)]': DOWN, '[1, (1, 8)]': DOWN, '[1, (1, 9)]': DOWN, '[1, (1, 10)]': DOWN, '[1, (1, 11)]': DOWN,
        '[1, (2, 7)]': DOWN, '[1, (2, 8)]': DOWN, '[1, (2, 9)]': DOWN, '[1, (2, 10)]': DOWN, '[1, (2, 11)]': DOWN,
        '[1, (3, 7)]': DOWN, '[1, (3, 8)]': DOWN, '[1, (3, 9)]': DOWN, '[1, (3, 10)]': DOWN, '[1, (3, 11)]': DOWN,
        '[1, (4, 7)]': DOWN, '[1, (4, 8)]': DOWN, '[1, (4, 9)]': DOWN, '[1, (4, 10)]': DOWN, '[1, (4, 11)]': DOWN,
        '[1, (5, 7)]': DOWN, '[1, (5, 8)]': DOWN, '[1, (5, 9)]': DOWN, '[1, (5, 10)]': DOWN, '[1, (5, 11)]': DOWN,
        '[1, (6, 7)]': RIGHT, '[1, (6, 8)]': RIGHT, '[1, (6, 9)]': DOWN, '[1, (6, 10)]': LEFT, '[1, (6, 11)]': LEFT,

        # ROOM 2
        '[2, (8, 7)]': DOWN, '[2, (8, 8)]': DOWN, '[2, (8, 9)]': DOWN, '[2, (8, 10)]': DOWN, '[2, (8, 11)]': DOWN,
        '[2, (9, 7)]': DOWN, '[2, (9, 8)]': DOWN, '[2, (9, 9)]': DOWN, '[2, (9, 10)]': DOWN, '[2, (9, 11)]': DOWN,
        '[2, (10, 7)]': LEFT, '[2, (10, 8)]': LEFT, '[2, (10, 9)]': LEFT, '[2, (10, 10)]': LEFT, '[2, (10, 11)]': LEFT,
        '[2, (11, 7)]': UP, '[2, (11, 8)]': UP, '[2, (11, 9)]': UP, '[2, (11, 10)]': UP, '[2, (11, 11)]': UP,

        # ROOM 3
        '[3, (7, 1)]': RIGHT, '[3, (7, 2)]': UP, '[3, (7, 3)]': LEFT, '[3, (7, 4)]': LEFT, '[3, (7, 5)]': LEFT,
        '[3, (8, 1)]': UP, '[3, (8, 2)]': UP, '[3, (8, 3)]': UP, '[3, (8, 4)]': UP, '[3, (8, 5)]': UP,
        '[3, (9, 1)]': UP, '[3, (9, 2)]': UP, '[3, (9, 3)]': UP, '[3, (9, 4)]': UP, '[3, (9, 5)]': UP,
        '[3, (10, 1)]': UP, '[3, (10, 2)]': UP, '[3, (10, 3)]': UP, '[3, (10, 4)]': UP, '[3, (10, 5)]': UP,
        '[3, (11, 1)]': UP, '[3, (11, 2)]': UP, '[3, (11, 3)]': UP, '[3, (11, 4)]': UP, '[3, (11, 5)]': UP,

        '[0, (3, 6)]': RIGHT,
        '[0, (6, 2)]': UP,
        '[2, (7, 9)]': DOWN,
        '[3, (10, 6)]': LEFT

    }

    term_set = ['[0, (3, 6)]', '[2, (7, 9)]', '[3, (10, 6)]', '[0, (6, 2)]']  # indexed by room
    policy_selection = lambda policy, state: policy[state]

    def termination_condition(termination_set, state):
        index = int(state[1])
        # if state in self.termination_set:  # if stat in hall way then the termination_set changes by index of 1
        #     index = (index + 1) % 3
        if state == termination_set[index]:
            return True
        return False

    # termination_condition = lambda self, state: state == self.termination_set[int(state[1])]

    option1 = QOption(env.observation_space.n, policy_option_1,
                      term_set, policy_selection,
                      termination_condition, env.action_space.n)

    save_option('FourRoomsO1', option1)

    # Option 2 - anti clockwise hall way points

    policy_option_2 = {
        # ROOM 0
        '[0, (1, 1)]': DOWN, '[0, (1, 2)]': DOWN, '[0, (1, 3)]': DOWN, '[0, (1, 4)]': DOWN, '[0, (1, 5)]': DOWN,
        '[0, (2, 1)]': DOWN, '[0, (2, 2)]': DOWN, '[0, (2, 3)]': DOWN, '[0, (2, 4)]': DOWN, '[0, (2, 5)]': DOWN,
        '[0, (3, 1)]': DOWN, '[0, (3, 2)]': DOWN, '[0, (3, 3)]': DOWN, '[0, (3, 4)]': DOWN, '[0, (3, 5)]': DOWN,
        '[0, (4, 1)]': DOWN, '[0, (4, 2)]': DOWN, '[0, (4, 3)]': DOWN, '[0, (4, 4)]': DOWN, '[0, (4, 5)]': DOWN,
        '[0, (5, 1)]': RIGHT, '[0, (5, 2)]': DOWN, '[0, (5, 3)]': LEFT, '[0, (5, 4)]': LEFT, '[0, (5, 5)]': LEFT,

        # ROOM 1
        '[1, (1, 7)]': DOWN, '[1, (1, 8)]': DOWN, '[1, (1, 9)]': DOWN, '[1, (1, 10)]': DOWN, '[1, (1, 11)]': DOWN,
        '[1, (2, 7)]': DOWN, '[1, (2, 8)]': DOWN, '[1, (2, 9)]': DOWN, '[1, (2, 10)]': DOWN, '[1, (2, 11)]': DOWN,
        '[1, (3, 7)]': LEFT, '[1, (3, 8)]': LEFT, '[1, (3, 9)]': LEFT, '[1, (3, 10)]': LEFT, '[1, (3, 11)]': LEFT,
        '[1, (4, 7)]': UP, '[1, (4, 8)]': UP, '[1, (4, 9)]': UP, '[1, (4, 10)]': UP, '[1, (4, 11)]': UP,
        '[1, (5, 7)]': UP, '[1, (5, 8)]': UP, '[1, (5, 9)]': UP, '[1, (5, 10)]': UP, '[1, (5, 11)]': UP,
        '[1, (6, 7)]': UP, '[1, (6, 8)]': UP, '[1, (6, 9)]': UP, '[1, (6, 10)]': UP, '[1, (6, 11)]': UP,

        # ROOM 2
        '[2, (8, 7)]': RIGHT, '[2, (8, 8)]': RIGHT, '[2, (8, 9)]': UP, '[2, (8, 10)]': LEFT, '[2, (8, 11)]': LEFT,
        '[2, (9, 7)]': UP, '[2, (9, 8)]': RIGHT, '[2, (9, 9)]': RIGHT, '[2, (9, 10)]': RIGHT, '[2, (9, 11)]': DOWN,
        '[2, (10, 7)]': UP, '[2, (10, 8)]': LEFT, '[2, (10, 9)]': LEFT, '[2, (10, 10)]': LEFT, '[2, (10, 11)]': LEFT,
        '[2, (11, 7)]': UP, '[2, (11, 8)]': UP, '[2, (11, 9)]': UP, '[2, (11, 10)]': UP, '[2, (11, 11)]': UP,
        # ROOM 3
        '[3, (7, 1)]': DOWN, '[3, (7, 2)]': DOWN, '[3, (7, 3)]': DOWN, '[3, (7, 4)]': DOWN, '[3, (7, 5)]': DOWN,
        '[3, (8, 1)]': DOWN, '[3, (8, 2)]': DOWN, '[3, (8, 3)]': DOWN, '[3, (8, 4)]': DOWN, '[3, (8, 5)]': DOWN,
        '[3, (9, 1)]': DOWN, '[3, (9, 2)]': DOWN, '[3, (9, 3)]': DOWN, '[3, (9, 4)]': DOWN, '[3, (9, 5)]': DOWN,
        '[3, (10, 1)]': RIGHT, '[3, (10, 2)]': RIGHT, '[3, (10, 3)]': RIGHT, '[3, (10, 4)]': RIGHT,
        '[3, (10, 5)]': RIGHT,
        '[3, (11, 1)]': UP, '[3, (11, 2)]': UP, '[3, (11, 3)]': UP, '[3, (11, 4)]': UP, '[3, (11, 5)]': UP,

        '[0, (3, 6)]': LEFT,
        '[0, (6, 2)]': DOWN,
        '[2, (7, 9)]': UP,
        '[3, (10, 6)]': RIGHT

    }

    term_set = ['[0, (6, 2)]', '[3, (10, 6)]', '[2, (7, 9)]', '[0, (3, 6)]']  # indexed by room

    option2 = QOption(env.observation_space.n, policy_option_2,
                      term_set, policy_selection,
                      termination_condition, env.action_space.n)

    save_option('FourRoomsO2', option2)

    # return option1, option2


def view_option_policy(env, option):
    policy = option.policy
    env.render(drawArrows=True, policy=policy)


if __name__ == '__main__':
    env = gym.make("FourRooms-v1")
    create_options(env)

    # test load
    o = load_option('FourRoomsO2')
    view_option_policy(env, o)
